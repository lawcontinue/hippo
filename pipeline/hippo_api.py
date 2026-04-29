#!/usr/bin/env python3
"""
hippo_api.py — OpenAI-compatible HTTP API for Hippo Pipeline.

Endpoints:
  POST /v1/chat/completions   — Chat completion (streaming supported)
  GET  /v1/models             — List available models
  GET  /health                — Health check

Auth: Bearer token via Authorization header or ?key= query param.

Usage:
  python3 hippo_api.py --config hippo-api.conf.yaml
  python3 hippo_api.py --port 8080 --mode dflash --model qwen3-4b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid

import yaml

# aiohttp for lightweight async HTTP
try:
    from aiohttp import web
except ImportError:
    print("❌ aiohttp not installed. Run: pip install aiohttp")
    raise

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Config ─────────────────────────────────────────────

DEFAULT_API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8080,
    "api_keys": [],  # empty = no auth
    "mode": "standalone",
    "model": "qwen3-4b",
    "max_tokens": 2048,
    "timeout_s": 120,
}


def load_api_config(path: str | None = None) -> dict:
    cfg = dict(DEFAULT_API_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    # Also load pipeline config for model info
    pipeline_conf = os.path.join(SCRIPT_DIR, "hippo.conf.yaml")
    if os.path.exists(pipeline_conf):
        with open(pipeline_conf) as f:
            cfg["_pipeline"] = yaml.safe_load(f)
    return cfg


# ─── Backend ────────────────────────────────────────────

class HippoBackend:
    """Abstract backend for inference."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg.get("mode", "standalone")
        self.model_name = cfg.get("model", "qwen3-4b")
        self._ready = False

    async def ready(self) -> bool:
        return self._ready

    async def generate(self, messages: list[dict], max_tokens: int = 256,
                       temperature: float = 0.0, stream: bool = False):
        raise NotImplementedError


class DFlashBackend(HippoBackend):
    """DFlash single-machine backend."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.runner = None

    async def ready(self) -> bool:
        if self._ready:
            return True
        try:
            from rank0_dflash import R0DraftRunner
            pipeline_cfg = self.cfg.get("_pipeline", {})
            m = pipeline_cfg.get("models", {}).get(self.model_name, {})
            dflash_cfg = m.get("dflash", {})
            target = os.path.expanduser(f"~/.cache/modelscope/{m.get('repo', 'Qwen/Qwen3-4B')}")
            draft_repo = dflash_cfg.get("draft_repo", "z-lab/Qwen3-4B-DFlash-b16")
            draft = os.path.expanduser(f"~/.cache/modelscope/{draft_repo}")
            self.runner = R0DraftRunner(target_model=target, draft_model=draft)
            self._ready = True
            return True
        except Exception as e:
            print(f"❌ DFlash init failed: {e}")
            return False

    async def generate(self, messages: list[dict], max_tokens: int = 256,
                       temperature: float = 0.0, stream: bool = False):
        prompt = self._messages_to_prompt(messages)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.runner.generate(
                prompt=prompt, max_new_tokens=max_tokens, temperature=temperature
            )
        )
        return {
            "text": result.output_text,
            "tokens": result.accepted_tokens,
            "tok_s": result.accepted_tokens / max(result.total_time_s, 0.001),
            "ar": result.acceptance_rate,
            "time_s": result.total_time_s,
        }

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert OpenAI-style messages to a single prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        return "\n".join(parts)


class PipelineBackend(HippoBackend):
    """Pipeline dual-machine backend (R0 only — R1 must be running separately)."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)

    async def ready(self) -> bool:
        if self._ready:
            return True
        try:
            self._ready = True
            return True
        except Exception as e:
            print(f"❌ Pipeline init failed: {e}")
            return False

    async def generate(self, messages: list[dict], max_tokens: int = 256,
                       temperature: float = 0.0, stream: bool = False):
        prompt = "\n".join(m.get("content", "") for m in messages)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: asyncio.run(
                __import__("rank0").rank0_generate(
                    "0.0.0.0", 9998, prompt,
                    max_tokens=max_tokens, temperature=temperature,
                )
            )
        )
        return {
            "text": result,
            "tokens": 0,
            "tok_s": 0,
            "ar": 0,
            "time_s": 0,
        }


class StandaloneBackend(HippoBackend):
    """Standalone single-machine backend (no acceleration)."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)

    async def ready(self) -> bool:
        self._ready = True
        return True

    async def generate(self, messages: list[dict], max_tokens: int = 256,
                       temperature: float = 0.0, stream: bool = False):
        # TODO: implement standalone MLX generation
        return {
            "text": "[standalone mode not yet implemented]",
            "tokens": 0,
            "tok_s": 0,
            "ar": 0,
            "time_s": 0,
        }


def create_backend(cfg: dict) -> HippoBackend:
    mode = cfg.get("mode", "standalone")
    if mode == "dflash":
        return DFlashBackend(cfg)
    elif mode == "pipeline":
        return PipelineBackend(cfg)
    else:
        return StandaloneBackend(cfg)


# ─── HTTP Handlers ──────────────────────────────────────

def check_auth(request: web.Request, cfg: dict) -> bool:
    keys = cfg.get("api_keys", [])
    if not keys:
        return True  # no auth required
    # Check header
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:] in keys
    # Check query param
    key = request.query.get("key", "")
    return key in keys


async def handle_health(request: web.Request) -> web.Response:
    cfg: dict = request.app["cfg"]
    backend: HippoBackend = request.app["backend"]
    ready = await backend.ready()
    return web.json_response({
        "status": "ok" if ready else "loading",
        "mode": cfg.get("mode"),
        "model": cfg.get("model"),
    })


async def handle_models(request: web.Request) -> web.Response:
    if not check_auth(request, request.app["cfg"]):
        return web.json_response({"error": "unauthorized"}, status=401)

    pipeline_cfg = request.app["cfg"].get("_pipeline", {})
    models = []
    for name, m in pipeline_cfg.get("models", {}).items():
        models.append({
            "id": name,
            "object": "model",
            "owned_by": "hippo",
            "modes": m.get("modes", []),
        })
    return web.json_response({
        "object": "list",
        "data": models,
    })


async def handle_chat(request: web.Request) -> web.Response:
    cfg: dict = request.app["cfg"]
    backend: HippoBackend = request.app["backend"]

    if not check_auth(request, cfg):
        return web.json_response({"error": "unauthorized"}, status=401)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid JSON"}, status=400)

    messages = body.get("messages", [])
    if not messages:
        return web.json_response({"error": "messages is required"}, status=400)

    max_tokens = body.get("max_tokens", cfg.get("max_tokens", 256))
    temperature = body.get("temperature", 0.0)
    stream = body.get("stream", False)
    model = body.get("model", cfg.get("model", "qwen3-4b"))

    # Generate
    if not await backend.ready():
        return web.json_response({"error": "backend not ready"}, status=503)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if stream:
        return await _handle_chat_stream(request, backend, messages, max_tokens, temperature, request_id, created, model)

    result = await backend.generate(messages, max_tokens=max_tokens, temperature=temperature)

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": result["tokens"],
            "total_tokens": result["tokens"],
        },
        "_hippo": {
            "tok_s": round(result["tok_s"], 1),
            "ar": round(result["ar"], 3) if result["ar"] else None,
            "time_s": round(result["time_s"], 2),
            "mode": cfg.get("mode"),
        },
    }
    return web.json_response(response)


async def _handle_chat_stream(request, backend, messages, max_tokens, temperature, request_id, created, model):
    """SSE streaming response."""
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
    await resp.prepare(request)

    # For now, generate full text then stream it out chunk by chunk
    # (true streaming requires backend support — future work)
    result = await backend.generate(messages, max_tokens=max_tokens, temperature=temperature)
    text = result["text"]

    # Stream word by word
    words = text.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == 0 else f" {word}"
        data = json.dumps({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None,
            }],
        })
        await resp.write(f"data: {data}\n\n".encode())
        await asyncio.sleep(0.01)

    # Final chunk
    data = json.dumps({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    })
    await resp.write(f"data: {data}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


# ─── Server ─────────────────────────────────────────────

async def create_app(cfg: dict) -> web.Application:
    app = web.Application()
    app["cfg"] = cfg
    app["backend"] = create_backend(cfg)

    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat)

    return app


def main():
    parser = argparse.ArgumentParser(description="Hippo API Server — OpenAI-compatible")
    parser.add_argument("--config", default=None, help="API config YAML")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", default=None, choices=["standalone", "pipeline", "dflash"])
    parser.add_argument("--api-key", default=None, help="Set API key (or comma-separated for multiple)")
    args = parser.parse_args()

    cfg = load_api_config(args.config)

    # CLI overrides
    if args.host:
        cfg["host"] = args.host
    if args.port:
        cfg["port"] = args.port
    if args.model:
        cfg["model"] = args.model
    if args.mode:
        cfg["mode"] = args.mode
    if args.api_key:
        cfg["api_keys"] = [k.strip() for k in args.api_key.split(",")]

    host = cfg.get("host", "0.0.0.0")
    port = cfg.get("port", 8080)
    mode = cfg.get("mode", "standalone")
    model = cfg.get("model", "qwen3-4b")
    auth_status = "enabled" if cfg.get("api_keys") else "disabled"

    print("🦛 Hippo API Server")
    print(f"   Host: {host}:{port}")
    print(f"   Model: {model}")
    print(f"   Mode: {mode}")
    print(f"   Auth: {auth_status}")
    print("   Endpoints:")
    print("     POST /v1/chat/completions")
    print("     GET  /v1/models")
    print("     GET  /health")
    print()

    web.run_app(create_app(cfg), host=host, port=port, print=None)


if __name__ == "__main__":
    main()
