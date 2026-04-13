"""FastAPI server — Ollama-compatible API."""

import os
import time
import json
import struct
import hashlib
import logging
import asyncio
import threading
from contextlib import asynccontextmanager

import urllib.parse
import urllib.request

from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse

from hippo.config import HippoConfig
from hippo.model_manager import ModelManager
from hippo.downloader import pull_model

logger = logging.getLogger("hippo")

# Concurrency tracking
_active_requests = 0
_active_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: startup logging + pre-cache, shutdown cleanup."""
    # Startup
    log_dir = os.path.expanduser("~/.hippo")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, "hippo.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logging.getLogger("hippo").addHandler(file_handler)
    logger.info("Hippo server started")

    # Pre-cache model families in background asyncio task
    manager = getattr(app.state, "manager", None)
    if manager:
        async def _precache():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _precache_sync, manager)

        def _precache_sync(mgr):
            for p in mgr.config.models_dir.rglob("*.gguf"):
                try:
                    mgr._detect_family(p)
                except Exception:
                    pass
            logger.info("Model family pre-caching complete")

        asyncio.create_task(_precache())

    yield

    # Shutdown
    if manager:
        manager.stop_cleanup_thread()
        manager.unload_all()
        logger.info("Hippo server shutdown complete")


app = FastAPI(title="Hippo 🦛", version="0.1.0", lifespan=lifespan)

# Concurrency tracking
_active_requests = 0
_active_lock = threading.Lock()


def _get_config(request: Request) -> HippoConfig:
    return request.app.state.config


def _get_manager(request: Request) -> ModelManager:
    return request.app.state.manager


async def _check_auth(request: Request):
    """Verify API key for write operations if HIPPO_API_KEY is set."""
    cfg = _get_config(request)
    api_key = cfg.api_key
    if not api_key:
        return  # No key configured, skip auth

    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        if token == api_key:
            return

    return JSONResponse(
        {"error": "Unauthorized: valid API key required"},
        status_code=401,
    )


async def _check_concurrency(request: Request):
    """Reject requests if over concurrency limit."""
    global _active_requests
    cfg = _get_config(request)
    limit = cfg.max_concurrent_requests
    if limit <= 0:
        return
    with _active_lock:
        if _active_requests >= limit:
            return JSONResponse({"error": "Server busy, try again later"}, status_code=503)
        _active_requests += 1


@app.middleware("http")
async def concurrency_middleware(request: Request, call_next):
    result = await _check_concurrency(request)
    if result:
        return result

    response = await call_next(request)

    global _active_requests
    cfg = _get_config(request)
    if cfg.max_concurrent_requests > 0:
        with _active_lock:
            _active_requests -= 1

    return response




def _read_gguf_metadata_fast(path, max_keys=None):
    """Read GGUF metadata without loading the model (fast, <1ms).

    Parses the GGUF header directly to extract key-value metadata pairs.
    This avoids the expensive llama_cpp.Llama() constructor which loads
    the entire model into memory just to read metadata.

    GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    """
    GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12

    try:
        with open(path, "rb") as f:
            # Read header: magic(4) + version(4) + n_tensors(8) + n_kv(8)
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                return {}

            version = struct.unpack("<I", f.read(4))[0]
            if version >= 3:
                n_tensors = struct.unpack("<Q", f.read(8))[0]
            else:
                n_tensors = struct.unpack("<I", f.read(4))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def read_string():
                length = struct.unpack("<Q", f.read(8))[0]
                return f.read(length).decode("utf-8", errors="replace")

            def read_value(vtype):
                if vtype == GGUF_TYPE_UINT8:
                    return struct.unpack("<B", f.read(1))[0]
                elif vtype == GGUF_TYPE_INT8:
                    return struct.unpack("<b", f.read(1))[0]
                elif vtype == GGUF_TYPE_UINT16:
                    return struct.unpack("<H", f.read(2))[0]
                elif vtype == GGUF_TYPE_INT16:
                    return struct.unpack("<h", f.read(2))[0]
                elif vtype == GGUF_TYPE_UINT32:
                    return struct.unpack("<I", f.read(4))[0]
                elif vtype == GGUF_TYPE_INT32:
                    return struct.unpack("<i", f.read(4))[0]
                elif vtype == GGUF_TYPE_FLOAT32:
                    return struct.unpack("<f", f.read(4))[0]
                elif vtype == GGUF_TYPE_BOOL:
                    return struct.unpack("<B", f.read(1))[0] != 0
                elif vtype == GGUF_TYPE_STRING:
                    return read_string()
                elif vtype == GGUF_TYPE_UINT64:
                    return struct.unpack("<Q", f.read(8))[0]
                elif vtype == GGUF_TYPE_INT64:
                    return struct.unpack("<q", f.read(8))[0]
                elif vtype == GGUF_TYPE_FLOAT64:
                    return struct.unpack("<d", f.read(8))[0]
                elif vtype == GGUF_TYPE_ARRAY:
                    elem_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    # Skip array data - we don't need it for metadata lookup
                    elem_sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                    if elem_type == 8:  # string array
                        for _ in range(arr_len):
                            slen = struct.unpack("<Q", f.read(8))[0]
                            f.read(slen)
                    elif elem_type in elem_sizes:
                        f.read(arr_len * elem_sizes[elem_type])
                    return None
                else:
                    return None

            metadata = {}
            for _ in range(n_kv):
                key = read_string()
                vtype = struct.unpack("<I", f.read(4))[0]
                value = read_value(vtype)
                if value is not None:
                    metadata[key] = value
                if max_keys and len(metadata) >= max_keys:
                    break

            return metadata
    except Exception:
        return {}


def _model_info(name: str, manager: ModelManager) -> dict:
    """Build Ollama-style model info dict."""
    model_path = manager._resolve_model_path(name)
    family = "unknown"
    quant = ""
    size_bytes = 0
    if model_path:
        family = manager._detect_family(model_path)
        size_bytes = model_path.stat().st_size
        # Extract quantization from filename (e.g., Q4_K_M)
        fname = model_path.name.upper()
        for q in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "F16", "F32"]:
            if q in fname:
                quant = q
                break

    # Quick digest (first 8 bytes of SHA256 of filename, not full file hash)
    digest = hashlib.sha256(name.encode()).hexdigest()[:64]

    return {
        "name": name,
        "model": name,
        "modified_at": "",
        "size": size_bytes,
        "digest": digest,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": "",
            "quantization_level": quant,
        },
    }


@app.get("/api/tags")
async def list_models(request: Request):
    """List available models (Ollama compatible)."""
    manager = _get_manager(request)
    models = manager.list_available()
    model_list = []
    for m in models:
        info = _model_info(m["name"], manager)
        info["size"] = int(m["size_gb"] * 1024**3)
        model_list.append(info)

    loaded_names = set(m["name"] for m in models)
    for entry in manager.list_loaded():
        name = entry["name"]
        if name not in loaded_names:
            model_list.append(_model_info(name, manager))

    return {"models": model_list}


@app.post("/api/generate")
async def generate(request: Request):
    """Generate completion (Ollama compatible)."""
    # Auth check for write-like inference (keep consistent with Ollama)
    config = _get_config(request)
    manager = _get_manager(request)
    body = await request.json()
    model_name = body.get("model", "")
    prompt = body.get("prompt", "")
    stream = body.get("stream", True)
    options = body.get("options", {})

    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    repeat_penalty = options.get("repeat_penalty", config.defaults.repeat_penalty)
    params = {
        "max_tokens": options.get("num_predict", config.defaults.max_tokens),
        "temperature": options.get("temperature", config.defaults.temperature),
        "top_p": options.get("top_p", 0.9),
        "repeat_penalty": repeat_penalty,
    }

    if not stream:
        t0 = time.time_ns()
        result = llama.create_completion(prompt=prompt, stream=False, **params)
        elapsed = time.time_ns() - t0
        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
            "response": result["choices"][0]["text"],
            "done": True,
            "done_reason": "stop",
            "context": [],
            "total_duration": elapsed,
        }

    def stream_generate():
        t0 = time.time_ns()
        try:
            full_response = ""
            for chunk in llama.create_completion(prompt=prompt, stream=True, **params):
                text = chunk["choices"][0].get("text", "")
                full_response += text
                yield json.dumps({
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                    "response": text,
                    "done": False,
                }) + "\n"

            elapsed = time.time_ns() - t0
            yield json.dumps({
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                "response": "",
                "done": True,
                "done_reason": "stop",
                "context": [],
                "total_duration": elapsed,
            }) + "\n"
        except Exception as e:
            logger.exception("Stream generate error")
            yield json.dumps({
                "model": model_name,
                "error": str(e),
                "done": True,
                "done_reason": "error",
            }) + "\n"

    return StreamingResponse(stream_generate(), media_type="application/x-ndjson")


@app.post("/api/chat")
async def chat(request: Request):
    """Chat completion (Ollama compatible)."""
    config = _get_config(request)
    manager = _get_manager(request)
    body = await request.json()
    model_name = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", True)
    options = body.get("options", {})

    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    repeat_penalty = options.get("repeat_penalty", config.defaults.repeat_penalty)
    params = {
        "max_tokens": options.get("num_predict", config.defaults.max_tokens),
        "temperature": options.get("temperature", config.defaults.temperature),
        "repeat_penalty": repeat_penalty,
    }

    if not stream:
        t0 = time.time_ns()
        result = llama.create_chat_completion(messages=messages, stream=False, **params)
        msg = result["choices"][0]["message"]
        elapsed = time.time_ns() - t0
        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
            "message": msg,
            "done": True,
            "done_reason": "stop",
            "total_duration": elapsed,
        }

    def stream_chat():
        try:
            for chunk in llama.create_chat_completion(messages=messages, stream=True, **params):
                delta = chunk["choices"][0].get("delta", {})
                yield json.dumps({
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                    "message": delta,
                    "done": False,
                }) + "\n"

            yield json.dumps({
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            }) + "\n"
        except Exception as e:
            logger.exception("Stream chat error")
            yield json.dumps({
                "model": model_name,
                "error": str(e),
                "done": True,
                "done_reason": "error",
            }) + "\n"

    return StreamingResponse(stream_chat(), media_type="application/x-ndjson")


@app.post("/api/pull")
async def pull(request: Request):
    """Pull (download) a model from HuggingFace."""
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    config = _get_config(request)
    body = await request.json()
    name = body.get("name", "")

    try:
        path = pull_model(name, config.models_dir)
        return {"status": "success", "path": str(path)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/delete")
async def delete(request: Request):
    """Delete a model from disk."""
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    manager = _get_manager(request)
    body = await request.json()
    name = body.get("name", "")

    manager.unload(name)

    model_path = manager._resolve_model_path(name)
    if model_path and model_path.exists():
        model_path.unlink()
        return {"status": "success"}
    return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)


@app.post("/api/show")
async def show_model(request: Request):
    """Show model details (Ollama compatible)."""
    manager = _get_manager(request)
    body = await request.json()
    name = body.get("name", "") or body.get("model", "")

    model_path = manager._resolve_model_path(name)
    if not model_path:
        return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)

    family = manager._detect_family(model_path)
    size_bytes = model_path.stat().st_size

    # Extract quantization from filename (e.g., Q4_K_M)
    quant = "unknown"
    fname = model_path.name.upper()
    for q in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "F16", "F32"]:
        if q in fname:
            quant = q
            break

    # Read context length from GGUF metadata directly (fast, <1ms)
    # No need to load the model with llama_cpp
    n_ctx = 4096
    param_size = ""
    meta = _read_gguf_metadata_fast(model_path)
    if meta:
        n_ctx = int(meta.get("llama.context_length",
                   meta.get("default.context_length",
                   meta.get("general.context_length", 4096))))
        # Also try to get parameter count for parameter_size
        n_params = meta.get("llama.parameter_count",
                   meta.get("general.parameter_count", 0))
        if n_params:
            param_size = f"{n_params / 1e9:.1f}B"

    return {
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": param_size,
            "quantization_level": quant,
        },
        "model_info": {
            "general.architecture": family,
            "general.file_type": quant,
            "llama.context_length": n_ctx,
        },
        "modified_at": "",
        "type": "model",
        "size": size_bytes,
    }


@app.get("/api/search")
async def search_models(q: str = Query("", alias="q"), limit: int = Query(10)):
    """Search GGUF models on HuggingFace."""
    if not q:
        return {"models": []}

    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    url = f"{hf_endpoint}/api/models?search={urllib.parse.quote(q + ' gguf')}&limit={limit}&sort=downloads&direction=-1&filter=gguf"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hippo/0.1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read().decode())

        results = []
        for item in items[:limit]:
            results.append({
                "id": item.get("id", ""),
                "author": item.get("author", ""),
                "downloads": item.get("downloads", 0),
                "tags": item.get("tags", []),
                "url": f"{hf_endpoint}/{item.get('id', '')}",
            })
        return {"models": results}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/version")
async def version():
    """Return server version (Ollama compatible)."""
    return {"version": "0.1.0"}


@app.get("/v1/models")
async def list_models_openai(request: Request):
    """List models in OpenAI-compatible format."""
    manager = _get_manager(request)
    models = manager.list_available()
    data = []
    for m in models:
        data.append({
            "id": m["name"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "hippo",
        })
    for entry in manager.list_loaded():
        name = entry["name"]
        if not any(d["id"] == name for d in data):
            data.append({
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hippo",
            })
    return {"object": "list", "data": data}


@app.get("/")
async def root():
    return {"message": "Hippo 🦛 is running", "version": "0.1.0"}
