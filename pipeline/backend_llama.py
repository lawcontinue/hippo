#!/usr/bin/env python3
"""
backend_llama.py — LlamaBackend for Hippo Pipeline.

Uses llama-cpp-python to run GGUF models on Windows/Linux/macOS with GPU support.
"""

from __future__ import annotations

import asyncio
import glob
import os
import time


def _init_cuda_dlls():
    """Auto-discover CUDA DLLs on Windows (nvidia-cublas-cu12 pip package)."""
    if os.name != 'nt':
        return
    import site
    try:
        sp = site.getsitepackages()
        # Check user site-packages too
        user_sp = [site.getusersitepackages()] if hasattr(site, 'getusersitepackages') else []
        all_sp = sp + user_sp
    except Exception:
        return
    
    for base in all_sp:
        nvidia_dir = os.path.join(base, 'nvidia')
        if not os.path.isdir(nvidia_dir):
            continue
        for sub in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, sub, 'bin')
            if os.path.isdir(bin_dir) and any(f.endswith('.dll') for f in os.listdir(bin_dir)):
                try:
                    os.add_dll_directory(bin_dir)
                except (OSError, FileNotFoundError):
                    pass


def _find_gguf(model_name: str) -> str | None:
    """Search common directories for a GGUF file matching model_name."""
    search_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/modelscope"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        ".",
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True):
            base = os.path.basename(f).lower()
            name_lower = model_name.lower().replace("/", "_")
            if name_lower in base or model_name.lower() in base:
                return f
    return None


class LlamaBackend:
    """Backend using llama-cpp-python for GGUF model inference.

    Duck-type compatible with HippoBackend (same interface).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg.get("mode", "llama")
        self.model_name = cfg.get("model", "")
        self._ready = False
        self._llama = None
        self._model_path = cfg.get("gguf_path") or cfg.get("model", "")
        self._n_gpu_layers = int(cfg.get("n_gpu_layers", -1))
        self._n_ctx = int(cfg.get("n_ctx", 4096))
        self._thinking = cfg.get("thinking", False)

        # Loop detection for thinking models
        self._loop_detect = cfg.get("loop_detect", False)
        self._loop_detector = None
        if self._loop_detect:
            from loop_detector import LoopDetector
            self._loop_detector = LoopDetector(
                window=cfg.get("loop_detect_window", 20),
                threshold=cfg.get("loop_detect_threshold", 3),
                similarity=cfg.get("loop_detect_similarity", 0.7),
                action=cfg.get("loop_detect_action", "escape"),
            )

    async def ready(self) -> bool:
        if self._llama is not None:
            return True
        _init_cuda_dlls()
        try:
            from llama_cpp import Llama
        except ImportError:
            print("❌ llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return False

        # Resolve model path
        path = self._model_path
        if path and not os.path.isfile(path):
            found = _find_gguf(path)
            if found:
                path = found
            else:
                print(f"❌ GGUF model not found: {self._model_path}")
                return False

        if not path:
            print("❌ No model path specified. Use --gguf-path or --model.")
            return False

        if not os.path.isfile(path):
            print(f"❌ GGUF file not found: {path}")
            return False

        print(f"🔄 Loading GGUF model: {path} (n_gpu_layers={self._n_gpu_layers}, n_ctx={self._n_ctx})")
        loop = asyncio.get_event_loop()
        self._llama = await loop.run_in_executor(
            None,
            lambda: Llama(
                model_path=path,
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=self._n_ctx,
                verbose=False,
            ),
        )
        print(f"✅ Model loaded: {os.path.basename(path)}")
        self._ready = True
        return True

    async def generate(self, messages: list[dict], max_tokens: int = 256,
                       temperature: float = 0.0, stream: bool = False):
        if not self._llama:
            raise RuntimeError("Model not loaded. Call ready() first.")

        loop = asyncio.get_event_loop()
        t0 = time.time()

        if stream:
            return self._generate_stream(messages, max_tokens, temperature, t0)

        # Disable Qwen3 thinking mode by appending /no_think to last user message
        if "qwen3" in self.model_name.lower() and not self._thinking:
            messages = list(messages)  # shallow copy
            if messages and messages[-1].get("role") == "user":
                messages[-1] = {**messages[-1], "content": messages[-1]["content"].rstrip() + " /no_think"}

        result = await loop.run_in_executor(
            None,
            lambda: self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        elapsed = time.time() - t0
        choice = result.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = result.get("usage", {})
        tokens = usage.get("completion_tokens", 0)

        # Post-hoc loop detection for non-streaming mode
        if self._loop_detector:
            loop_result = self._loop_detector.check_text(text)
            if loop_result:
                print(f"[LOOP] Loop detected in output: {loop_result['line'][:60]}... (action={loop_result['action']})")
            self._loop_detector.reset()

        return {
            "text": text,
            "tokens": tokens,
            "tok_s": tokens / max(elapsed, 0.001),
            "ar": 1.0,
            "time_s": elapsed,
        }

    async def _generate_stream(self, messages, max_tokens, temperature, t0):
        """Yield chunks as an async generator."""
        loop = asyncio.get_event_loop()

        # Disable Qwen3 thinking mode by appending /no_think to last user message
        if "qwen3" in self.model_name.lower() and not self._thinking:
            messages = list(messages)  # shallow copy
            if messages and messages[-1].get("role") == "user":
                messages[-1] = {**messages[-1], "content": messages[-1]["content"].rstrip() + " /no_think"}

        def _stream_sync():
            return self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

        chunks = await loop.run_in_executor(None, _stream_sync)
        token_count = 0
        for chunk in chunks:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                token_count += 1

                # Streaming loop detection
                if self._loop_detector:
                    loop_result = self._loop_detector.feed(content)
                    if loop_result and loop_result.get("loop"):
                        action = loop_result.get("action", "warn")
                        print(f"[LOOP] Loop detected: {loop_result['line'][:60]}... (action={action})")
                        if action == "stop":
                            break
                        # action="escape" or "warn": continue generating

                yield {
                    "text": content,
                    "tokens": token_count,
                    "tok_s": 0,
                    "ar": 1.0,
                    "time_s": time.time() - t0,
                }
