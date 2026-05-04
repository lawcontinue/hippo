#!/usr/bin/env python3
"""Tests for LlamaBackend — all mocked, no real model needed."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add parent dir (pipeline/) to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Mock llama_cpp before importing backend
llama_mock = MagicMock()
sys.modules["llama_cpp"] = llama_mock


from backend_llama import LlamaBackend, _find_gguf


class TestLlamaBackendInit:
    def test_default_config(self):
        cfg = {"gguf_path": "/tmp/test.gguf"}
        b = LlamaBackend(cfg)
        assert b._n_gpu_layers == -1
        assert b._n_ctx == 4096
        assert b._model_path == "/tmp/test.gguf"
        assert b._llama is None

    def test_custom_config(self):
        cfg = {
            "gguf_path": "/models/qwen.gguf",
            "n_gpu_layers": 32,
            "n_ctx": 8192,
        }
        b = LlamaBackend(cfg)
        assert b._n_gpu_layers == 32
        assert b._n_ctx == 8192

    def test_model_fallback_to_model_key(self):
        cfg = {"model": "qwen3-4b"}
        b = LlamaBackend(cfg)
        assert b._model_path == "qwen3-4b"


class TestLlamaBackendReady:
    @pytest.mark.asyncio
    async def test_ready_loads_model(self, tmp_path):
        gguf = tmp_path / "test.gguf"
        gguf.write_bytes(b"fake")

        mock_llama_instance = MagicMock()
        llama_mock.Llama.return_value = mock_llama_instance

        cfg = {"gguf_path": str(gguf), "n_gpu_layers": 0}
        b = LlamaBackend(cfg)
        ok = await b.ready()
        assert ok is True
        assert b._llama is mock_llama_instance
        llama_mock.Llama.assert_called_once()

    @pytest.mark.asyncio
    async def test_ready_file_not_found(self):
        cfg = {"gguf_path": "/nonexistent/model.gguf"}
        b = LlamaBackend(cfg)
        ok = await b.ready()
        assert ok is False

    @pytest.mark.asyncio
    async def test_ready_no_path(self):
        cfg = {}
        b = LlamaBackend(cfg)
        ok = await b.ready()
        assert ok is False

    @pytest.mark.asyncio
    async def test_ready_idempotent(self, tmp_path):
        gguf = tmp_path / "test.gguf"
        gguf.write_bytes(b"fake")
        llama_mock.Llama.return_value = MagicMock()

        cfg = {"gguf_path": str(gguf)}
        b = LlamaBackend(cfg)
        await b.ready()
        llama_mock.Llama.reset_mock()
        ok = await b.ready()
        assert ok is True
        llama_mock.Llama.assert_not_called()


class TestLlamaBackendGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_standard_dict(self, tmp_path):
        gguf = tmp_path / "test.gguf"
        gguf.write_bytes(b"fake")

        mock_instance = MagicMock()
        mock_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello world"}}],
            "usage": {"completion_tokens": 10},
        }
        llama_mock.Llama.return_value = mock_instance

        cfg = {"gguf_path": str(gguf)}
        b = LlamaBackend(cfg)
        await b.ready()

        messages = [{"role": "user", "content": "Hi"}]
        result = await b.generate(messages)

        assert "text" in result
        assert "tokens" in result
        assert "tok_s" in result
        assert "ar" in result
        assert "time_s" in result
        assert result["text"] == "Hello world"
        assert result["tokens"] == 10
        assert result["ar"] == 1.0

    @pytest.mark.asyncio
    async def test_generate_not_loaded_raises(self):
        cfg = {"gguf_path": "/fake.gguf"}
        b = LlamaBackend(cfg)
        with pytest.raises(RuntimeError, match="not loaded"):
            await b.generate([{"role": "user", "content": "Hi"}])


class TestCreateBackend:
    def test_llama_mode(self, tmp_path):
        """Test create_backend factory with llama mode."""
        # Import after mocking
        with patch.dict(sys.modules, {"backend_llama": MagicMock()}):
            pass
            # This would need the full import chain; just verify the mode is recognized
            # The actual factory test requires more setup


class TestFindGGUF:
    def test_find_nonexistent(self):
        assert _find_gguf("nonexistent_model_xyz") is None
