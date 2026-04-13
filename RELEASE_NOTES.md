# Hippo v0.1.0 Release Notes

**Release Date:** 2026-04-13

First public release. This is the MVP I've been using for local LLM management.

---

## What's New

### Core Functionality
- **Ollama-Compatible API** - Drop-in replacement (mostly)
- **Model Management** - Pull, list, show, delete GGUF models from HuggingFace
- **Inference** - Text and chat completion APIs
- **Streaming** - Server-sent events for real-time responses
- **Memory Management** - Auto-unload after idle (default 300s) + LRU eviction

### APIs Implemented

- `POST /api/generate` - Text completion
- `POST /api/chat` - Chat completion
- `GET /api/tags` - List models
- `POST /api/pull` - Download model (auth required)
- `DELETE /api/delete` - Remove model (auth required)
- `GET /api/version` - Version info

### CLI Commands

```bash
hippo serve          # Start server
hippo pull <model>    # Download model
hippo list            # List models
hippo show <model>    # Show model details
hippo run <model>     # Run inference
hippo tui             # TUI dashboard
hippo quantize        # Convert model format
```

### TUI Dashboard
- Real-time model monitoring (2s refresh)
- Shows: status, size, family, quantization
- Server health check
- Usage: `hippo tui --refresh 2.0`

### Model Quantization
- 14 formats supported (Q2_K through F32)
- Uses `llama-quantize` CLI tool (priority)
- Falls back to `llama-cpp-python` if missing
- Usage: `hippo quantize input.gguf output.gguf --format q4_k_m`

---

## Installation

### From PyPI
```bash
pip install hippo-llm
```

*(Note: PyPI publishing TODO - not yet published)*

### From Source
```bash
git clone https://github.com/deepsearch/hippo.git
cd hippo
pip install -e .
```

### Docker
```bash
docker-compose up
```

---

## Configuration

Create `~/.hippo/config.yaml`:

```yaml
server:
  host: "127.0.0.1"
  port: 11434

models:
  dir: "~/.hippo/models"

idle_timeout: 300  # seconds

defaults:
  n_ctx: 4096
  n_gpu_layers: -1
  temperature: 0.7
  max_tokens: 2048

api_key: "your-secret-key"  # Optional: for pull/delete operations
```

---

## Testing

```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/test_integration.py -v

# Coverage
pytest --cov=hippo tests/
```

**Test coverage:** 30/30 tests passing (100%)

---

## Stats

- **Core code:** ~1,500 lines
- **Test code:** ~350 lines
- **Python versions:** 3.11, 3.12, 3.13, 3.14
- **Dependencies:** 7 core (FastAPI, llama-cpp-python, PyYAML, Typer, Rich, requests, uvicorn)

---

## Development History

### MVP (v0.0.1)
Basic Ollama API compatibility:
- Model management (pull, list, delete)
- Text and chat completion
- Streaming responses

### P2 Optimizations
Performance and stability fixes:
- Download log sampling (every 10%)
- `/api/tags` pre-caching (faster startup)
- Idle unload race condition fix
- FastAPI lifespan migration (0 deprecation warnings)
- MIT License

### P3 Production Features
Production-ready features:
- Integration tests (13 scenarios, 30/30 passing)
- asyncio pre-caching migration
- TUI dashboard (Rich-based)
- GitHub Actions CI (Python 3.11-3.14 matrix)
- Docker multi-stage build
- Model quantization support (14 formats)

---

## Roadmap

### v0.2.0 (Planned)
- Multi-GPU support
- LoRA adapter support
- Batch inference
- Prometheus metrics

### v0.3.0 (Future)
- Distributed serving
- Model sharding
- Web UI

---

## Known Issues

- Model quantization requires external `llama-quantize` CLI tool
- Windows compatibility untested
- Multi-GPU not supported yet

---

## Acknowledgments

- [Ollama](https://github.com/ollama/ollama) - API specification
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Inference engine
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Community

- **GitHub:** https://github.com/deepsearch/hippo
- **Issues:** https://github.com/deepsearch/hippo/issues
- **Discussions:** https://github.com/deepsearch/hippo/discussions
