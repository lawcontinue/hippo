# Hippo

Lightweight local LLM manager — Ollama's Python alternative.

[![CI](https://github.com/lawcontinue/hippo/workflows/CI/badge.svg)](https://github.com/lawcontinue/hippo/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Features](#features) · [Quick Start](#quick-start) · [API](#api-endpoints) · [Configuration](#configuration) · [Contributing](CONTRIBUTING.md)

---

## Why Hippo?

I built Hippo because Ollama's Go codebase was hard to debug and extend. I wanted something I could actually read and modify in production.

**Three things Hippo does differently:**

1. **Auto-unload** - Models free memory after N seconds idle (default 300s). No more manual unloading or OOM kills.
2. **Pure Python** - Fully auditable. If something breaks, you can actually read the code and fix it.
3. **Ollama-compatible** - Drop-in API replacement. Just change the base URL.

**Trade-offs:** Hippo is slower than Ollama (Python overhead), and doesn't support all features yet (no multi-GPU, no LoRA). If you need production-grade performance, stick with Ollama. If you need hackability, Hippo might work better.

---

## Features

- **Ollama-Compatible API**: Drop-in replacement (mostly)
- **Auto-Unload**: Models free memory after idle timeout
- **Pure Python**: Auditable, debuggable, modifyable
- **TUI Dashboard**: Real-time monitoring with Rich
- **Model Quantization**: 14 GGUF formats supported
- **Docker & CI**: Production deployment ready

---

## Quick Start

### Installation

```bash
# From PyPI (TODO - not published yet)
pip install hippo-llm

# From source
git clone https://github.com/deepsearch/hippo.git
cd hippo
pip install -e .
```

### Basic Usage

```bash
# Start the server
hippo serve

# Pull a model (from HuggingFace GGUF)
hippo pull bartowski/Llama-3.2-3B-Instruct-GGUF

# List models
hippo list

# Run inference
hippo run llama-3.2-3b "What is the meaning of life?"

# TUI dashboard (2s refresh)
hippo tui
```

### API Usage

```bash
# Chat API
curl http://localhost:11434/api/chat -d '{
  "model": "llama-3.2-3b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'

# Generate API
curl http://localhost:11434/api/generate -d '{
  "model": "llama-3.2-3b",
  "prompt": "Once upon a time..."
}'
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/tags` | GET | List models |
| `/api/generate` | POST | Text completion |
| `/api/chat` | POST | Chat completion |
| `/api/pull` | POST | Download model (auth required) |
| `/api/delete` | DELETE | Remove model (auth required) |
| `/api/version` | GET | Version info |

**Streaming:** All endpoints support Server-Sent Events (set `stream: true`).

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

## TUI Dashboard

```bash
hippo tui --refresh 2.0
```

Shows:
- Model list (status, size, family, quantization)
- 2-second auto-refresh
- Server health check
- Color-coded output

---

## Model Quantization

Convert between 14 GGUF formats:

```bash
# List formats
hippo quantize --list

# Convert
hippo quantize input.gguf output.gguf --format q4_k_m
```

**Supported:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16, F32

**Note:** Requires `llama-quantize` CLI tool (llama.cpp). Falls back to `llama-cpp-python` if missing.

---

## Docker

```bash
# Docker Compose (recommended)
docker-compose up

# Manual build
docker build -t hippo:latest .
docker run -p 11434:8000 -v ~/.hippo/models:/models hippo:latest
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Coverage
pytest --cov=hippo tests/

# Lint
ruff check hippo/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Architecture

```
hippo/
├── api.py              # FastAPI app, REST endpoints
├── model_manager.py    # Model lifecycle (load/unload/LRU)
├── downloader.py       # HuggingFace download
├── cli.py              # Typer CLI
├── tui.py              # Rich TUI dashboard
├── quantize.py         # Model conversion
└── config.py           # YAML config loader
```

**Design choices:**
- FastAPI for async performance
- llama.cpp for inference (via llama-cpp-python)
- Rich for TUI (better than curses)
- Typer for CLI (better than argparse)

---

## Roadmap

### v0.2.0
- Multi-GPU support
- LoRA adapter support
- Batch inference
- Prometheus metrics

### v0.3.0
- Distributed serving
- Model sharding
- Web UI

---

## Known Issues

- Windows compatibility needs testing
- Multi-GPU not yet supported
- Model quantization requires external `llama-quantize` tool

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

- GitHub: https://github.com/deepsearch/hippo
- Issues: https://github.com/deepsearch/hippo/issues
- Discussions: https://github.com/deepsearch/hippo/discussions
