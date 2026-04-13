# Hippo 🦛

Lightweight local LLM manager — Ollama's Python alternative.

## Why Hippo?

- **Auto-unload**: Models free memory after N seconds idle (default 300s)
- **Transparent**: Pure Python, fully auditable
- **Ollama-compatible**: Drop-in API replacement

## Quick Start

```bash
# Install
pip install -e .

# Start server
hippo serve

# Pull a model (from HuggingFace GGUF)
hippo pull bartowski/Llama-3.2-3B-Instruct-GGUF

# List models
hippo list

# Run inference
hippo run llama-3.2-3b "What is the meaning of life?"

# Chat API (curl)
curl http://localhost:11434/api/chat -d '{
  "model": "llama-3.2-3b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/tags` | GET | List models |
| `/api/generate` | POST | Text completion |
| `/api/chat` | POST | Chat completion |
| `/api/pull` | POST | Download model |
| `/api/delete` | DELETE | Remove model |

## Configuration

`~/.hippo/config.yaml`:

```yaml
server:
  host: "127.0.0.1"
  port: 11434

models:
  dir: "~/.hippo/models"

idle_timeout: 300

defaults:
  n_ctx: 4096
  n_gpu_layers: -1
  temperature: 0.7
  max_tokens: 2048
```
