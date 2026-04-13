# Hippo 🦛 Architecture Design

> Lightweight local LLM manager — Ollama's Python alternative

## Problem Statement

| Pain Point | Ollama | Hippo |
|---|---|---|
| Memory | Model stays loaded, no auto-unload | Auto-unload after N seconds idle |
| Transparency | Go binary, opaque | Python, fully auditable |
| Cold start | ~8.6s for 8B | Same backend (llama.cpp), but transparent lifecycle |

## System Architecture

```
┌─────────────────────────────────────────────┐
│                   CLI (hippo)               │
│  serve / list / run / pull / remove         │
└──────────────┬──────────────────────────────┘
               │ HTTP (localhost:11434)
┌──────────────▼──────────────────────────────┐
│              FastAPI Server                  │
│  /api/generate  /api/chat  /api/tags        │
│  /api/pull      /api/delete                 │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│            Model Manager                    │
│  ┌─────────────────────────────────────┐    │
│  │ Model Registry (in-memory dict)     │    │
│  │  loaded_models: {name: Llama}       │    │
│  │  last_used: {name: timestamp}       │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │ Auto-Unload (background thread)     │    │
│  │  checks every 30s, unloads idle>N   │    │
│  └─────────────────────────────────────┘    │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│         llama-cpp-python (Llama)            │
│         GGUF inference engine               │
└─────────────────────────────────────────────┘
```

## Module Layout

```
hippo/
├── __init__.py          # version
├── config.py            # YAML config loader
├── model_manager.py     # model lifecycle (load/unload/auto-cleanup)
├── api.py               # FastAPI routes (Ollama-compatible)
├── downloader.py        # HuggingFace GGUF downloader
└── cli.py               # CLI entry point (typer)
```

## Key Design Decisions

1. **Single process, in-process model store** — No external service dependency. Models live in a Python dict.
2. **Background thread for auto-unload** — Simple, no asyncio complexity. Checks every 30s.
3. **Ollama API compatible** — Drop-in replacement. Same routes, same JSON shapes.
4. **YAML config at `~/.hippo/config.yaml`** — Model dir, idle timeout, server port, default params.
5. **GGUF files in `~/.hippo/models/`** — Flat directory, filename = model identity.

## API Compatibility

| Endpoint | Method | Status |
|---|---|---|
| `/api/generate` | POST | ✅ MVP |
| `/api/chat` | POST | ✅ MVP |
| `/api/tags` | GET | ✅ MVP |
| `/api/pull` | POST | ✅ MVP |
| `/api/delete` | DELETE | ✅ MVP |

## Data Flow: Generate Request

```
POST /api/generate {"model": "llama3.2:3b", "prompt": "Hello"}
  → ModelManager.get(model_name)
    → if loaded: update last_used, return
    → if not: load GGUF, store in dict, return
  → llama.create_completion(prompt, stream=...)
  → if stream: SSE chunks
  → if not: single JSON response
  → Background thread later checks last_used
```

## Config Schema

```yaml
server:
  host: "127.0.0.1"
  port: 11434

models:
  dir: "~/.hippo/models"

idle_timeout: 300  # seconds

defaults:
  n_ctx: 4096
  n_gpu_layers: -1  # all layers on GPU if available
  temperature: 0.7
  max_tokens: 2048
```
