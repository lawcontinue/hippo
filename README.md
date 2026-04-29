# Hippo

Distributed LLM inference on Apple Silicon. Pipeline parallelism across dual Mac Minis, speculative decoding for single-machine speedup, and an OpenAI-compatible API.

[![CI](https://github.com/lawcontinue/hippo/workflows/CI/badge.svg)](https://github.com/lawcontinue/hippo/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Quick Start** · [Performance](#performance) · [Modes](#modes) · [API](#openai-compatible-api) · [Configuration](#configuration) · [Architecture](#architecture)

---

## Performance

Real numbers from dual Mac Mini M4 (16 GB each). No benchmarks cherry-picked.

| Model | Mode | Hardware | tok/s | Notes |
|-------|------|----------|-------|-------|
| Qwen3-4B | DFlash | Single machine | **47.8** | 4× single-machine speedup |
| Gemma-3-12B | Pipeline (Thunderbolt) | 2 machines | **8.3** | 2.4× faster than single-machine |
| Gemma-3-12B | Pipeline (Wi-Fi) | 2 machines | **7.0** | Works without Thunderbolt |
| Qwen3-4B | Standalone | Single machine | 12.0 | Baseline |
| Gemma-3-12B | Standalone | Single machine | 3.5 | Baseline |

## Modes

Hippo supports three inference modes, pick based on model size:

- **Standalone** — Single machine, vanilla MLX inference. For models that fit in RAM.
- **DFlash** — Single machine, speculative decoding via [DFlash](https://arxiv.org/abs/2602.06036). ~4× faster than standalone. For models under ~8B.
- **Pipeline** — Two machines, model split across Thunderbolt or Wi-Fi. For models too large for one machine (8-15B on 16 GB machines).

**Rule of thumb**: Small model → DFlash. Big model → Pipeline. Don't stack them (ADR-163: 16 GB can't hold shard + target + draft simultaneously).

## Quick Start

### Prerequisites

- Apple Silicon Mac (M1+)
- Python 3.11+
- [MLX](https://github.com/ml-explore/mlx) (`pip install mlx`)

### Pipeline mode (two machines)

**R1** (start first):

```bash
cd hippo/pipeline
./start.sh r1
```

**R0** (start second):

```bash
cd hippo/pipeline
./start.sh r0 --model gemma-3-12b --prompt "Explain quantum computing"
```

### DFlash mode (single machine)

```bash
./start.sh dflash --model qwen3-4b --prompt "Write a Python web server"
```

### Benchmark

```bash
./benchmark.sh 3 50 thunderbolt    # 3 runs, 50 tokens, Thunderbolt
./benchmark.sh 3 50 wifi           # 3 runs, 50 tokens, Wi-Fi
```

## OpenAI-Compatible API

Start the API server:

```bash
python hippo_api.py --config hippo.conf.yaml
```

Three endpoints, drop-in replacement for OpenAI SDK:

```bash
# List models
curl http://localhost:8002/v1/models

# Chat completion (streaming)
curl http://localhost:8002/v1/chat/completions \
  -H "Authorization: Bearer your-token" \
  -d '{"model":"gemma-3-12b","messages":[{"role":"user","content":"Hello"}]}'

# Health check
curl http://localhost:8002/health
```

Works with **Cursor**, **Open WebUI**, **Continue**, and any OpenAI SDK client — just change `base_url`.

### Web UI

```bash
python hippo_web.py --config hippo.conf.yaml
```

Gradio chat interface at `http://localhost:7860`.

## Configuration

`hippo.conf.yaml` drives everything:

```yaml
defaults:
  mode: standalone       # standalone | pipeline | dflash
  host: "0.0.0.0"
  port: 9998

models:
  qwen3-4b:
    repo: "Qwen/Qwen3-4B"
    precision: "bf16"
    size_gb: 7.8
    modes: [standalone, dflash]
    dflash:
      draft_repo: "Aryagm/dflash-draft-qwen3-4b"

  gemma-3-12b:
    repo: "google/gemma-3-12b-pt"
    precision: "qat-4bit"
    size_gb: 6.9
    modes: [standalone, pipeline]
    pipeline:
      shards: 2
      r0_layers: [0, 24]
      r1_layers: [25, 47]
```

Memory guard built in — refuses to load if estimated usage exceeds `RAM × safety_factor`.

## Architecture

```
R0 (Mac Mini 1)                    R1 (Mac Mini 2)
┌─────────────────┐                ┌─────────────────┐
│  Layers 0-23    │  hidden state  │  Layers 24-47   │
│  (prefill +     │ ──────────────>│  (forward +     │
│   decode loop)  │  Thunderbolt/  │   lm_head)      │
│                 │<─────────────── │                 │
│  sample token   │   top-k logits │                 │
└─────────────────┘                └─────────────────┘
```

### Why SD doesn't help Pipeline

Counter-intuitive but实测verified: speculative decoding (including DFlash) does **not** accelerate pipeline inference. The bottleneck is R0's forward pass (~100ms/step). SD saves time on sampling, but verification also requires R0 forward — so SD doesn't reduce forward passes. Net result: slower than baseline (4.3 tok/s vs 6.8 tok/s).

> Pipeline solves the **memory** problem. SD solves the **speed** problem. They're orthogonal.

### Memory budget (16 GB machines)

| Model | Mode | Per-machine | Margin | Verdict |
|-------|------|-------------|--------|---------|
| Gemma-3-12B | Pipeline | 3.5 GB | +4.1 GB | ✅ Comfortable |
| Qwen3-4B | DFlash | 8.8 GB | -1.2 GB | ❌ Needs 48 GB |
| Qwen3-4B | Standalone | 7.8 GB | -0.2 GB | ⚠️ Tight |
| Qwen3-8B | Pipeline | 7.8 GB | -0.2 GB | ⚠️ Tight |

## Project structure

```
hippo/
├── hippo_api.py          # OpenAI-compatible API server
├── hippo_web.py          # Gradio chat UI
├── hippo_cli.py          # Unified CLI (serve/benchmark/list-models)
├── hippo.conf.yaml       # Configuration (models × modes)
├── start.sh              # One-command launcher
├── pipeline/             # Core inference engine
│   ├── rank0.py          # R0: autoregressive generation
│   ├── rank1.py          # R1: persistent server
│   ├── model_ops.py      # MLX ops (RoPE, quantized linear)
│   ├── tcp_transport.py  # Thunderbolt/Wi-Fi transport
│   └── benchmark.sh      # Multi-run benchmark tool
├── api.py                # Legacy Ollama-compatible API
├── model_manager.py      # Legacy model lifecycle
└── tests/                # Test suite
```

## Roadmap

- [ ] Continuous batching
- [ ] Model hot-swap via API
- [ ] Qwen3-8B pipeline optimization
- [ ] Benchmark dashboard (Prometheus + Grafana)

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

## Credits

Built by [lawcontinue](https://github.com/lawcontinue) with help from the T-Mind agent family. Powered by [MLX](https://github.com/ml-explore/mlx).
