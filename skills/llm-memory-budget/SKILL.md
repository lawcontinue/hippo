---
name: llm-memory-budget
description: Calculate memory budget before loading LLM models (GGUF/safetensors). Prevents OOM crashes by checking if model fits in available RAM. Supports single-machine, dual-machine RPC, and MLX Pipeline Parallelism modes. Use when loading, downloading, or planning to run any LLM model locally or in distributed mode. Triggers on "memory budget", "will this model fit", "OOM risk", "can I run this model", "memory check before loading", "pipeline parallelism memory", "distributed inference memory".
---

# LLM Memory Budget Calculator

Prevents OOM crashes by calculating memory requirements **before** loading any LLM model.

## Why This Exists

Loading a 15.5GB model on a 16GB Mac causes OOM — even with RPC tensor splitting — because `llama-cpp-python` mmaps the **entire** GGUF file regardless of `tensor_split`. This skill catches that before the crash.

## Quick Usage

```bash
# Single machine
python3 scripts/memory_budget.py --model-size 15.5 --mode single

# Dual-machine llama.cpp RPC (deprecated — mmap still loads full file!)
python3 scripts/memory_budget.py --model-size 15.5 --mode dual-rpc --tensor-split 0.5,0.5

# MLX Pipeline Parallelism (recommended for distributed)
python3 scripts/memory_budget.py --model-size 15.5 --mode pipeline --world-size 2

# Auto-detect size from file
python3 scripts/memory_budget.py --model-path ~/.hippo/models/qwen3.6-35b-a3b-q3_k_m.gguf --mode pipeline --world-size 2

# JSON output
python3 scripts/memory_budget.py --model-size 4.6 --mode single --json
```

## Memory Models

### Mode 1: Single Machine

```
Peak = model_size × 1.2 + kv_cache + 1.5GB system
Available = free + inactive + speculative
```

### Mode 2: Dual-Machine llama.cpp RPC ⚠️ DEPRECATED

```
Peak on PRIMARY = model_size × 1.2 (full mmap!) + 1.5GB system
Peak on REMOTE = remote_ratio × model_size + 1.5GB system
```

**Critical**: `tensor_split` does NOT reduce the primary node's mmap peak. The entire GGUF file is memory-mapped regardless. This is an architectural limitation of llama.cpp's RPC implementation.

**Real-world proof**: 2026-04-19, Qwen3.6-35B (15.5GB) on dual 16GB Mac Mini M4 — OOM 4 times with every tensor_split ratio tested (50/50, 30/70, etc.)

### Mode 3: MLX Pipeline Parallelism ✅ RECOMMENDED

```
Peak per device = (model_size / world_size) × 1.3 + embedding/lm_head + 1.5GB system
```

**Why it works**: MLX + safetensors format loads only the assigned layers (`model.layers[start:end]`), NOT the full model. Each device only evals its portion — un-evaluated tensors stay lazy (no physical memory allocation).

**Real-world validation**: Exo (43K⭐) successfully runs 35B models on dual 8GB devices using this approach.

## Decision Matrix

| Available RAM | Model Size | Mode | Recommendation |
|--------------|-----------|------|---------------|
| > model × 1.3 | any | single | ✅ Single machine OK |
| > (model/world_size) × 1.3 | distributed | pipeline | ✅ MLX Pipeline OK |
| > model × 0.6 | < available | rpc | ⚠️ RPC risky — mmap peak = full file |
| < model / world_size | any | any | ❌ Even pipeline won't help — smaller quant |

## Quantization Size Reference

| Quantization | 7B | 14B | 35B | 70B |
|-------------|-----|------|------|------|
| Q2_K | 3GB | 6GB | 11GB | 22GB |
| Q3_K_M | 3.5GB | 7GB | 15.5GB | 31GB |
| Q4_K_M | 4.5GB | 9GB | 20GB | 40GB |
| Q5_K_M | 5.5GB | 11GB | 24GB | 48GB |
| Q8_0 | 8GB | 15GB | 35GB | 70GB |

## Recovery Tips (if OOM happens)

1. Stop Ollama: `brew services stop ollama` (frees ~0.8GB)
2. Use `llama-cli` instead of `llama-cpp-python` (lower Python overhead)
3. Use smaller quantization (Q3_K_M → IQ3_XXS saves ~20%)
4. Close other apps (browser, etc.)
5. Switch to MLX Pipeline Parallelism (distributes memory across machines)
