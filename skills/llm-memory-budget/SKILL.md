---
name: llm-memory-budget
description: Calculate memory budget before loading LLM models (GGUF/safetensors). Prevents OOM crashes by checking if model fits in available RAM. Use when loading, downloading, or planning to run any LLM model locally or in distributed mode. Triggers on "memory budget", "will this model fit", "OOM risk", "can I run this model", "memory check before loading".
---

# LLM Memory Budget Calculator

Prevents OOM crashes by calculating memory requirements **before** loading any LLM model.

## Why This Exists

Loading a 15.5GB model on a 16GB Mac causes OOM — even with RPC tensor splitting — because `llama-cpp-python` mmaps the **entire** GGUF file regardless of `tensor_split`. This skill catches that before the crash.

## Quick Usage

```bash
# Check if a model fits (single machine)
python3 scripts/memory_budget.py --model-size 15.5 --mode single

# Check dual-machine RPC split
python3 scripts/memory_budget.py --model-size 15.5 --mode dual-rpc --tensor-split 0.5,0.5

# Auto-detect size from file
python3 scripts/memory_budget.py --model-path ~/.hippo/models/qwen3.6-35b-a3b-q3_k_m.gguf --mode dual-rpc --tensor-split 0.3,0.7

# JSON output for programmatic use
python3 scripts/memory_budget.py --model-size 4.6 --mode single --json
```

## Key Rules

1. **mmap peak = full GGUF size** — `tensor_split` does NOT reduce loading peak. The entire file is memory-mapped first.
2. **Loading peak = GGUF size × 1.2 + KV cache + 1.5GB system** — 20% safety margin
3. **Steady state = local_ratio × GGUF + KV cache + 1.5GB** — after distribution
4. **Available = free + inactive** — macOS inactive pages are reclaimable

## Decision Matrix

| Available RAM | Model Size | Recommendation |
|--------------|-----------|---------------|
| > model × 1.3 | any | ✅ Single machine OK |
| > model × 0.6 | < available | 🟡 RPC split may work, stop other services first |
| < model | any | ❌ Even RPC won't help — use smaller quant |

## Recovery Tips (if OOM happens)

1. Stop Ollama: `brew services stop ollama` (frees ~0.8GB)
2. Use `llama-cli` instead of `llama-cpp-python` (lower Python overhead)
3. Use smaller quantization (Q3_K_M → IQ3_XXS saves ~20%)
4. Close other apps (browser, etc.)
