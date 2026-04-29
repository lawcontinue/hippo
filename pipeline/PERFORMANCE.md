# Hippo Pipeline — Performance Reference

**Hardware**: Mac Mini M4 16GB × 2 (Wi-Fi / Thunderbolt)

## Performance Table

| Model | Precision | Mode | Hardware | tok/s | Prefill | Notes |
|-------|-----------|------|----------|-------|---------|-------|
| Qwen3-4B | BF16 | standalone | R0 only | ~12 | ~2s | Baseline, no acceleration |
| Qwen3-4B | BF16 | **dflash** | R0 only | **~42** | ~7s | 4.08x speedup, 70.6% AR |
| Gemma-3-12B | QAT-4bit | standalone | R0 only | ~3.5 | ~4s | Baseline |
| Gemma-3-12B | QAT-4bit | **pipeline** | R0 + R1 | **~7.0** | ~15s | Dual-machine, Wi-Fi |
| Gemma-3-12B | QAT-4bit | pipeline | R0 + R1 | **~8.3** | ~15s | Thunderbolt |
| Qwen3-8B | BF16 | pipeline | R0 + R1 | ~5.0 | ~15s | Estimated |

## Memory Budget (per machine, 16GB)

| Model | Mode | Model Size | Per-machine | Available | Margin | Status |
|-------|------|-----------|-------------|-----------|--------|--------|
| Qwen3-4B | dflash | 7.8 + 1.0 (draft) | 8.8 GB | 7.6 GB | -1.2 GB | ⚠️ Needs 48GB machine |
| Qwen3-4B | standalone | 7.8 GB | 7.8 GB | 7.6 GB | -0.2 GB | ⚠️ Tight |
| Gemma-3-12B | pipeline | 6.9 GB | 3.5 GB | 7.6 GB | 4.1 GB | ✅ Comfortable |
| Qwen3-8B | pipeline | 15.5 GB | 7.8 GB | 7.6 GB | -0.2 GB | ⚠️ Tight |

**Available = 16GB × 0.85 (safety) - 4GB (system) - 2GB (MLX overhead)**

## Key Findings

1. **DFlash + Pipeline cannot stack** on 16GB (needs shard + full target + draft = 15.4GB > 16GB)
2. **SD accelerates lm_head sampling, not transformer forward** — Pipeline bottleneck is forward, so SD doesn't help
3. **Thunderbolt vs Wi-Fi**: ~15% faster step time (120ms vs 139ms), same prefill
4. **DFlash acceptance rate**: 70.6% with block_size=16, avg acceptance length 11.29 tokens

## Architecture Decision

- **Small models (< 8B)**: Use DFlash single-machine (42 tok/s)
- **Medium models (8-15B)**: Use Pipeline dual-machine (7 tok/s)
- **Large models (> 15B)**: Need 32GB+ machines or cloud GPU for quantization

Source: ADR-163 (family meeting #143), ADR-161 (DFlash local reproduction)
