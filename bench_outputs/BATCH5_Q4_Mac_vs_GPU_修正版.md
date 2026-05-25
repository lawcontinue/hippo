# Mac vs GPU: real numbers from local LLM inference

I have two machines on my desk: a Mac Mini M4 (16GB unified memory) and an RTX 5060 Ti (16GB VRAM). I ran the same models on both.

The GPU is faster. That's not surprising. What *is* surprising: the Mac, with Hippo's DFlash speculative decoding, runs Qwen3-4B at 42 tok/s — faster than the 5060 Ti running Qwen3-14B at 41 tok/s.

A 4B model on a Mac beating a 14B model on a gaming GPU. That changes the calculus.

## The hardware

| | Mac Mini M4 | RTX 5060 Ti |
|---|---|---|
| Memory | 16GB unified | 16GB VRAM |
| Engine | MLX (Apple) | llama.cpp (CUDA) |
| Price | ~¥4000 | ~¥3800 |

Similar price. Completely different strengths.

## The numbers

**Mac Mini M4, MLX backend:**

| Model | Mode | tok/s |
|-------|------|-------|
| Qwen3-4B | standalone | 12 |
| Qwen3-4B | **DFlash** | **42** |
| Gemma-3-12B | standalone | 3.5 |
| Gemma-3-12B | pipeline (2 Macs) | 8.3 |

**RTX 5060 Ti, llama.cpp backend:**

| Model | Quant | tok/s |
|-------|-------|-------|
| Gemma4-E4B | Q4_K_M | 90 |
| Qwen3-30B-A3B | Q3_K_M | 78 |
| Qwen3-8B | Q4_K_M | 71 |
| Qwen3-14B | Q4_K_M | 41 |

## When to use which

**Use the Mac when:**
- You need a fast 4B model for code completion, chat, or quick tasks (42 tok/s with DFlash)
- You're already using the Mac as your daily driver — no extra hardware needed
- You want to experiment with speculative decoding (DFlash is genuinely interesting tech)

**Use the GPU when:**
- You need models larger than 8B (the Mac can technically run 12B but at 3.5 tok/s it's painful)
- You need consistent high speed across multiple model sizes
- You're doing batch inference or serving multiple users

**Use both (pipeline mode) when:**
- One machine can't fit the model (a 12B model at 3.5 tok/s on Mac → 8.3 tok/s with two Macs on Thunderbolt)
- You want to run 30B+ models without buying a data center GPU

## The DFlash surprise

DFlash is Hippo's speculative decoding mode for Apple Silicon. It runs a small draft model (Qwen3-0.6B) alongside the target model (Qwen3-4B), predicts 3-4 tokens ahead, and verifies in one forward pass.

Result: 12 tok/s → 42 tok/s. A 3.5x speedup. The draft model is small enough that it doesn't meaningfully impact memory.

But there's a constraint: DFlash and pipeline parallelism don't stack on 16GB. You need memory for the target model shard + the full draft model + MLX overhead. On 16GB, it's one or the other. On a 48GB Mac, you could do both.

## One tool, two engines

The practical takeaway: you don't need to learn two different tools. Hippo wraps both MLX and llama.cpp behind the same CLI:

```bash
# Mac — uses MLX automatically
hippo-pipeline serve --model qwen3-4b --mode dflash

# GPU — uses llama.cpp automatically
hippo-pipeline serve --model qwen3-14b --mode standalone

# Both machines — pipeline across Mac + PC
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 0
```

Same API format (OpenAI-compatible), same loop detection, same config structure. The engine selection is automatic based on detected hardware.

---

*[Hippo](https://github.com/lawcontinue/hippo) — `pip install hippo-llm`*
