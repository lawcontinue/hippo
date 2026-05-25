# Picking the right model for a 16GB GPU

I tested four models on an RTX 5060 Ti (16GB) over two days. 36 prompts covering code, math, Chinese writing, English writing, creative tasks, and technical analysis. Each prompt run at 2000 tokens to catch repetition issues.

Here's what I found — and what I'd recommend.

## The candidates

| Model | Quant | VRAM | tok/s | What it is |
|-------|-------|------|-------|-----------|
| Gemma4-E4B | Q4_K_M | 9.6GB | 90 | Dense 8B with MoE routing |
| Qwen3-30B-A3B | Q3_K_M | 14GB | 78 | MoE: 30B total, 3B active |
| Qwen3-8B | Q4_K_M | 5.2GB | 71 | Dense 8B, general purpose |
| Qwen3-14B | Q4_K_M | 9.3GB | 41 | Dense 14B, strongest quality |

## The decision tree

```
What do you need?
│
├─ Fastest possible response (chat, code completion, quick Q&A)
│  → Gemma4-E4B (90 tok/s)
│  Caveat: think=True by default, must pass think=False or content is empty
│
├─ Best overall quality (technical writing, analysis, structured output)
│  → Qwen3-14B (41 tok/s)
│  Consistent quality across all domains. Slow but reliable.
│
├─ Largest model that fits (need "big model" quality for complex reasoning)
│  → Qwen3-30B-A3B (78 tok/s)
│  MUST use with loop detection — 78% loop rate at Q3 without it.
│  With Hippo's loop detector: 0% loop rate.
│
└─ Balanced speed + quality (general coding, writing, everyday use)
   → Qwen3-8B (71 tok/s)
   Never the best at anything, never the worst. Safe default.
```

## Quality: the blind test

I ran 12 prompts through all four models, then compared outputs side by side without knowing which model generated which. Key findings:

**Code generation**: 14B wins on type annotations, edge cases, and documentation. 8B is close but occasionally misses error handling. E4B is fine for simple functions, struggles with complex logic.

**Chinese writing**: 14B produces the most natural, well-structured Chinese output. 30B is comparable but requires loop detection. 8B tends to be terse.

**Creative tasks**: E4B is surprisingly good — more varied vocabulary and unexpected phrasing. 14B is more conservative but more coherent. E4B wins on creativity, 14B wins on coherence.

**Math/reasoning**: 14B clearly ahead. Shows work, handles edge cases. 8B often skips steps. 30B is comparable to 14B when it's not looping.

## The stability problem nobody talks about

Speed benchmarks at 500 tokens are misleading. I ran the same tests at 2000 tokens:

| Model | Loops at 500 tok | Loops at 2000 tok |
|-------|-----------------|------------------|
| 8B | 0% | 25% |
| 14B | 0% | 8% |
| 30B-A3B | 0% | 78% |
| E4B | 0% | 0% |

The 30B MoE at Q3 is the worst offender by far. But even 8B and 14B show some looping at longer outputs. Only E4B stays clean.

This is why I built loop detection into Hippo. It runs on every output, adds zero measurable overhead, and catches both the severe MoE loops and the milder dense model loops.

## My daily setup

I use two models:

- **Default**: E4B at 90 tok/s for most things — fast enough that it feels like a cloud API
- **Deep reasoning**: 14B at 41 tok/s for anything that needs accuracy (code, analysis, Chinese writing)

I skip 30B-A3B despite its speed advantage because even with loop detection, the output quality isn't consistently better than 14B to justify the complexity. Your mileage may vary.

```bash
# With Hippo
hippo-pipeline serve --model gemma4-e4b --mode standalone    # daily driver
hippo-pipeline serve --model qwen3-14b --mode standalone     # when I need quality
```

Loop detection is on by default. I don't think about it.

---

*[Hippo](https://github.com/lawcontinue/hippo) — `pip install hippo-llm` — check model licenses for your use case*
