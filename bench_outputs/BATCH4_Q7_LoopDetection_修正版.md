# My model kept repeating itself: debugging MoE routing at Q3

I was running Qwen3-30B-A3B — a Mixture-of-Experts model with 30B total parameters but only 3B active per token. The whole point of MoE is efficiency: you get 30B-quality output while only computing 3B parameters per forward pass.

Except it kept looping. Same idea, rephrased, over and over. 78% of the time. This is the bug that led me to build Hippo's thinking loop detector.

## What repeating output actually looks like

This is a real output from Qwen3-30B-A3B Q3_K_M, prompted to explain binary search:

```
Binary search works by dividing the sorted array in half each time.
The algorithm compares the target with the middle element.
If the target is smaller, it searches the left half.
If the target is larger, it searches the right half.
This process continues until the target is found.
Binary search has O(log n) time complexity.
The key insight is that each step eliminates half the candidates.
Binary search requires the input to be sorted beforehand.
The algorithm repeatedly divides the search space in half.
Each comparison reduces the remaining candidates by half.
The process of halving continues until the element is found.
Binary search achieves logarithmic time by halving each time.
The fundamental operation is dividing the search space in two.
```

Every line after line 6 is a paraphrase of lines 1-5. Token-level, they're all different. Semantically, it's the same four ideas on loop.

## Why token-level repeat penalty doesn't catch this

Most inference engines apply `repeat_penalty` at the token level: if the model just generated token "search", penalize it next time. This works for exact word repetition.

But MoE loops don't repeat tokens. They repeat *semantic units*. "dividing the array in half" and "halving the search space" share no tokens, but mean the same thing. The model's routing network keeps activating the same expert group, which keeps producing the same conceptual cluster, just with different surface forms.

I filed [llama.cpp #21264](https://github.com/ggerganov/llama.cpp/issues/21264) requesting line-level repetition detection. It's an open feature request.

## The Jaccard approach (now in Hippo)

The fix turned out to be simpler than I expected. Instead of comparing tokens, compare lines as word sets. This is what Hippo's `loop_detector.py` does:

```python
_STOP_WORDS = frozenset({"the", "a", "an", "is", "are", ...})

def tokenize(text: str) -> set:
    words = text.lower().split()
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}

def jaccard(set_a: set, set_b: set) -> float:
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0
```

For each new line in the output:
1. Tokenize to a word set (minus stop words)
2. Compare against the last 20 lines using Jaccard similarity
3. If ≥3 lines exceed 0.7 similarity → loop detected

The parameters (window=20, threshold=3, similarity=0.7) were tuned empirically. Lower similarity catches more but risks false positives on naturally repetitive content (code, lists). Higher threshold misses short loops. These values worked across 30 test generations with zero false positives.

## Three actions on detection

Not all loops are the same. I implemented three responses:

- **escape**: Inject a redirect prompt (`"Moving on to the next point:"`) to nudge the model away from the loop. Best for creative generation where you want to keep going.

- **stop**: Terminate generation cleanly. Best for factual queries where the model has already answered and is just padding.

- **warn**: Log the detection but don't intervene. Useful for monitoring loop rates in production without affecting output.

## Why this happens specifically at Q3

MoE models use a gating network to route each token to a subset of experts. Qwen3-30B-A3B has 8 experts per layer, activating 2 per token.

At Q3 quantization (3-bit), the gating network has roughly 8 discrete values to work with. That's not enough precision to make fine-grained routing decisions. The network collapses into repeatedly selecting the same 2 experts, and those experts produce semantically similar output.

At Q4 (4-bit, 16 discrete values), the routing is more precise and the problem largely disappears. But Q4 for a 30B model doesn't fit in 16GB VRAM — that's why people reach for Q3 in the first place.

## The results

Same GPU (RTX 5060 Ti 16GB), same model (Qwen3-30B-A3B Q3_K_M):

| Metric | Without loop detection | With loop detection |
|--------|----------------------|-------------------|
| Loop rate | 7/9 prompts (78%) | 0/10 (0%) |
| Decode speed | 77.5 tok/s | 77.5 tok/s |
| Usable output | ~22% (rest is repetition) | 100% |

The loop detector adds negligible overhead — Jaccard on word sets is O(n) where n is window size (20 lines). No GPU computation involved.

## What I learned

1. **Benchmark at your actual output length.** At 500 tokens, zero loops. At 2000 tokens, 78%. Short tests give false confidence.

2. **Token-level and semantic-level repetition are different problems.** One is solved by `repeat_penalty`, the other needs meaning-level comparison.

3. **Q3 + MoE is a dangerous combination.** The quantization directly degrades the routing mechanism. It's not a model quality issue — it's an architecture-quantization interaction.

4. **Loop detection should be built into the inference layer, not bolted on after.** In Hippo, it runs on every token in the stream with negligible overhead. You don't have to think about it.

## Using it

Hippo ships with loop detection on by default for MoE models:

```bash
pip install hippo-llm
hippo-pipeline serve --model qwen3-30b-a3b-q3 --mode standalone
```

The detector runs transparently — you get clean output without configuring anything. If you want to customize the behavior (window size, threshold, action), pass options in the config. Three actions available: `escape` (redirect the model away from the loop), `stop` (terminate cleanly), `warn` (log only).

Source code is in `hippo/pipeline/loop_detector.py` — it's a single file, ~100 lines, no dependencies beyond Python stdlib. Read it, fork it, improve it.

---

*[Hippo](https://github.com/lawcontinue/hippo) — distributed LLM inference for consumer hardware.*
