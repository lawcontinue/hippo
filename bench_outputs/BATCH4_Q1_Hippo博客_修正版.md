# Running a 30B model on a ¥3800 GPU: what I learned building Hippo

I wanted to run Qwen3-30B-A3B locally. Not on A100s, not on cloud instances — on a single RTX 5060 Ti I bought for gaming. So I built Hippo, an inference framework that solves the problems I ran into.

It took two days of debugging. Here's what happened.

## The hardware

RTX 5060 Ti, 16GB VRAM, bought for ¥3800. The 16GB version specifically — the 8GB model is useless for LLM inference. That's the single most important hardware decision: VRAM, not CUDA cores.

## First attempt: Ollama

```bash
ollama run qwen3-30b-a3b-q3
```

It loaded. It ran. And then it started repeating the same paragraph, over and over.

I tried 9 different prompts — code, math, creative writing, Chinese, English. 7 out of 9 produced looping output. The model would generate 3-4 coherent sentences, then get stuck repeating the same idea with slightly different wording.

Not a one-off. Consistent, reproducible, 78% loop rate.

Turns out this is a known issue ([Ollama #10976](https://github.com/ollama/ollama/issues/10976)). Q3 quantization degrades the MoE gating router — at 3-bit precision, the router can't distinguish between experts well enough, keeps selecting the same group, and the output collapses into repetition.

## Second attempt: llama.cpp directly

I switched to llama-cpp-python. Same model file, same GPU.

```bash
pip install llama-cpp-python --extra-index-url https://ghfast.top/cu124
```

That install command took two hours. Windows doesn't have a C++ compiler by default, the official PyPI doesn't ship CUDA wheels, and HuggingFace is blocked in China. The solution: download the CUDA wheel from a mirror (`ghfast.top`), then install `nvidia-cublas-cu12` from Alibaba's mirror for DLL dependencies, then inject the DLL path with `os.add_dll_directory()`.

Two hours for one `pip install`. But once it ran:

```
Qwen3-30B-A3B Q3_K_M on RTX 5060 Ti 16GB:
- Decode speed: 77.5 tok/s
- 49/49 layers on GPU
- Loop rate: 0/10 prompts
```

Zero loops. Same model, same quantization, different inference engine. The sampling configuration matters more than I expected.

## Third attempt: building the loop detector into Hippo

Even with llama.cpp, I wasn't comfortable shipping Hippo without a safety net for MoE loops. So I built a line-level loop detector directly into the inference pipeline.

The idea: token-level `repeat_penalty` catches exact token repetition. But MoE loops are semantic — the model says the same thing in different words. You need to compare meaning, not tokens.

```python
# Core of Hippo's loop_detector.py — simplified
class LoopDetector:
    def __init__(self, window=20, threshold=3, similarity=0.7):
        self.window = window      # compare against last N lines
        self.threshold = threshold  # need this many matches to trigger
        self.similarity = similarity  # Jaccard threshold

    def feed(self, token_text):
        # Buffer tokens, flush on newline
        # For each complete line, Jaccard-compare against window
        # If >=threshold lines match above similarity → trigger
        ...
```

Jaccard similarity on word sets (minus stop words), sliding window of 20 lines, trigger after 3 matches above 0.7 similarity. When triggered: inject a redirect prompt to escape the loop, or stop generation cleanly.

Tested on 30 generations across different models and prompt types: zero false positives, zero missed loops. The Q3 MoE problem is fully mitigated.

## What Hippo does now

One command, and you're running 30B on a consumer GPU with loop protection:

```bash
hippo-pipeline serve --model qwen3-30b-a3b-q3 --mode standalone
# → OpenAI-compatible API at localhost:8000/v1/chat/completions
```

Under the hood: llama.cpp backend for reliable inference, built-in loop detection for MoE models, OpenAI-compatible API so your existing tools just work. If one GPU isn't enough, switch to pipeline mode and split the model across two machines.

```bash
# One machine
hippo-pipeline serve --model qwen3-30b-a3b-q3 --mode standalone

# Two machines
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 0  # main
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 1 --coordinator http://main:9000
```

## What I'd do differently

1. **Skip Ollama for MoE models at Q3 or below.** The default sampler configuration makes the gating problem worse. llama.cpp with direct sampling is more reliable.

2. **Budget 2× the time you think for Windows setup.** CUDA wheels, DLL paths, mirrors — every step has a surprise.

3. **Test at your actual output length.** I initially tested at 500 tokens, zero loops. Extended to 2000 tokens: 78% loop rate. Short benchmarks are misleading.

4. **16GB VRAM is the sweet spot for consumer inference.** 8B models at 71 tok/s, 14B at 41 tok/s, 30B MoE at 78 tok/s. The 8GB version of the same GPU can barely run 8B.

## The numbers

All on RTX 5060 Ti 16GB, llama-cpp-python with CUDA:

| Model | Quant | VRAM | tok/s |
|-------|-------|------|-------|
| Gemma4-E4B | Q4_K_M | 9.6GB | 90 |
| Qwen3-30B-A3B | Q3_K_M | 14GB | 78 |
| Qwen3-8B | Q4_K_M | 5.2GB | 71 |
| Qwen3-14B | Q4_K_M | 9.3GB | 41 |

Cloud cost for similar performance: ~$2/hour. The GPU pays for itself in 1,900 hours of inference.

---

*Hippo is available at [github.com/lawcontinue/hippo](https://github.com/lawcontinue/hippo) — `pip install hippo-llm`.*
