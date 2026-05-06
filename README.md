# Hippo 🦛

`pip install hippo-llm` | Python 3.9+ | MIT

Run 30B models on a ¥3800 GPU at 78 tok/s. Then chain machines together when you need to go bigger.

## 30-second setup

```bash
hippo-pipeline serve --model qwen3-30b-a3b-q3 --mode standalone
# → OpenAI-compatible API at localhost:8000/v1/chat/completions
```

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="qwen3-30b-a3b-q3",
    messages=[{"role": "user", "content": "Explain pipeline parallelism"}],
    max_tokens=500
)
print(r.choices[0].message.content)
```

<details>
<summary>Two-machine setup</summary>

```bash
# Machine 1
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 0

# Machine 2
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 1 \
  --coordinator http://192.168.1.10:9000
```

Split the model across machines. Run what doesn't fit on one GPU.

</details>

## The loop detection problem

Q3-quantized MoE models have a known issue: the routing network gets imprecise at 3-bit, picks the same experts over and over, and the output turns into repeating garbage. We measured this at **78% loop rate** on Qwen3-30B-A3B Q3_K_M.

Nobody was catching this because repeat penalties work on tokens, not semantics. The model says the same *thing* in different words, and token-level samplers let it through.

Hippo's loop detector runs Jaccard similarity on a sliding window of output lines — catches meaning-level repetition, not just token matches.

**Result on the same GPU, same model:**

| | With detection | Without |
|---|---|---|
| Effective speed | **78 tok/s** | ~7 tok/s usable (rest is junk) |
| Loop rate | **0%** | 78% |

Three actions when a loop is detected: `escape` (inject a redirect), `stop` (terminate), `warn` (log). Zero false positives across 30+ test runs.

## Benchmarks

RTX 5060 Ti 16GB, llama.cpp backend:

| Model | Quant | VRAM | tok/s |
|-------|-------|------|-------|
| Gemma4-E4B | Q4_K_M | 9.6GB | 90 |
| Qwen3-30B-A3B | Q3_K_M | 14GB | 78 |
| Qwen3-8B | Q4_K_M | 5.2GB | 71 |
| Qwen3-14B | Q4_K_M | 9.3GB | 41 |

Cloud equivalent (~$2/hr for 30B): a 5060 Ti breaks even at ~1,900 hours.

## What else it does

- **Pipeline parallelism** — split any HF model across N machines (Mac + PC mixed)
- **DFlash** — speculative decoding for Apple Silicon
- **Auto memory budget** — calculates shard splits from available VRAM
- **OpenAI-compatible API** — point existing tools at localhost

## When to use what

| Situation | Mode |
|-----------|------|
| Model fits on one GPU | standalone |
| Model doesn't fit | pipeline (2+ machines) |
| Mac, want raw speed | dflash |
| You're fine with cloud APIs | this isn't for you |

## Safety positioning

Hippo's loop detector and output controls are **L1 safety measures** — they constrain behavior, not intent. This means:

- ✅ We can detect and stop repetitive/degenerate outputs
- ✅ We can escape loops and retry with different parameters
- ❌ We cannot guarantee the model "understands" your safety requirements
- ❌ We cannot prevent a sufficiently capable model from generating harmful content if prompted

This is an honest limitation. Hippo makes local inference *usable* (catching the 78% loop rate that makes Q3 MoE models practically useless). It does not make models *safe* in any philosophical sense.

If you need production-grade content safety, layer a content filter **on top of** Hippo.

## License

MIT
