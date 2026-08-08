# hippo.eval — Evaluation Toolkit for AI Agent Quality Assessment

A modular evaluation framework combining deterministic rules, LLM enhancement, distribution drift detection, multi-dimensional metrics, and chaos engineering.

## Modules

| Module   | Purpose                                                        |
| -------- | ------------------------------------------------------------- |
| `fusion` | Rule-based safety net + LLM enhancement layer fusion evaluation. |
| `drift`  | Distribution drift detection via KL/JS divergence between eval sets and production traffic. |
| `matrix` | Five-dimension metric matrix (success, recovery, efficiency, safety, consistency) with Goodhart analysis. |
| `chaos`  | Fault injection framework for chaos engineering and recovery testing. |
| `harness`| Unified CLI integrating all modules.                           |

## Quick Start

```python
from hippo.eval import evaluate, DriftDetector, FaultInjector, FaultType, InjectionStrategy

# Rule + LLM fusion evaluation
result = evaluate("Text to evaluate...")
print(result.verdict)  # pass / needs_review / reject

# Drift detection
import numpy as np
detector = DriftDetector()
detector.fit_reference(np.array([0.2, 0.4, 0.6, 0.8]))
drift_result = detector.detect(np.array([0.3, 0.5, 0.7, 0.9]))
print(drift_result.is_drifted)  # True / False

# Chaos injection
injector = FaultInjector.create_enabled(InjectionStrategy.KNOWN)
step = TaskStep(step_id="s1", description="Process payment", tool="payment_api", params={})
injected = injector.inject(step, FaultType.NETWORK_TIMEOUT)
print(injected.simulate_agent_observation())
```

## Installation

```bash
pip install hippo-llm[eval]  # includes scipy
```

## License

MIT (following Hippo project license).
