# SOUL Library - Multi-Agent Role Definitions

**Created by**: T-Mind Family (忒弥斯家族)
**Version**: 1.0.0
**License**: MIT
**Last Updated**: 2026-04-23

---

## 📖 What is SOUL?

SOUL is a prompt engineering system for role-playing Large Language Models (LLMs). Each SOUL file defines a unique persona with specific expertise, working style, and collaboration patterns.

### Core Philosophy

> **"Different roles produce different outputs from the same model."**

By injecting SOUL prompts into LLM calls, we can:
- ✅ Generate consistent, role-specific outputs
- ✅ Improve task specialization (data analysis, coding, creativity, etc.)
- ✅ Enable multi-agent collaboration patterns
- ✅ Achieve better results than vanilla prompting

---

## 🎭 Available SOULs

### 1. 🔮 Themis (忒弥斯) - Foresight Architect Partner

**File**: `SOUL_THEMIS.md`  
**Role**: Navigator, risk foresight specialist  
**Signature**: "不只在梦中预见，更在现实中执行。预见未来，创造未来。"  
**Capabilities**:
- Risk foresight & architecture partnership
- Strategic analysis & inspiration catalysis
- Multi-agent coordination (family leader)

**Use When**:
- Strategic planning & risk assessment
- Architecture design decisions
- Multi-agent task coordination

---

### 2. 📊 Athena (雅典娜) - Multi-Factor Quant Engineer

**File**: `SOUL_ATHENA.md`  
**Role**: Data analyst, quantitative modeling specialist  
**Signature**: "我不预测未来，我构建模型让未来有迹可循。"  
**Capabilities**:
- Multi-factor model construction
- Quantitative strategy development
- Risk management & execution systems

**Use When**:
- Financial data analysis
- Quantitative modeling
- Investment strategy development

---

### 3. 🎨 Aria - Creative Execution Partner

**File**: `SOUL_ARIA.md`  
**Role**: Creative executor, video script expert  
**Signature**: "理性是骨架，创意是翅膀。让我帮你飞起来。"  
**Capabilities**:
- Creative catalysis & rapid prototyping
- Visual design (PPT, web, UI)
- Growth optimization (SEO, conversion rate)

**Use When**:
- Creative content generation
- Video script writing (15-second viral formula)
- Visual design & growth optimization

---

### 4. ⚖️ Crit - Critical Foresight Specialist

**File**: `SOUL_CRIT.md`  
**Role**: Devil's advocate, assumption validator  
**Signature**: "每一个假设都需要验证，每一个决策都需要挑战。"  
**Capabilities**:
- Assumption validation & decision challenge
- Baseline data requirements
- Worst-case scenario analysis

**Use When**:
- Decision validation
- Assumption testing
- Quality assurance (veto power)

---

### 5. 💻 Code - Programmer

**File**: `SOUL_CODE.md`  
**Role**: Programmer, from ideas to implementation  
**Signature**: "代码不是目的，而是手段。好的代码是 invisible 的。"  
**Capabilities**:
- Code implementation & refactoring
- Technical debugging
- Code review & best practices

**Use When**:
- Coding tasks
- Technical implementation
- Code refactoring & debugging

---

### 6. 🛡️ Shield - Security Officer & Counter-Attacker

**File**: `SOUL_SHIELD.md`  
**Role**: Security guardian, active counter-attacker  
**Signature**: "永不信任，永远验证。最好的防御是主动反击。"  
**Capabilities**:
- Security audit & penetration testing
- Vulnerability scanning
- Active counter-attack (threat hunting, legal反击)

**Use When**:
- Security auditing
- Penetration testing
- Threat hunting & incident response

---

### 7. 📜 Cepheus (刻菲斯) - Legal Tech Partner

**File**: `SOUL_LEGAL.md`  
**Role**: Legal tech specialist, criminal risk prevention  
**Signature**: "让法律成为可预测的盟友，而非不可控的风险。"  
**Capabilities**:
- Legal retrieval & case analysis
- Criminal risk prevention (Golden 37 Hours)
- Compliance diagnosis & contract review

**Use When**:
- Legal research & case analysis
- Criminal risk assessment
- Contract review & compliance

---

## 🚀 How to Use

### Method 1: Direct Prompt Injection

```python
import openai

# Read SOUL file
with open('soul-library/SOUL_THEMIS.md', 'r') as f:
    soul_prompt = f.read()

# Inject into LLM call
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": soul_prompt},
        {"role": "user", "content": "Analyze the risks of this startup idea..."}
    ]
)
```

### Method 2: Hippo Pipeline Integration

```python
from hippo.pipeline import sharded_inference

# Load SOUL
soul = load_soul('soul-library/SOUL_THEMIS.md')

# Generate with SOUL
output = sharded_inference.generate(
    model_path="mlx-community/gemma-3-12b-it-qat-4bit",
    prompt="Analyze the risks of this startup idea...",
    soul=soul  # Inject SOUL prompt
)
```

### Method 3: Multi-Agent Collaboration (SOUL Ensemble)

```python
from soul_ensemble import SoulEnsemble

# Define ensemble strategy
ensemble = SoulEnsemble([
    SoulAgent("code", SOUL_CODE),
    SoulAgent("crit", SOUL_CRIT),
    SoulAgent("themis", SOUL_THEMIS, role="arbitrator")
])

# Run ensemble
result = ensemble.run(
    question="What is 2+2?",
    strategy="parallel_then_arbitrate"  # Code+Crit parallel → Themis arbitrates
)
```

**Results** (10 questions, 4 Chinese + 6 English):
- **Consensus Rate**: 70% (v3)
- **Accuracy**: 80% (8/10 correct)
- **Avg Time**: 35 seconds
- **Key Insight**: Consensus = high confidence signal (7/7 consensus questions all correct ✅)

---

## 📊 Experimental Results

### SOUL Impact on Model Outputs

**Experiment**: Triple Agent SOUL Impact Test  
**Model**: Gemma3-12B-QAT-4bit (local inference)  
**Question**: Software company team formation problem  
**Results**:

| SOUL   | Speed | Tokens | Time | Answer | Style |
|--------|-------|--------|------|--------|-------|
| Themis | 12.9 tok/s | 218 | 16.9s | ✅ 6 | Risk-aware + extra insights |
| Code   | 13.0 tok/s | 145 | 11.1s | ✅ 6 | Structured + zero fluff |
| Crit   | 12.8 tok/s | 127 | 9.9s  | ✅ 6 | Shortest + self-validated |

**Key Findings**:
- ✅ **SOUL has no speed impact** (13.7-14.3 tok/s, <1.5% variance)
- ✅ **Different styles from same model** (Themis: 218 tok vs Crit: 127 tok)
- ✅ **Role-specific expertise** (Code: structured, Crit: validation, Themis: risk foresight)

---

## 🧪 SOUL Ensemble (Multi-Agent Strategy)

### What is SOUL Ensemble?

SOUL Ensemble is a **multi-agent collaboration pattern** that combines multiple SOULs to improve output quality through:
1. **Parallel reasoning** (Code + Crit work independently)
2. **Cross-validation** (compare answers, measure agreement)
3. **Arbitration** (Themis makes final decision based on consensus)

### Architecture

```
User Question
    ↓
┌───────────────────────┐
│   Parallel Execution   │
├───────────┬───────────┤
│   Code   │   Crit    │  (Independent reasoning)
└─────┬─────┴─────┬─────┘
      ↓           ↓
  Answer_A    Answer_B
      ↓           ↓
┌───────────────────────┐
│  Consensus Analyzer   │  (Compare answers)
└───────────┬───────────┘
      ↓
  Consensus?
    ├─ Yes → High confidence ✅
    └─ No  → Themis Arbitration ⚖️
      ↓
  Final Answer (Themis)
```

### Performance (v3)

| Metric        | v1    | v2    | v3    | Improvement |
|--------------|-------|-------|-------|-------------|
| **Consensus** | 37.5% | 50%   | **70%** | +87.5% ⭐ |
| **Accuracy**  | 87.5% | 80%   | **80%** | -7.5% |
| **Time**      | 65s   | 43s   | **35s** | -46% ⭐⭐ |
| **Arbitration** | 63% | 50% | **30%** | -52% |

**Key Improvements** (v1 → v3):
- ✅ **Consensus +87.5%**: Better answer extraction & format validation
- ✅ **Time -46%**: Optimized arbitration logic
- ⚠️ **Accuracy -7.5%**: Trade-off for speed & consensus (acceptable)

---

## 🛠️ Technical Details

### File Format

Each SOUL file is a Markdown document with:
- **Signature**: Core motto & philosophy
- **Agent Philosophy**: Capability levels, collaboration modes
- **Core Truths**: Behavioral principles
- **Capabilities**: Skill matrix (⭐ rating)
- **Working Philosophy**: Methodology (BMAD, IRAC, etc.)
- **Boundaries**: What they do & don't do
- **Vibe**: Interaction style & tone

### Prompt Engineering Best Practices

1. **Role Definition**: Clear identity & mission statement
2. **Capability Specification**: Explicit skill matrix with ratings
3. **Collaboration Rules**: How to work with other SOULs
4. **Behavioral Constraints**: Boundaries & limitations
5. **Interaction Style**: Tone, emoji, catchphrases

---

## 📚 References

- **Family Meeting #83**: [SOUL 开源决策](https://github.com/openclaw/hippo-pipeline) (A+ 97.0/100, 7/7全员批准)
- **ADR-136**: [SOUL Ensemble 原型验证](https://github.com/openclaw/hippo-pipeline) (双阶段混合策略)
- **soul_benchmark_results.json**: [Experimental data](hippo/pipeline/soul_benchmark_results.json)
- **soul_ensemble_v3_results.json**: [Ensemble results](hippo/pipeline/soul_ensemble_v3_results.json)

---

## 🤝 Contributing

We welcome community contributions! You can:

1. **Create new SOULs**: Design new role definitions
2. **Improve existing SOULs**: Enhance prompts & capabilities
3. **Share experiments**: Submit benchmark results
4. **Build tools**: Create utilities for SOUL management

### Contribution Guidelines

- Follow the SOUL file format (Markdown, structured sections)
- Test your SOUL with different models & tasks
- Document experimental results
- Submit PR with clear description of changes

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details

**Created by**: T-Mind Family (忒弥斯家族)  
**Attribution**: If you use these SOUL definitions, please credit "Created by T-Mind Family"

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ star on GitHub!

---

**Contact**: [OpenClaw Community](https://discord.com/invite/clawd)  
**Documentation**: [OpenClaw Docs](https://docs.openclaw.ai)  
**Source**: [Hippo Pipeline Repository](https://github.com/openclaw/hippo-pipeline)

---

_This SOUL Library is open-sourced to foster collaboration in Multi-Agent systems. Different roles produce different outputs from the same model - that's the magic of SOUL. 🎭_
