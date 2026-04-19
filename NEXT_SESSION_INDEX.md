# 下个会话索引 — 2026-04-19 Hippo Qwen3.6 双机实测

**创建时间**: 2026-04-19 13:22
**优先级**: P0 — Qwen3.6 双机实测

---

## 快速恢复要点

### 当前状态
- ✅ M1+M2 已完成并推 GitHub (commit a33d382)
- ✅ Qwen3.6-35B-A3B Q3_K_M (15.5GB) 已下载到两台机器
- ✅ 第二台 rpc-server 已验证可用
- ❌ 3 次 OOM 崩溃 — 16GB 内存不够加载 15.5GB 模型

### 关键发现
**llama-cpp-python 的 mmap 会映射整个 GGUF 文件，tensor_split 不减少加载峰值**
- 单机 100%: 需要 ~18GB → ❌
- 双机 50/50: 加载峰值仍 ~18GB → ❌（mmap 全文件）
- 双机 30/70: 加载峰值仍 ~18GB → ❌（mmap 全文件）

### 下一步方案（按优先级）

#### 方案 A: 用 llama-cli 替代 llama-cpp-python（推荐 ⭐）
llama-cli 内存效率更高（无 Python 开销），且可能使用不同的 tensor 加载策略：
```bash
# 先停 Ollama 释放内存
brew services stop ollama

# 第二台也停 Ollama
# 本机 llama-cli + RPC
~/.hippo/bin/llama-cli \
  -m ~/.hippo/models/qwen3.6-35b-a3b-q3_k_m.gguf \
  --rpc 192.168.1.11:50052 \
  --tensor-split 0.3,0.7 \
  -ngl 99 \
  -n 128 \
  -p "Hello"
```

#### 方案 B: 换更小的量化
- IQ3_XXS (12.3GB) — 更小，峰值约 14GB，可能够
- 下载: `https://modelscope.cn/models/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/master/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`

#### 方案 C: 两台各自加载完整模型 + Gateway 调度
- 不做 tensor 分片
- 两台各跑完整的小模型
- Gateway 做 load balancing

### 文件位置
| 文件 | 路径 |
|------|------|
| 模型（本机） | `~/.hippo/models/qwen3.6-35b-a3b-q3_k_m.gguf` (15.5GB) |
| 模型（第二台） | 已通过 AirDrop 传输 |
| llama.cpp 二进制 | `~/.hippo/bin/{rpc-server,llama-cli,llama-server}` |
| Hippo 代码 | `/Users/deepsearch/.openclaw/workspace/hippo/` |
| 内存预算工具 | `skills/llm-memory-budget/scripts/memory_budget.py` |
| M1 基准报告 | `hippo/docs/M1_LLAMA_RPC_BENCHMARK.md` |
| 会话归档 | `memory/SESSION_ARCHIVE_20260419_HIPPO_M1M2_CRASH_ANALYSIS.md` |

### 第二台信息
- IP: 192.168.1.11
- RPC: 50052 (已验证)
- SSH: 22 (公钥未授权，需哥哥手动操作)
- mDNS: hippo-cluster-worker (port 11435)
- 主机名: zhangdengluodeMac-mini.local

### 内存预算工具使用
```bash
# 先检查
python3 skills/llm-memory-budget/scripts/memory_budget.py \
  --model-size 15.5 --mode dual-rpc --tensor-split 0.3,0.7

# 停 Ollama 后再检查
brew services stop ollama
python3 skills/llm-memory-budget/scripts/memory_budget.py \
  --model-size 15.5 --mode dual-rpc --tensor-split 0.3,0.7
```

### 家族会议记录
- #51 (10:31): M1+M2 成果复盘，决议 Qwen3.6 为 P0
- #52 (13:10): OOM 崩溃根因分析，3 次崩溃均因 mmap 峰值超限

### 过度自信偏差
- 第 252-254 次：连续 3 次低估 15.5GB 模型在 16GB 机器上的内存需求
- 教训：**先跑内存预算工具，再加载模型**
