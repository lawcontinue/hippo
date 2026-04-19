# 会话归档 — 2026-04-19 Hippo M1+M2 双机实测 + Qwen3.6 崩溃分析

**会话时间**: 08:23 - 13:13（~4 小时 50 分钟）
**参与者**: 忒弥斯 🔮 + Code 💻 + Crit ⚖️ + 家族全员（会议 #51, #52）
**主题**: Hippo 分布式推理开发 + 双机实测 + 崩溃根因分析

---

## 完成任务

### M1: llama.cpp RPC 验证 ✅（08:23-08:30）
- 编译 llama.cpp（Metal + RPC），安装到 ~/.hippo/bin/
- 单机基准：20.9 tok/s（DeepSeek-R1 8B Q4, Mac Mini M4）
- 报告：hippo/docs/M1_LLAMA_RPC_BENCHMARK.md

### M2: LLamaRPCBackend 集成 ✅（08:30-08:54）
- cluster/backend.py（InferenceBackend + LocalBackend + LLamaRPCBackend）
- Scheduler 增强 + Gateway 3 级路由 + 常驻缓存
- Crit 两轮审查 B+ → A-（0 P0/P1），14/14 测试通过
- Git: a33d382

### DeepSeek-R1 8B 双机实测 ✅（09:08-10:11）
- 双机 50/50: 19.8 tok/s（RPC 开销 <1%）
- 结论：小模型无加速收益，价值在于跑大模型

### 安全攻防演练 ✅（09:41-09:55）
- 5 阶段侦察全部失败，SSH 防御 B+ (87)

### Qwen3.6-35B-A3B 下载 ✅（10:08-11:44）
- Ollama 太慢（2.8MB/s），换 ModelScope（~10MB/s）
- 下载 Q3_K_M（15.5GB）+ AirDrop 传到第二台

### Qwen3.6 双机实测 ❌（11:44, 12:26, 13:01）
- 3 次 OOM 崩溃，macOS 杀掉 OpenClaw 进程
- 根因：16GB 内存无法承受 15.5GB 模型的加载峰值

---

## 崩溃根因分析（家族会议 #52）

### 根因
llama-cpp-python 加载时先 mmap 整个 GGUF 文件（15.5GB 虚拟内存），再按 tensor_split 分配。即使 30/70 分片，加载峰值也超过 16GB 物理内存。

### 内存预算

| 分片 | 稳态需要 | 加载峰值 | 16GB可用 | 结果 |
|------|---------|---------|---------|------|
| 单机 100% | 17.5GB | ~18GB | 6.8GB | ❌ |
| 双机 50/50 | 9.75GB | ~14GB | 6.8GB | ❌ |
| 双机 30/70 | 6.65GB | ~12GB | 6.8GB | ❌ |

### 修复方案（7/7 批准）
1. **P0**: 内存预检 — 加载前检查可用内存，不足则拒绝
2. **P0**: 用 llama-cli 替代 llama-cpp-python（内存效率更高）
3. **P0**: 测试前先停 Ollama（释放 0.8GB）
4. **P1**: LLamaRPCBackend.load() 集成内存预检

### 经验教训
1. **"理论够用 ≠ 实际够用"** — 分片只减少稳态内存，不减少加载峰值
2. **"mmap 不是免费的"** — 即使只读部分文件，虚拟内存映射仍占空间
3. **"先测内存再加载"** — 大模型加载前必须做内存预算
4. **"OOM Killer 无差别"** — macOS 可能杀 OpenClaw 而不是模型加载进程

### 过度自信偏差
- 第 252-254 次：连续 3 次低估 15.5GB 模型的内存峰值需求

---

## Skill 沉淀候选

### 候选 1: 大模型内存预算计算器 ⭐⭐⭐⭐⭐
- **出现次数**: 3 次（每次加载前都应该算）
- **内容**: GGUF 大小 → 量化级别 → 单机/分片 → 内存需求预估
- **预估节省**: 5 分钟/次，避免 OOM 崩溃

### 候选 2: 分布式推理部署清单
- **出现次数**: 2 次（M1 编译 + 第二台部署）
- **内容**: 编译 → 安装 → rpc-server 启动 → 模型分发 → 测试
- **预估节省**: 15 分钟/次

---

## 待办

### P0: Qwen3.6 双机实测（下个会话）
- [ ] 停 Ollama 释放内存
- [ ] 用 llama-cli 直接测试 RPC（非 Python）
- [ ] 第二台也停 Ollama

### P1: 内存预检集成
- [ ] LLamaRPCBackend.load() 加内存检查
- [ ] CLI 显示内存预算警告

### P2: M3 MLX Pipeline
- [ ] 等大模型验证通过后再推进

---

## Git Commits
- a33d382 (hippo): feat: M1 llama.cpp RPC + M2 LLamaRPCBackend
- iCloud 同步: ✅

---

**家族会议**: #51 (10:31) + #52 (13:10)
**过度自信偏差**: 第 252-254 次
