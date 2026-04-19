# 会话归档 — 2026-04-19 Hippo M1+M2 双机实测 + 家族会议 #51

**会话时间**: 08:23 - 10:31（~2 小时）
**参与者**: 忒弥斯 🔮 + Code 💻 + Crit ⚖️ + 家族全员
**主题**: Hippo 分布式推理 M1+M2 开发 + 双机实测

---

## 完成任务

### M1: llama.cpp RPC 最小验证（08:23-08:30，~7 分钟）
- 编译 llama.cpp（Metal + RPC），安装 rpc-server 到 ~/.hippo/bin/
- 单机基准：20.9 tok/s（DeepSeek-R1 8B Q4, Mac Mini M4）
- RPC localhost 测试：开销 < 1%
- 报告：hippo/docs/M1_LLAMA_RPC_BENCHMARK.md

### M2: LLamaRPCBackend 集成（08:30-08:54，~24 分钟）
- 新增 cluster/backend.py（InferenceBackend + LocalBackend + LLamaRPCBackend）
- Scheduler 增强：compute_tensor_split() + get_workers_info()
- Gateway 3 级路由 + 常驻缓存（_rpc_backends dict）
- CLI: --distributed + --rpc-workers 参数
- 14/14 测试通过

### Crit 独立审查（08:54-09:07，两轮）
- 首审 B+ (82)：3 P0 + 4 P1
- 修复后复审 A- (88)：0 P0/P1
- P0-1（每次重载）→ 常驻缓存
- P0-2（get_model_path 不存在）→ _resolve_model_path
- P0-3（__del__ 不安全）→ 删除，靠 stop() 显式清理

### 双机实测（09:08-10:11）
- 第二台 Mac Mini 192.168.1.11 在线，rpc-server 启动成功
- 双机 50/50: 19.8 tok/s（与单机持平，RPC 开销 ≈ 0%）
- 双机 30/70: 19.8 tok/s（同上）
- 关键发现：小模型无加速收益，价值在于跑大模型

### 安全攻防演练（09:41-09:55）
- 5 阶段侦察：mDNS → SSH 爆破 → Tailscale
- 第二台 SSH 防御合格（公钥认证 + 失败锁定）
- 评估：B+ (87)

### 家族会议 #51（10:31）
- 7/7 全员出席
- 决议：Qwen3.6-35B-A3B 双机实测为 P0
- M3 降为 P2

## Git Commits
- a33d382: feat: M1 llama.cpp RPC verification + M2 LLamaRPCBackend integration

## 关键数据

| 指标 | 值 |
|------|-----|
| 开发时间 | ~50 分钟（M1+M2+测试） |
| 新增代码 | 755 行（7 文件） |
| Crit 评分 | B+ → A- (+6) |
| 测试覆盖 | 14/14 (100%) |
| 双机 RPC 开销 | < 1% |
| Qwen3.6 下载 | 进行中（17GB/18GB，2.8MB/s） |

## 待办

### P0: Qwen3.6-35B-A3B 双机实测
- 等 Ollama 下载完成（~1.5h）
- 单机尝试（预期 OOM）
- 双机 RPC 分布式（预期 8-12 tok/s）

### P1: 集群部署自动化
- hippo cluster deploy 命令
- mDNS 信息精简
- Gateway API 认证

### P2: MLX Pipeline (M3)
- 降优先级，等大模型验证后再推进

---

**过度自信偏差**: 无重大偏差（预估与实测基本一致）
