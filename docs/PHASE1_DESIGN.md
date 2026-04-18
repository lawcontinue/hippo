# Hippo Phase 1 — 模型分片 + 分布式推理

**日期**: 2026-04-19
**状态**: 🚧 开发中
**前置**: Phase 0 ✅（双机注册 + 心跳 + 基础路由）

---

## 目标

**两台 Mac Mini M4 (16GB) 通过 Thunderbolt 桥接，联合跑 Qwen3-35B MoE (Q4, ~18GB)**

验收标准：
- tok/s ≥ 单机 Q3_K_M 的 80%
- 延迟 ≤ 单机的 1.5x
- 自动分片，零配置

---

## 技术路线

### 方案 A：MLX 分布式（推荐 ⭐）

**优势**：Apple Silicon 原生，零拷贝，延迟最低
**依赖**：`mlx.distributed`（send/recv/all_gather）

```python
import mlx.core as mx

# 初始化分布式
mx.distributed.init()  # 自动发现或指定 backend
group = mx.distributed.Group()

# 每台机器负责部分层
# Machine 0: layers 0-31 (14GB)
# Machine 1: layers 32-47 (4GB)
```

**通信**：Thunderbolt 桥接 + MPI/NCCL-like
**预估延迟**：< 1ms（Thunderbolt 40Gbps）
**预估 tok/s**：12-20

### 方案 B：llama.cpp RPC（备选）

**优势**：不依赖 Apple Silicon，跨平台
**依赖**：llama-cpp-python 编译时 `CMAKE_ARGS="-DGGML_RPC=ON"`

```bash
# Worker 端启动 RPC server
./llama-rpc-server -H 0.0.0.0 -p 50052

# 主节点连接
./llama-cli --rpc 192.168.1.11:50052 -m model.gguf
```

**劣势**：需要重新编译，Apple Silicon 上不如 MLX 高效
**预估 tok/s**：8-15

### 方案 C：自定义 Tensor 分片（Hippo 原生）

**优势**：完全控制，可优化通信模式
**劣势**：开发量大，需要实现 GGUF 层级解析 + tensor 序列化

---

## 推荐路线：方案 A（MLX 分布式）

**理由**：
1. 两台都是 Apple Silicon M4 → MLX 是最优推理引擎
2. `mlx.distributed` 已内置 send/recv/all_gather
3. Thunderbolt 延迟 < 1ms，适合 pipeline parallelism
4. 不需要 GGUF 格式，直接用 MLX 的 safetensors

### 架构

```
Mac Mini #1 (主节点, rank=0)
├─ Hippo Gateway + Pipeline Scheduler
├─ Qwen3-35B layers 0-31 (~14GB)
├─ MLX distributed: send hidden state to rank=1
└─ 收到 rank=1 的输出 → 返回给用户

Mac Mini #2 (工作节点, rank=1)  
├─ Hippo Worker
├─ Qwen3-35B layers 32-47 (~4GB)
├─ MLX distributed: recv hidden state from rank=0
└─ send output back to rank=0
```

### Pipeline Parallelism

```
时间线：
t0: [token1 layer0-31 @ M1] → send → [layer32-47 @ M2] → 返回
t1: [token2 layer0-31 @ M1] → send → [layer32-47 @ M2] → 返回
    ↑ M1 在等 M2 时可以预处理 token3
```

### 实施步骤

| 步骤 | 内容 | 预估 |
|------|------|------|
| P1-1 | MLX distributed 研究 + PoC | 1h |
| P1-2 | Thunderbolt 延迟基准测试 | 20min |
| P1-3 | GGUF → MLX 模型转换器 | 30min |
| P1-4 | Pipeline Scheduler 实现 | 1h |
| P1-5 | Qwen3-35B 下载 + 分片 | 30min |
| P1-6 | 双机实测 + benchmark | 30min |
| **总计** | | **~4h** |

---

## 依赖

- [x] Phase 0 完成（注册、心跳、路由）
- [x] MLX 0.30.5 已安装（系统级）
- [ ] MLX 安装到 Hippo venv
- [ ] Thunderbolt 桥接配置
- [ ] Qwen3-35B MoE 模型文件

---

**文档版本**: v1.0
**作者**: 忒弥斯 🔮 + Code 💻 + 雅典娜 📊
**创建时间**: 2026-04-19 02:16
