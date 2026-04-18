# exo 源码架构分析 — Hippo 集成参考

**分析日期**: 2026-04-19
**版本**: exo v0.3.69
**License**: Apache 2.0 ✅
**语言**: Python + Rust（网络层）

---

## 核心架构

```
exo master (协调器)
├── placement.py — 分片调度策略
├── placement_utils.py — 按内存比例分配层
└── topology.py — 网络拓扑管理

exo worker (工作节点)
├── runner/llm_inference/ — 推理执行
│   ├── runner.py — Runner 生命周期
│   └── batch_generator.py — 批量生成
├── engines/mlx/ — MLX 引擎
│   ├── auto_parallel.py — 自动并行
│   ├── cache.py — KV cache
│   └── generator/ — 生成器
└── plan.py — 执行计划

exo routing (通信)
├── router.py — 消息路由
├── event_router.py — 事件分发
└── topics.py — 发布/订阅

exo api (对外接口)
├── adapters/chat_completions.py — OpenAI 兼容
├── adapters/ollama.py — Ollama 兼容
└── adapters/claude.py — Claude API 兼容

rust/networking (底层通信)
├── discovery.rs — mDNS/swarm 发现
└── swarm.rs — P2P 通信
```

## 分片策略（placement_utils.py）

**核心函数**: `allocate_layers_proportionally()`

```python
# 按可用内存比例分配模型层
def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],  # 每台机器的内存占比
) -> list[int]:  # 每台机器分配的层数
```

**示例**（双 M4 16GB 跑 Qwen3-35B 48层）：
- M4 #1: 13GB / 26GB = 50% → 24 层
- M4 #2: 13GB / 26GB = 50% → 24 层

**分片类型**:
- `PipelineShardMetadata` — Pipeline 并行（逐层传递 hidden state）
- `TensorShardMetadata` — Tensor 并行（层内切分矩阵）
- `CfgShardMetadata` — 条件推理（MoE 专家分发）

## 关键设计决策

### 1. Pipeline Parallelism（默认）
- 每个 Worker 负责连续的层
- Hidden state 通过网络传递到下一个 Worker
- 简单、延迟可预测、适合家庭网络

### 2. Ring 通信（MLX）
- `MlxRingInstance` — MLX ring backend 分布式
- 利用 `mx.distributed` 的 send/recv
- 需要 `MLX_RANK` + `MLX_HOSTFILE`

### 3. JAGGR 通信（自定义）
- `MlxJacclInstance` — exo 自定义的集合通信
- 比 ring 更灵活，支持 RDMA
- Rust 实现，性能更好

### 4. 设备发现
- mDNS（和 Hippo 一样）
- + Bootstrap peers（手动指定）
- + Swarm 协议（P2P 扩散）

## Hippo 可借鉴的部分

| 组件 | exo 实现 | Hippo 应该 |
|------|---------|-----------|
| 分片策略 | `allocate_layers_proportionally` | 直接借鉴，简单有效 |
| Pipeline 调度 | Runner + Event 系统 | 简化版，用 asyncio |
| 设备发现 | mDNS + Rust swarm | 已有（Phase 0） |
| MLX 分布式 | ring/jaccl | 借鉴 ring 接口 |
| API 兼容 | OpenAI/Ollama/Claude | 已有（Phase 0） |

## Hippo 不应该依赖的部分

| 组件 | 原因 |
|------|------|
| exo 的 Rust 网络层 | 太重，Hippo 用 Python aiohttp 足够 |
| exo 的完整 API 层 | Hippo 已有自己的 |
| exo 的 election 机制 | 家庭场景不需要选主 |
| exo 的 image pipeline | Hippo 专注 LLM |

## 集成策略

**不 import exo**，而是借鉴核心算法 + 可选进程级集成：

```python
# 借鉴算法（Hippo 自己实现）
class PipelineScheduler:
    def allocate_layers(self, total_layers, workers):
        # 源自 exo 的 allocate_layers_proportionally
        memory_fractions = [w.memory / total for w in workers]
        return allocate_layers_proportionally(total_layers, memory_fractions)

# 可选集成（进程级，不 import）
class ExoBackend(InferenceBackend):
    async def start(self):
        self._process = subprocess.Popen(["exo", ...])
    async def generate(self, prompt):
        # 通过 HTTP API 调用 exo
        async with aiohttp.ClientSession() as session:
            resp = await session.post("http://localhost:8000/v1/chat/completions", ...)
```

---

**结论**: exo 的核心分片算法（~200行）值得直接借鉴。
完整集成用进程级（subprocess + HTTP），不深度依赖。
