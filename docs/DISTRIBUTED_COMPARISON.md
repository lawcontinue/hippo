# 四大分布式推理方案对比分析

**分析日期**: 2026-04-19
**分析者**: 忒弥斯 🔮 + 家族全员

---

## 1. exo（43,768⭐）— P2P 家用集群

**License**: Apache 2.0 ✅
**语言**: Python + Rust
**平台**: Mac/iPhone/iPad/Linux

### 架构
```
Master（协调器）
├── placement.py — 按内存比例自动分片
├── topology.py — 网络拓扑
└── event_router.py — 事件路由

Worker（执行器）
├── engines/mlx/ — MLX 推理引擎
│   ├── auto_parallel.py
│   └── cache.py — KV cache
├── runner/ — 推理运行器
└── plan.py — 执行计划

Rust 网络（底层）
├── discovery.rs — mDNS/Swarm 发现
└── swarm.rs — P2P 通信
```

### 分片策略
- `allocate_layers_proportionally()` — 按内存比例分配层
- 支持 Pipeline / Tensor / CFG（MoE）三种分片
- 自动拓扑发现 + 最优 cycle 选择

### 优势
- 🏆 最成熟，43K⭐，社区活跃
- 跨平台（Mac/Linux/iPhone）
- 自动分片，零配置
- OpenAI/Ollama/Claude API 兼容

### 劣势
- Rust 网络层太重（编译复杂）
- Python 3.13+ 要求
- 对 Hippo 来说功能过度（image pipeline 等）
- 不支持 Windows

### Hippo 可借鉴
- ✅ `allocate_layers_proportionally` 算法（~50行）
- ✅ Pipeline Parallelism 设计
- ✅ MLX ring 接口封装
- ❌ Rust 网络层（太重）
- ❌ 完整集成（功能过度）

---

## 2. Petals（10,079⭐）— BitTorrent-style 分布式

**License**: MIT ✅
**语言**: Python
**平台**: Linux GPU only

### 架构
```
Client
├── from_pretrained.py — 获取远程模型
├── inference_session.py — 推理会话
├── remote_sequential.py — 远程顺序执行
└── routing/sequence_manager.py — 路由管理

Server
├── block_functions.py — 层级计算
├── memory_cache.py — 内存缓存
├── task_pool.py — 任务池
└── throughput.py — 吞吐量优化

DHT（分布式哈希表）
└── 去中心化设备发现和路由
```

### 分片策略
- 每台服务器负责若干 transformer block
- Client 按顺序调度到不同 server
- DHT 路由（类似 BitTorrent）

### 优势
- 去中心化（DHT 路由）
- 支持微调（fine-tuning）
- 大规模（互联网级）

### 劣势
- ❌ 仅支持 GPU（CUDA 必需）
- ❌ 不支持 Apple Silicon
- ❌ 不支持 Windows
- ❌ 主要支持 BLOOM/Falcon/Mixtral
- ❌ 延迟高（互联网级）

### Hippo 可借鉴
- ✅ DHT 路由思想（未来扩展用）
- ✅ task_pool 任务调度模式
- ❌ 不适合家用场景（GPU only + 互联网延迟）

---

## 3. mlx_sharding（100⭐）— MLX Pipeline Parallelism

**License**: 未明确（需确认）
**语言**: Python
**平台**: Mac only

### 架构（极简，~10个文件）
```
shard/
├── server/
│   ├── server.py — gRPC 服务端（处理 tensor）
│   └── model/
│       ├── base.py — IdentityBlock（跳过层）
│       ├── llama.py — LLaMA 模型切分
│       ├── deepseek_v2.py — DeepSeek 切分
│       └── gemma2.py — Gemma2 切分
├── grpc/ — protobuf 定义
├── main.py — CLI 入口
├── openai_api.py — OpenAI API 兼容
└── utils.py — tensor 序列化
```

### 分片策略
- **手动指定 start_layer / end_layer**
- Worker 加载部分层，其余用 IdentityBlock 跳过
- gRPC 传递 hidden state（tensor bytes）
- 每个 Worker 接收 tensor → 过自己的层 → 返回 tensor

### 核心代码（server.py 精华）
```python
class MLXTensorServicer:
    def SendTensor(self, request, context):
        tensor = bytes_to_tensor(request.tensor_data)
        tensor = mx.reshape(tensor, request.shape)
        processed = MODEL(tensor, cache=CACHE)  # 过模型层
        return tensor_to_bytes(processed)
```

### 优势
- ✅ 极简（~1000行核心代码）
- ✅ Pipeline parallelism 原理清晰
- ✅ gRPC 通信（比 HTTP 快）
- ✅ OpenAI API 兼容
- ✅ 教育价值高（容易理解）

### 劣势
- ❌ Mac only（MLX 绑定）
- ❌ 手动分片（不自动）
- ❌ 100⭐，不够成熟
- ❌ 无故障转移

### Hippo 可借鉴
- ✅ **Pipeline 架构设计**（最直接可参考）
- ✅ tensor 序列化方案
- ✅ IdentityBlock 跳过层技术
- ✅ gRPC 通信模式（未来可选）
- ❌ 不适合作为后端（Mac only）

---

## 4. llama.cpp RPC（内置）— Tensor Offload

**License**: MIT ✅
**语言**: C/C++
**平台**: 全平台（Mac/Windows/Linux）

### 架构
```
主节点（llama-cli/llama-server）
├── 加载模型
├── GGML RPC backend
└── --rpc host:port 指定远程 Worker

远程 Worker（rpc-server）
├── 暴露本地 GPU/CPU 设备
├── 接收 tensor 计算请求
└── 返回计算结果
```

### 分片策略
- **自动按显存比例分配 tensor**
- 主节点根据每个设备的可用内存自动分配
- 可用 `--tensor-split` 手动覆盖比例
- 支持 RDMA（Linux，更低延迟）

### 优势
- ✅ **真正的跨平台**（Mac Metal + CUDA + CPU）
- ✅ 全自动分片（无需手动指定层）
- ✅ 本地缓存（避免重复传输）
- ✅ 命令行直接用：`llama-cli --rpc host:port`
- ✅ 支持 RDMA（低延迟）

### 劣势
- ❌ Python 绑定未暴露 RPC 接口（需用 C 或命令行）
- ❌ PoC 阶段（README 明确说 fragile and insecure）
- ❌ 需要编译 rpc-server 二进制
- ❌ 无认证/加密

### Hippo 可借鉴
- ✅ **最佳跨平台方案**（Windows 兼容）
- ✅ 自动 tensor split 算法
- ✅ 本地缓存优化思路
- ❌ Python 不可用，需 subprocess 或 C 扩展

---

## 综合对比

| 维度 | exo | Petals | mlx_sharding | llama.cpp RPC |
|------|-----|--------|-------------|---------------|
| ⭐ | 43,768 | 10,079 | 100 | — |
| License | Apache 2.0 | MIT | 未明确 | MIT |
| 跨平台 | Mac/Linux | GPU Linux | Mac | **全平台** ✅ |
| Apple Silicon | ✅ | ❌ | ✅ | ✅ |
| Windows | ❌ | ❌ | ❌ | ✅ |
| 自动分片 | ✅ | ✅ | ❌ 手动 | ✅ |
| Pipeline | ✅ | ✅ | ✅ | ✅ |
| Tensor parallel | ✅ | ❌ | ❌ | ✅ |
| MoE 支持 | ✅ | ❌ | ❌ | ❌ |
| 成熟度 | 🏆 生产级 | 生产级 | 教育级 | PoC |
| 集成难度 | 中（进程级） | 难（GPU only） | 低（代码少） | 中（需编译） |

---

## Hippo Phase 1 策略建议

**三层架构**：

```
Hippo Gateway（路由/治理）
├── InferenceBackend（抽象层）
│   ├── LocalBackend（单机 llama.cpp）— Phase 0 ✅
│   ├── MLXPipelineBackend（借鉴 mlx_sharding）— Phase 1
│   ├── LLamaRPCBackend（llama.cpp RPC）— Phase 1.5
│   └── ExoBackend（exo 进程级集成）— Phase 2 可选
```

**优先级**：
1. **MLXPipelineBackend** — 借鉴 mlx_sharding 的 Pipeline 架构（~500行），Mac 双机跑通
2. **LLamaRPCBackend** — subprocess 调用 llama.cpp rpc-server，实现跨平台
3. **ExoBackend** — 可选，未来扩展用

---

**文档版本**: v1.0
**作者**: 忒弥斯 🔮 + 家族全员
**创建时间**: 2026-04-19 02:41
