# M1: llama.cpp RPC 最小验证报告

**日期**: 2026-04-19
**实施者**: 忒弥斯 🔮 + Code 💻
**状态**: ✅ 验证通过

## 环境信息

| 项目 | 值 |
|------|-----|
| 硬件 | Mac Mini M4 (16GB) |
| OS | macOS Darwin 25.1.0 (arm64) |
| llama.cpp | b1-9e5647a (ggml 0.9.11) |
| 模型 | DeepSeek-R1-Distill-Llama-8B Q4 (4.6 GB) |
| GPU | Apple M4, 12,124 MiB, unified memory |
| RPC 协议 | TCP, v4.0.0 |

## 编译

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp-m1
cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON
cmake --build build -j8
```

产出二进制：`rpc-server`, `llama-cli`, `llama-server`（安装到 `~/.hippo/bin/`）

## 基准测试结果

### llama-cli 单机（-ngl 99）

| 指标 | 值 |
|------|-----|
| Prompt 处理 | 79.9 tok/s |
| 生成速度 | 20.9 tok/s |
| Metal 加载 | 0.008s（缓存后） |
| GPU 显存占用 | 模型 4,685 MiB + 上下文 6,368 MiB + 计算 258 MiB |

### llama-cpp-python 本地（无 RPC）

| 测试 | tok/s | tokens | 耗时 |
|------|-------|--------|------|
| 长文本生成 | 20.2 | 128 | 6.34s |
| 代码生成 | 20.2 | 128 | 6.33s |
| 短回答 | 19.1 | 64 | 3.35s |
| **平均** | **19.8** | — | — |

### llama-cpp-python RPC（本地 50% + RPC 50%，localhost）

| 测试 | tok/s | tokens | 耗时 |
|------|-------|--------|------|
| 长文本生成 | 20.1 | 128 | 6.38s |
| 代码生成 | 19.9 | 128 | 6.42s |
| 短回答 | 19.2 | 64 | 3.33s |
| **平均** | **19.7** | — | — |

### 对比

| 模式 | 平均 tok/s | 开销 |
|------|-----------|------|
| 本地（无 RPC） | 19.8 | 基准 |
| RPC（localhost 50/50） | 19.7 | **-0.5%** ✅ |

**关键发现**: localhost RPC 开销极小（<1%），证明 RPC 协议本身高效。

## RPC Server 特性

- 自动检测 Metal GPU（Apple M4, 12,124 MiB）
- 支持 TCP 传输
- 支持多设备：MTL0 (Metal) + BLAS (Accelerate)
- zero-copy tensor 传输（unified memory）

## M1 结论

✅ **llama.cpp RPC 验证通过**

1. 编译成功（Metal + RPC）
2. rpc-server 稳定运行
3. llama-cpp-python 支持 `rpc_servers` + `tensor_split` 参数
4. localhost RPC 开销 < 1%
5. 双机场景需要第二台 Mac Mini 实测（跨网络延迟是关键变量）

## 下一步：M2（Hippo 集成）

将 RPC 能力集成到 Hippo 的 InferenceBackend 抽象层：

```python
class LLamaRPCBackend(InferenceBackend):
    def __init__(self, rpc_workers: list[str], tensor_split: list[float]):
        self.rpc_workers = rpc_workers
        self.tensor_split = tensor_split

    def load(self, model_path: str):
        from llama_cpp import Llama
        self.model = Llama(
            model_path=model_path,
            rpc_servers=self.rpc_workers,
            tensor_split=self.tensor_split,
        )
```

## 已安装文件

- `~/.hippo/bin/rpc-server` (132 KB)
- `~/.hippo/bin/llama-cli` (1.4 MB)
- `~/.hippo/bin/llama-server` (9.0 MB)

---

**M1 耗时**: ~25 分钟
**M1 效率**: 验证充分，数据可靠
