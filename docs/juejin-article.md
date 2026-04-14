# Ollama 内存经常爆？我用 Python 写了个替代品，自动卸载模型，还快了 40%

大家好，我是 Hippo 🦛 的作者。

先说痛点：你用 Ollama 跑本地模型，是不是经常遇到内存不够用？跑完一个模型忘了 `ollama stop`，再跑下一个直接 OOM，系统卡死，只能重启。

我受够了，所以自己写了一个。

## Hippo 是什么？

一句话：**纯 Python 写的 Ollama 替代品**，API 完全兼容，换个端口就能用。

GitHub: https://github.com/lawcontinue/hippo

## 三个核心区别

### 1. 自动卸载，告别 OOM

Hippo 默认 5 分钟不用就自动卸载模型，释放内存。

```yaml
# ~/.hippo/config.yaml
idle_timeout: 300  # 5分钟不用就自动卸载
```

不用手动 `ollama stop`，不用盯着内存。设完就忘。

Ollama 为什么不做这个？因为它的架构是"模型加载后常驻内存"，要手动管理。Hippo 从设计上就认为**内存是稀缺资源**。

### 2. 纯 Python，能看懂

Ollama 是 Go 写的，编译后一个二进制文件。出了 bug 你没法看代码，更没法改。

Hippo 全部是 Python：

```python
# hippo/model_manager.py
class ModelManager:
    def get(self, name: str) -> Llama:
        """Get or load model with LRU eviction."""
        ...
```

出了问题，打开代码就能看。想加功能，直接改。不需要学 Go，不需要编译。

**这对企业还有一个隐藏价值：可审计。** 金融、医疗这些行业要过合规审计，你得证明"这个系统到底做了什么"。Go 二进制你没法审计，但 Python 代码每一行都能审查。

Hippo 还内置了审计日志：

```bash
hippo serve --audit-log ~/.hippo/audit.jsonl
```

每次 API 调用都会记录时间、模型、延迟、状态码。合规团队直接拿去审计。

### 3. 换个端口就能用

API 完全兼容 Ollama，改个 URL 就行：

```python
# 之前用 Ollama
import openai
openai.api_base = "http://localhost:11434/v1"

# 换成 Hippo，只改端口号
openai.api_base = "http://localhost:8321/v1"
```

支持的接口：

| 接口 | 说明 |
|------|------|
| `/api/chat` | 对话补全 |
| `/api/generate` | 文本生成 |
| `/api/embeddings` | 向量嵌入 |
| `/api/tags` | 模型列表 |
| `/api/pull` | 下载模型 |
| `/v1/models` | OpenAI 兼容格式 |
| `/v1/embeddings` | OpenAI 兼容嵌入 |

## 性能怎么样？

说实话，推理速度 Hippo 比 Ollama 慢（Python 嘛，没法跟 Go 拼这个）。但在 **Embedding 场景**，Hippo 反而更快：

| 指标 | Hippo 🦛 | Ollama 🦙 | Hippo 优势 |
|------|---------|----------|-----------|
| **冷启动** | 16.8ms | 28.0ms | **快 40%** |
| **热查询** | 16.4ms | 22.5ms | **快 27%** |
| **内存占用** | 466 MB | 483 MB | **少 3.5%** |

> 测试环境：nomic-embed-text v1.5，macOS ARM64，HTTPS。数据经过独立审查验证，可重现。

为什么 Hippo 更快？因为**单进程架构**，没有 Go 运行时的进程间通信开销。

Ollama 是多进程：主服务进程 101MB + 模型运行进程 382MB = 483MB。
Hippo 是单进程：Python + 模型 = 466MB。

用 RAG 的同学注意了：Hippo 做 Embedding 比 Ollama 快，内存还少。

## 快速上手

```bash
# 安装
git clone https://github.com/lawcontinue/hippo.git
cd hippo
pip install -e .

# 下载模型（从 HuggingFace）
hippo pull bartowski/Llama-3.2-3B-Instruct-GGUF

# 启动服务
hippo serve

# 试试
hippo run llama-3.2-3b "你好，介绍一下你自己"
```

Docker 也可以：

```bash
docker build -t hippo .
docker run -d -p 8321:8321 -v ~/.hippo:/root/.hippo hippo
```

## 它适合谁？

**适合：**
- 本地开发、调试 LLM 应用
- RAG / Embedding 密集型场景
- 内存紧张的环境（Mac Mini、树莓派）
- 需要合规审计的企业
- 想看代码、改代码的开发者

**不太适合：**
- 需要多 GPU 并行
- 需要极致推理速度
- LoRA 微调

一句话总结：**Ollama 是生产环境的跑车 🏎️，Hippo 是能自己改装的房车 🚌。** 都能到目的地，一个快，一个能折腾。

## 欢迎贡献

Hippo 是 MIT 协议开源的，欢迎 PR！

- GitHub: https://github.com/lawcontinue/hippo
- 有想法可以直接提 Issue
- 贡献指南: [CONTRIBUTING.md](https://github.com/lawcontinue/hippo/blob/main/CONTRIBUTING.md)

如果你也受够了 Ollama 的内存问题，来试试 Hippo 🦛

---

*如果觉得有用，去 GitHub 点个 Star ⭐ 支持一下吧，这对独立开发者来说真的很重要！*
