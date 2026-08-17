# DSH Cross-Session Memory 🦛🧠

给运行在 DeepSeek Harness (DSH) 上的 agent 加上**跨会话长期记忆**：agent 主动决定记什么、怎么检索、何时遗忘，全部本地运行，零云端依赖。检索引擎由本仓库的 `hippo.embedding.VectorStore` 提供。

```
┌─────────────┐  memory_* tools   ┌──────────────────────┐  JSON-lines (rid)  ┌────────────────────┐
│  DSH Agent   │ ────────────────▶ │  hippo-memory 插件    │ ─────────────────▶ │  hippo_bridge.py    │
│  (preset)    │                   │  (静态 Cordis 插件)   │   常驻进程          │  (serve 模式)        │
└─────────────┘                   └──────────────────────┘                    └─────────┬──────────┘
                                                                                        │
                                                          BM25 + bge-small-zh 语义向量 RRF 融合        │
                                                          置信度加权 · 来源溯源 · 低置信衰减           ▼
                                                                       ~/.dsh/hippo-memory.db (SQLite)
```

**模型常驻内存**：插件启动时拉起 Python 桥接进程并加载嵌入模型（首次约 60–90s），此后每次记忆操作仅需毫秒级响应。

## 五个模型工具

| 工具 | 用途 |
|---|---|
| `memory_store` | 写入值得跨会话记住的事实（来源分级 user/model/inference，置信度加权） |
| `memory_recall` | 混合检索：BM25 关键词 + 语义向量 RRF 融合 + 置信度加权排序 |
| `memory_forget` | 按 id 删除记忆 |
| `memory_list` | 浏览最近记忆 |
| `memory_rebuild` | 为 sparse 时代的旧记忆补建语义向量 |

**记忆治理**（继承 `hippo.embedding.memory_safety`）：
- 来源溯源：`user`（用户亲口所述，默认 conf 0.9）> `model`（0.5）> `inference`（0.4）
- 置信度加权检索：低置信记忆即使关键词命中也自动沉底
- 时间衰减：`decay` 操作可批量降低超过 N 天且置信度低于阈值的模型记忆

## 安装

### 1. Python 环境（在工作目录，即本目录）

```powershell
# 本目录 = <hippo 仓库>/dsh-memory/
cd dsh-memory

# 创建 venv（复用系统已装的 torch/numpy）
python -m venv --system-site-packages .venv

# 安装依赖（中国大陆建议加清华镜像 -i https://pypi.tuna.tsinghua.edu.cn/simple）
.venv\Scripts\python.exe -m pip install sentence-transformers

# 下载嵌入模型 bge-small-zh-v1.5（约 96MB，HuggingFace 不通时用 ModelScope）
# 放到 dsh-memory/models/bge-small-zh-v1.5/
```

模型文件清单（ModelScope `BAAI/bge-small-zh-v1.5`）：
`config.json`、`model.safetensors`、`tokenizer.json`、`tokenizer_config.json`、`vocab.txt`、`special_tokens_map.json`、`modules.json`、`sentence_bert_config.json`、`config_sentence_transformers.json`、`1_Pooling/config.json`

> 没有模型也能跑：自动回退 sparse 模式（纯 BM25），零额外依赖。

### 2. 接入 DSH 预设

把 `hippo-memory.plugin.js` 放进你的 preset 目录（或直接引用本仓库路径），并在 `agent.cordis.yml` 末尾加一行：

```yaml
- id: hippo-memory
  name: './plugins/hippo-memory.js'   # 插件实际路径（相对 preset 目录或绝对路径）
```

插件通过环境变量定位桥接（都有默认值，通常无需设置）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `HIPPO_DSH_DIR` | 插件所在目录 | 存放 `hippo_bridge.py` 与 `.venv` 的工作目录 |
| `HIPPO_DSH_PYTHON` | `<DIR>/.venv/Scripts/python.exe` | 桥接用的 Python 解释器 |
| `HIPPO_MEMORY_DB` | `~/.dsh/hippo-memory.db` | 全局记忆库（所有会话共享） |
| `HIPPO_MEMORY_MODE` | `auto` | `hybrid` / `sparse` / 自动 |
| `HIPPO_EMBED_MODEL_PATH` | `<DIR>/models/bge-small-zh-v1.5` | 本地嵌入模型目录 |

### 3. 验证

新会话里对 agent 说：

> "记住：我们的生产环境域名是 example.com" → 调用 `memory_store`
> "我之前告诉过你什么域名？" → 调用 `memory_recall`，跨会话命中

## 协议（供其他运行时接入）

桥接进程以换行分隔的 JSON 行通信，关联键为 `rid`（**不是** `id`——它属于 delete/update 等业务操作）：

```
→ {"rid":1,"op":"store","text":"...","source":"user","confidence":0.9}
← {"rid":1,"ok":true,"id":7,"count":3}
→ {"rid":2,"op":"recall","query":"域名","top_k":5}
← {"rid":2,"ok":true,"results":[{"id":7,"text":"...","score":2.31,"source":"user","confidence":0.9,...}]}
```

操作一览：`store` `recall` `list` `delete` `update` `count` `decay` `rebuild`。一次性调用（不带 `--serve`）也支持：stdin 收一条 JSON，stdout 回一条。

## 设计说明

- **为什么不每次调用起一个进程**：torch + transformers 导入约 40–60s、模型加载数秒，逐次冷启动不可接受；常驻进程让模型只加载一次
- **为什么 `rid` 而不是 `id` 做关联键**：协议序号曾覆盖 `delete`/`update` 的业务 id 字段导致"删除成功但没删"的静默错误——字段命名隔离是硬教训
- **隐私**：记忆、向量、模型全部本地；不发起任何网络请求（`TRANSFORMERS_OFFLINE=1`）

## License

MIT（随主仓库）
