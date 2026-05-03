# Hippo Continuous Batch API — 使用指南

**作者**: 忒弥斯 (T-Mind) 🔮
**版本**: v1.0
**创建日期**: 2026-04-22

---

## 🎯 功能概述

Hippo 现在支持 **Continuous Batching**（连续批处理），允许最多 7 个 Agent 同时推理，显著提升吞吐量。

### 核心特性

1. ✅ **7 Agent 并发**：最多 7 个请求同时处理
2. ✅ **Prompt Padding**：自动处理不同长度 prompt
3. ✅ **FIFO 队列**：先到先服务
4. ✅ **内存监控**：>15GB 自动拒绝新请求
5. ✅ **Ollama 兼容**：与现有 API 完全兼容

---

## 📊 性能数据（Gemma3-12B）

| Batch Size | 总吞吐 | 每序列 | 加速比 |
|-----------|--------|--------|--------|
| B=1 | 13.8 tok/s | 13.8 | 1.00x |
| B=2 | 27.2 tok/s | 13.6 | 1.97x |
| B=4 | 33.1 tok/s | 8.3 | 2.40x |
| B=7 | 33.7 tok/s | 4.8 | 2.44x |

**结论**: B=2 是甜点（1.97x 加速，Step 仅 +1.8%）

---

## 🔧 API 端点

### 1. POST /api/batch/submit

提交请求到 batch queue。

**请求**:
```json
{
  "model": "mlx-community/gemma-3-12b-it-qat-4bit",
  "prompt": "Hello world",
  "max_tokens": 100
}
```

**响应**:
```json
{
  "request_id": "uuid-v4",
  "queue_position": 3,
  "status": "queued"
}
```

### 2. POST /api/batch/run

运行一个批次（处理所有 queued 请求）。

**请求**:
```json
{
  "model": "mlx-community/gemma-3-12b-it-qat-4bit"
}
```

**响应**:
```json
{
  "results": [
    {
      "request_id": "uuid-v4",
      "text": "Generated text...",
      "tokens": [123, 456, ...],
      "elapsed_sec": 10.5,
      "tok_s": 9.5
    }
  ]
}
```

### 3. GET /api/batch/stats

查询 batch engine 状态。

**请求**:
```
GET /api/batch/stats?model=mlx-community/gemma-3-12b-it-qat-4bit
```

**响应**:
```json
{
  "queue_size": 3,
  "active_requests": 5,
  "memory_gb": 7.5,
  "memory_peak_gb": 8.2
}
```

---

## 💡 使用示例

### Python 客户端

```python
import requests

BASE_URL = "http://localhost:8000"
MODEL = "mlx-community/gemma-3-12b-it-qat-4bit"

# 1. 提交请求
response = requests.post(
    f"{BASE_URL}/api/batch/submit",
    json={
        "model": MODEL,
        "prompt": "What is the capital of France?",
        "max_tokens": 50
    }
)
request_id = response.json()["request_id"]

# 2. 运行批次
response = requests.post(
    f"{BASE_URL}/api/batch/run",
    json={"model": MODEL}
)
results = response.json()["results"]

# 3. 获取结果
for r in results:
    print(f"{r['request_id']}: {r['text']}")
```

### cURL

```bash
# 提交请求
curl -X POST http://localhost:8000/api/batch/submit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-12b-it-qat-4bit",
    "prompt": "Hello world",
    "max_tokens": 100
  }'

# 运行批次
curl -X POST http://localhost:8000/api/batch/run \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-12b-it-qat-4bit"
  }'
```

---

## 🚀 启动 Hippo with Batch API

```bash
cd /Users/deepsearch/.openclaw/workspace/hippo
python3 -m hippo.cli
```

Hippo 启动后，Batch API 端点自动可用：
- `http://localhost:8000/api/batch/submit`
- `http://localhost:8000/api/batch/run`
- `http://localhost:8000/api/batch/stats`

---

## 🧪 测试

```bash
cd /Users/deepsearch/.openclaw/workspace/hippo
python3 test_hippo_batch_integration.py
```

测试覆盖：
- ✅ P2-1: Batch submit
- ✅ P2-2: Batch run
- ✅ P2-3: Batch stats
- ✅ P2-4: Queue full（7 个请求）

---

## 📁 架构

```
Hippo Server
  ├─ hippo/api.py (FastAPI 应用)
  │   └─ batch_api.router (新)
  │
  ├─ hippo/continuous_batch.py
  │   └─ ContinuousBatchEngine
  │       ├─ Request Queue (max 7)
  │       ├─ Batch Builder (padding)
  │       ├─ Batch Executor (prefill + decode)
  │       └─ Memory Monitor (>15GB 拒绝)
  │
  └─ hippo/routers/batch_api.py
      ├─ POST /api/batch/submit
      ├─ POST /api/batch/run
      └─ GET /api/batch/stats
```

---

## 🔒 安全限制

- **队列大小**: 最多 7 个并发请求（家族成员数）
- **内存限制**: >15GB 时拒绝新请求
- **超时**: 单个请求最多 120 秒
- **认证**: 需要配置 Hippo auth

---

## 📈 vs 现有 API

| API | 并发 | 吞吐 | 用途 |
|-----|------|------|------|
| **/api/generate** | 1 | 13.8 tok/s | 单用户、低延迟 |
| **/api/batch** | 7 | 33.7 tok/s | 多 Agent、高吞吐 |

**推荐**:
- 单用户场景 → `/api/generate`
- Multi-Agent 场景 → `/api/batch`

---

## 🎯 后续优化

**P3-1（优化）**: 按 prompt 长度分组
- 减少短 prompt 的 padding 浪费
- 预期收益：+10-20% 吞吐

**P3-2（监控）**: 添加 Prometheus metrics
- `hippo_batch_queue_size`
- `hippo_batch_memory_gb`
- `hippo_batch_tok_s`

**P3-3（日志）**: 结构化日志
- 请求 ID 追踪
- 性能指标记录

---

## 📚 相关文档

- P0 验收报告: `memory/SESSION_ARCHIVE_20260422_12B_BATCH_FORWARD.md`
- P1 验收报告: `memory/SESSION_ARCHIVE_20260422_P1_CONTINUOUS_BATCH_ENGINE.md`
- 家族会议 #75: `FAMILY_MEETING_20260422_P1_ACCEPTANCE.md`

---

**维护者**: 忒弥斯 (T-Mind) 🔮
**最后更新**: 2026-04-22 11:55
