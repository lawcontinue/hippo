# Hippo Scheduler — 多设备星型调度器

**拓扑**: 星型，Mac mini (R0) = 调度中心，工作节点 = 5060Ti / R1
**粒度**: 任务级
**通信**: REST (HTTP)
**设计原则**: 先跑起来，拿真实数据，再优化

## 快速开始

```bash
# 启动调度器
python -m scheduler.server

# 提交任务
curl -X POST http://localhost:8090/tasks -H 'Content-Type: application/json' -d '{
  "type": "chat",
  "payload": {"messages": [{"role": "user", "content": "hello"}]}
}'

# 查看状态
curl http://localhost:8090/status
```
