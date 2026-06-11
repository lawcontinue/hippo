# Hippo 首发内容日历（3 周计划）

基于熔炉 #87 共识：首发后持续节奏 > 首发 4-10x

## Week 1：首发周

| Day | 动作 | 平台 | 内容 |
|-----|------|------|------|
| D1 | **HN 首发** | news.ycombinator.com | Show HN 文案（已备于 LAUNCH_KIT.md），24h 在线回复 |
| D2 | **Reddit** | r/LocalLLaMA + r/MachineLearning | 两版文案（已备），回复所有评论 |
| D3 | **中文平台** | 掘金 + V2EX | 中文博客（已备于 docs/blog_cn_launch.md） |
| D5 | **v0.3.2 发布** | GitHub + PyPI | Bug fix（基于首发反馈）+ 中文 BM25 benchmark 补充 |

### 博客 1（D3 发布）："花 3800 块的显卡跑 30B 模型，还自带混合搜索"
- 已写：`docs/blog_cn_launch.md`
- 英文版待写

## Week 2：技术深挖周

| Day | 动作 | 内容 |
|-----|------|------|
| D8 | **博客 2**（英文） | "How Pipeline Parallelism Works Over Plain TCP" |
| D9 | 博客 2 中文版 | 掘金同步 |
| D10 | **v0.3.3 发布** | 新功能：sentence-transformers engine 优化 + 中文分词 benchmark |
| D12 | **博客 3**（英文） | "BM25 + Dense Hybrid Search: Why RRF Beats Weighted Sum" |

### 博客 2 大纲：Pipeline Parallelism Over TCP
- 问题：模型太大单机放不下
- 方案：层切分 + TCP 通信 + 自动内存预算
- 对比：vs MPI（复杂度）vs Tensor Parallelism（需要 NVLink）
- 代码：10 行启动双机推理
- Benchmark：单机 vs 双机延迟/吞吐

### 博客 3 大纲：Hybrid Search 为什么比纯向量好
- 问题：纯向量搜索漏关键词匹配
- 方案：BM25 + dense + RRF 融合
- 实验：中文文档集，纯向量 vs 纯 BM25 vs hybrid
- Benchmark：Hit@1/Hit@3/Hit@5
- 为什么不用加权求和（权重敏感）

## Week 3：生态周

| Day | 动作 | 内容 |
|-----|------|------|
| D15 | **博客 4**（中英双语） | "One pip install for RAG: Hippo vs Ollama+ChromaDB+LangChain" |
| D17 | **v0.4.0 发布** | Multi-shard（>2 设备）或 speculative decoding 预览 |
| D19 | **Reddit AMA** | r/LocalLLaMA 问答帖 |

### 博客 4 大纲：Hippo vs 传统 RAG 栈
- 问题：RAG 最小依赖是什么？
- 对比：Hippo（1 install）vs Ollama + ChromaDB + LangChain（3 services）
- 安装步骤对比（截图）
- 功能矩阵：inference / embedding / BM25 / dense / ANN / Chinese
- 什么时候用 ChromaDB（>100K 文档、分布式）
- 什么时候用 Hippo（<10K 文档、个人/小团队、离线）

## 版本计划

| 版本 | 时间 | 内容 |
|------|------|------|
| v0.3.2 | Week 1 D5 | Bug fix + benchmark 补充 |
| v0.3.3 | Week 2 D10 | sentence-transformers 优化 + 中文 benchmark |
| v0.4.0 | Week 3 D17 | Multi-shard 或 speculative decoding |

## 成功指标检查点

| 指标 | 7 天检查 | 14 天检查 | 30 天终评 |
|------|---------|----------|----------|
| GitHub star | ≥15 | ≥30 | ≥50 |
| PyPI 周下载 | ≥30 | ≥60 | ≥100 |
| 外部 Issue/PR | ≥1 | ≥2 | ≥3 |
| 博客阅读 | ≥500 | ≥1200 | ≥2000 |
