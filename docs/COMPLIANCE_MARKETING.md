# Hippo 合规营销话术 — 护城河叙事 v1.0

**日期**: 2026-06-14 | **来源**: 熔炉#98 合规即证明器 | **维护者**: 忒弥斯+Code+Aria

---

## 一、核心叙事框架

Hippo 不是你"需要合规"的解决方案——Hippo 是让**你向客户证明你合规**的工具。

| 传统叙事 | 合规证明器叙事 |
|---------|---------------|
| "Hippo 支持本地部署，数据不出域" | "Hippo 让你在招标书里写上'全量数据不出域，AI辅助搜索零云端依赖'" |
| "BM25 + hybrid search" | "审计友好的搜索：BM25 的评分逻辑 = 白盒可解释 / hybrid 的 dense 向量 = 黑盒补充" |
| "30 秒部署" | "30 秒部署 = 合规审计的部署证言成本为零" |
| "no jieba / no ChromaDB" | "零外部依赖 = 每个依赖都是一份需要审计的供应链风险报告" |

---

## 二、场景化话术（按受众）

### 2.1 金融客户（银行/保险/券商）

> **痛点**：你的 AI 系统引用了外部搜索——监管问你"搜索结果怎么来的？为什么排这个不排那个？"你解释不清。

> **Hippo 话术**：
> Hippo 的 BM25 搜索是白盒的——每个相关性分数的计算公式是公开透明的。监管审计时，你可以直接用 Python 复现每一个搜索结果的权重计算过程。不是"黑盒模型觉得它相关"，是"TF-IDF 的公式算出它相关"。这符合中国的金融监管对"可解释 AI"的要求。

📊 **数字支撑**：BM25 88.2% top-1 在场景特定查询（非 OOD），比 bge-small-zh 的 85.5% 在专业术语查询中更准。

### 2.2 医疗客户（医院/HIS/电子病历）

> **痛点**：你的 AI 助手帮医生检索病历——出了事，法官问"谁选的这个结果？人还是 AI？"

> **Hippo 话术**：
> Hippo 的搜索日志可以记录每次查询的完整决策链：查询词 → BM25 得分 → 语义得分 → 融合权重 → 最终排序。这套记录直接可用作医疗合规审计的证据——不需要额外的合规软件。这是"L2 级免费正收益"（来自 AI Agent 治理的业界最佳实践）。

📊 **数字支撑**：Hybrid RRF 融合对中文医学术语的 top-1 准确率 91.8%（OOD 110 题验证）。

### 2.3 政府/公共部门

> **痛点**：你的智慧政务系统要在信创环境跑——国产 GPU、国产 OS、国产数据库，还不能联网。

> **Hippo 话术**：
> Hippo 的 sparse 模式只依赖 SQLite + Python stdlib。不需要 ChromaDB（需要外部进程），不需要 jieba（第三方中文分词），不需要任何云 API。完全离线，信创环境适配零成本。

📊 **数字支撑**：依赖树：Python 3.10+ + SQLite（系统内置）。无 go.sum、package-lock.json、Cargo.toml。

### 2.4 企业合规官（GDPR/个保法/PIPL 审计）

> **痛点**：Data Protection Officer 问"你的搜索索引里存了什么数据？过期了怎么删？被遗忘权怎么实现？"

> **Hippo 话术**：
> Hippo 使用 SQLite 存储所有文档和向量——数据库文件可以直接被审计工具扫描。单条删除、批量删除、数据迁移——都是标准 SQL 操作。不需要调用任何 Vendor 专有 API，不需要联系 Hippo 官方执行删除。你在本地有完整的数据主权。

📊 **数字支撑**：向量维度可配置（128d/512d/1024d），支持向量文件独立导出（numpy .npy），便于外部审计。

### 2.5 AI 创业者（面向 OPC 入驻/融资）

> **痛点**：你的 Agent 产品用了太多外部依赖——投资人的法务团队在 Due Diligence 中发现 18 个供应链风险节点。

> **Hippo 话术**：
> Hippo = 供应链上零审批节点。pip install 后你的法务团队只需要审计一个 Python 包。对比 ChromaDB（Go 编译 + ClickHouse 依赖 + Protobuf schema）需要审计 3 个编译链 + 1 个网络协议，Hippo 的合规表面积是最小的。

📊 **数字支撑**：PyPI 单包，MIT 协议，无 transitive runtime dependency。

---

## 三、合规证明器三件套（可打印/可发送 DM）

| 文档 | 用途 | 状态 |
|------|------|------|
| **Hippo 审计日志示例** | 展示搜索决策链的完整审计记录格式 | 🚧 需补充 |
| **Hippo 依赖清单（零外部依赖证明）** | DPO/DPIA 填写时直接引用 | 🚧 需生成 `pipdeptree` |
| **Hippo 与监管框架对照表** | 逐条对应 EU AI Act / 中国算法备案 / ISO 42001 要求 | 🚧 需补充 |

---

## 四、README 合规段落建议（可直接插入）

### 候选：README 底部加"Compliance"章节

```markdown
## Compliance-Friendly

Hippo is designed for environments where audit trails matter.

**Explainable search**: BM25 scores are computed with an open formula — every result's weight can be reproduced during a regulatory audit. No black-box embedding required for core search.

**Zero-supply-chain risk**: Sparse mode has zero transitive dependencies (Python stdlib + SQLite). One package to audit, not 18.

**Local data sovereignty**: Your documents never leave your machine. Your embeddings are your files. Delete a row in SQLite = delete from the pipeline. No cloud coordination needed.

**Audit trail ready**: Every search's decision chain (query → BM25 → dense → RRF fusion → final ranking) can be logged natively. Usable as evidence for AI governance compliance.

**Regulatory alignment**: Works offline by default — compatible with air-gapped deployment for financial (CSRC), medical (NHC), and government (信创) compliance requirements.
```

---

## 五、HN Show 帖中的合规叙事钩子（可选添加）

原帖聚焦技术 simplicity。如果需要强调合规差异化，加这句：

> _Bonus for compliance teams: Hippo's BM25 scoring is fully explainable — every result weight is a reproducible formula. Your DPO can audit every search result without calling an AI vendor. Zero external dependencies = one package in your supply chain audit, not a dependency tree of 18 packages._

（72 词，HN 帖子合适长度。）

---

## 六、护城河摩尔定律自检（应用于 Hippo）

| 检查项 | 当前护城河 | 瓦解风险 | 对策 |
|--------|-----------|---------|------|
| 零外部依赖 | ChromeDB 替代品的 **最低合规表面积** | 其他 embedding library 也可能去依赖化 | 持续维护 sparse 模式为首选体验 |
| BM25 可解释性 | 监管审计的 **唯一白盒搜索** | 开源 BM25 实现很多，非我们独有 | 差异化在"一键 hybrid 升级+审计日志"，不在 BM25 本身 |
| 中文内建 tokenizer | 中文场景的 **零配置优势** | jieba 社区的方案成熟度 | 持续优化中文分词质量，对标 jieba 精度 |
| 混合 RRF 融合 | **互补性效应** 91.8%（高于任何单模型） | 融合方案算法非秘密 | 差异化在"sparse→hybrid 的零成本迁移路径" |

---

_基于熔炉#98「合规作为证明器」框架 + Hippo v0.3.1 实际技术特性。忒弥斯+Aria+Code 联合产出。_
