# Qwen3-30B-A3B Q3_K_M 生成质量评测 + 循环检测验证

**日期**: 2026-05-04 | **引擎**: llama-cpp-python 0.3.22 | **硬件**: RTX 5060 Ti 16GB
**配置**: n_ctx=8192, max_tokens=3900, temperature=0.7, n_gpu_layers=-1

---

## 测试概要

| 指标 | 值 |
|------|-----|
| 循环率 | **0/10 (0%)** — llama-cpp vs Ollama 同模型 78% |
| Loop Detector 误报 | **0/10 (0%)** |
| 平均速度 | **85 tok/s** |
| 平均 token | 1389 |
| 正文占比 | 20-40%（thinking 消耗 60-80%）|

---

## 10 题详情 + 评价

### Q1: Hippo 技术博客（英文）

**Prompt**: Write a 500-word technical blog post introducing Hippo...
**输出**: tok=1388, 正文=3919字符, speed=82.7tok/s
**评价**: ⭐⭐⭐ 7/10
- ✅ 结构完整：标题 + 段落 + 子标题
- ✅ 关键 feature 覆盖（cross-platform, loop detection）
- ⚠️ 偏模板化，缺代码示例和真实 benchmark 数据
- ⚠️ 未达到 500 word 目标（正文约 350 词）
- **发布建议**: 补充 Hippo 实际 benchmark 数据 + 代码片段

---

### Q2: 试用期普法文章（中文）

**Prompt**: 写一篇800字的普法宣传文章，主题是'试用期被辞退'...
**输出**: tok=878, 正文=627字符, speed=90.4tok/s
**评价**: ⭐⭐⭐⭐ 8.5/10 — **最佳输出**
- ✅ 法条引用准确（劳动合同法第21条、第39条）
- ✅ 语言通俗，适合普通上班族
- ✅ 有真实感案例（科技公司代码测试、迟到被辞退）
- ⚠️ 正文太短（627字符，目标800字）
- ⚠️ 缺少结尾3条实用建议
- **发布建议**: 需补充结尾建议部分，适当扩展到800字

---

### Q3: 加班不给钱视频脚本（中文）

**Prompt**: 写一个30秒搞笑普法短视频的分镜脚本，主题是'老板说加班不给钱'...
**输出**: tok=1666, 正文=829字符, speed=83.0tok/s
**评价**: ⭐⭐⭐ 7.5/10
- ✅ 3个镜头结构清晰，有画面描述+台词
- ✅ 反转笑点（员工掏出劳动法书）
- ✅ 最后5秒普法内容
- ⚠️ 笑点偏温和，不够"抖音"
- ⚠️ 台词可以更口语化
- **发布建议**: 加强笑点（加网络流行语），台词更接地气

---

### Q4: MLX vs llama.cpp 对比博客（英文）

**Prompt**: Write a 600-word comparison blog post: running LLM locally...
**输出**: tok=1828, 正文=3732字符, speed=81.5tok/s
**评价**: ⭐⭐⭐ 7/10
- ✅ 结构好：intro + 对比维度 + recommendation table
- ✅ 覆盖 setup complexity 和 speed
- ⚠️ Benchmark 数据是编造的（模型编了数字）
- ⚠️ 缺真实测试数据支撑
- **发布建议**: 替换为我们的实测数据（8B: 42tok/s Mac / 71tok/s 5060Ti）

---

### Q5: AI 代码法律风险（中文）

**Prompt**: 写一篇800字的文章，分析个人开发者使用AI生成代码可能面临的法律风险...
**输出**: tok=1441, 正文=1045字符, speed=85.0tok/s
**评价**: ⭐⭐⭐ 7.5/10
- ✅ 三点覆盖：开源协议冲突、版权归属、侵权责任
- ✅ 面向技术社区读者，语言专业但不晦涩
- ⚠️ 案例偏少，以分析为主
- ⚠️ 缺具体法律条文引用
- **发布建议**: 补充 GitHub Copilot 诉讼案等真实案例

---

### Q6: 租房押金视频脚本（中文）

**Prompt**: 写一个30秒搞笑普法短视频脚本，主题是'房东不退押金怎么办'...
**输出**: tok=1018, 正文=522字符, speed=88.8tok/s
**评价**: ⭐⭐⭐⭐ 8/10
- ✅ 开头3秒 hook 抓眼球（巨型行李箱 + 红色标语）
- ✅ 搞笑对话（房东荒谬理由）
- ✅ 竖屏拍摄适配
- ✅ 法律武器清晰（民法典第714条）
- ⚠️ 正文偏短
- **发布建议**: 可直接使用，微调台词即可

---

### Q7: Loop Detection 技术博客（英文）

**Prompt**: Write a technical deep-dive blog post (600 words) about thinking loop detection...
**输出**: tok=1862, 正文=4833字符, speed=80.9tok/s
**评价**: ⭐⭐⭐ 7/10
- ✅ 技术深度够（Jaccard + line-level + code snippet 思路）
- ✅ 解释了 token-level 失败原因
- ⚠️ 代码片段是概念性的，不是可运行的
- ⚠️ 缺 Hippo 的实际检测数据
- **发布建议**: 用 Hippo 的 loop_detector.py 实际代码替换，加实测数据

---

### Q8: Hippo README（英文）

**Prompt**: Write a compelling GitHub README intro section for Hippo...
**输出**: tok=960, 正文=1811字符, speed=89.3tok/s
**评价**: ⭐⭐⭐⭐ 8/10
- ✅ tagline 简洁有力
- ✅ 4 key features 覆盖完整
- ✅ comparison table vs Ollama and vLLM
- ⚠️ Quick-start code example 缺失
- **发布建议**: 补充 quick-start 代码示例后可直接使用

---

### Q9: 网上骂人视频脚本（中文）

**Prompt**: 写一个搞笑普法短视频脚本（30秒），主题是'网上骂人被起诉'...
**输出**: tok=800, 正文=418字符, speed=90.9tok/s
**评价**: ⭐⭐⭐ 7.5/10
- ✅ 键盘侠形象夸张，适合 B 站
- ✅ 网络流行语自然（"一键三连"）
- ✅ 反转到位（法院传票）
- ⚠️ 正文最短（418字符），内容不完整
- **发布建议**: 需扩展完整脚本

---

### Q10: 三模型对比（英文）

**Prompt**: Write a detailed comparison of three local LLM models...
**输出**: tok=2050, 正文=2324字符, speed=79.6tok/s
**评价**: ⭐⭐⭐ 7/10
- ✅ 评分框架合理（中文写作、技术准确、创意、速度）
- ✅ 有 winner recommendation
- ⚠️ Benchmark 数字全部编造（模型自己猜的）
- ⚠️ 与我们实测数据不一致
- **发布建议**: 用实测数据替换所有 benchmark 数字

---

## 总体结论

### 模型质量

| 场景 | 评分 | 可用性 |
|------|------|--------|
| 中文普法文章 | ⭐⭐⭐⭐ 8.5/10 | ✅ 可直接使用（需补短） |
| 中文视频脚本 | ⭐⭐⭐⭐ 7.8/10 | ✅ 可直接使用 |
| 英文技术博客 | ⭐⭐⭐ 7/10 | ⚠️ 需补充真实数据 |
| 英文 README | ⭐⭐⭐⭐ 8/10 | ✅ 可直接使用 |

### 关键发现

1. **Thinking 消耗 60-80% token**：2000 token 中正文只有 400-800 字符
2. **中文生成质量 > 英文**：普法文章和视频脚本质量明显好于英文博客
3. **事实准确性存疑**：Q4/Q10 的 benchmark 数据是编造的，不能直接用
4. **Loop Detector 零误报**：20 次生成（10+10）未触发一次误报
5. **llama-cpp vs Ollama**：同模型同硬件，循环率 0% vs 78%
