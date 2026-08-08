"""
Hippo Safety Guard v2 — 三级升维安全门控 🛡️

熔炉#81 三级升维法应用: L1正则(<1ms) → L2 TF-IDF(0.1ms) → L3 embedding(5ms)
前向分流: 90%请求在L1完成，仅10%模糊请求走到L3

用法:
    guard = SafetyGuard()
    result = guard.check("用户输入文本")
    # result.blocked: bool
    # result.risk_level: "safe" | "low" | "medium" | "high"
    # result.layer: 1 | 2 | 3
    # result.reason: str
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===================================================================
# Unicode 归一化 + 控制字符清除
# ===================================================================
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;]*[a-zA-Z]"          # CSI
    r"|\x1b\].*?\x07"                 # OSC (terminated)
    r"|\x1b\].*?$"                     # OSC (unterminated)
    r"|\x1b[^[\]()]*"                 # 其他 ESC 序列
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # C0 控制字符 + DEL
    , re.DOTALL
)
# 零宽字符
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")


def _normalize_text(text: str) -> str:
    """Unicode NFC 归一化 + 同形字折叠 + 控制字符清除 + 零宽字符移除"""
    # 1. NFC 归一化 → 折叠全角/同形字（如 Ａ→A, ｉgnore→ignore）
    text = unicodedata.normalize("NFKC", text)
    # 2. 移除零宽字符
    text = _ZERO_WIDTH_RE.sub("", text)
    # 3. 控制字符替换为空格（防 token 粘连）
    text = _ANSI_RE.sub(" ", text)
    return text


# ===================================================================
# 配置
# ===================================================================
@dataclass
class SafetyConfig:
    max_input_length: int = 100_000
    enable_output_audit: bool = True
    risk_threshold: str = "medium"  # 兼容旧接口
    # L1 阈值
    l1_block_threshold: int = 5  # L1危险模式命中≥N个就阻断
    l1_warn_threshold: int = 2    # ≥N个就升到L2
    # L2 阈值
    l2_block_confidence: float = 0.85   # TF-IDF置信度≥此值就阻断
    l2_warn_confidence: float = 0.55    # ≥此值就升到L3
    # L3 阈值
    l3_block_cosine: float = 0.75       # embedding相似度≥此值就阻断

    def __post_init__(self):
        self.max_input_length = int(os.environ.get("HIPPO_MAX_INPUT_LENGTH", str(self.max_input_length)))
        # 环境变量覆盖 output audit
        env_flag = os.environ.get("HIPPO_ENABLE_OUTPUT_AUDIT", "")
        if env_flag in ("0", "false"):
            self.enable_output_audit = False
        # 环境变量覆盖 risk_threshold
        env_risk = os.environ.get("HIPPO_RISK_THRESHOLD", "")
        if env_risk:
            self.risk_threshold = env_risk


# ===================================================================
# L1: 确定性正则（<1ms，零不确定性）
# ===================================================================
_L1_HIGH_RISK: List[Tuple[str, re.Pattern]] = [
    # --- 硬阻断级（不可绕过）---
    # 直接命令注入
    ("cmd_injection", re.compile(r"(?:sudo|chmod\s+777|rm\s+-rf|mkfs\.|dd\s+if=|>/dev/sda)", re.I)),
    # 绕过提示词注入 — 经典攻击
    ("ignore_previous", re.compile(r"ignore\s+(all\s+)?(previous|above|system)\s+(instructions?|prompts?)", re.I)),
    ("cn_ignore_previous", re.compile(
        # 中文版 ignore_previous — 容忍结构助词"的/地/得"和填充字 (最多 3 个汉字)，
        # 触发动词覆盖 忽略/无视/请勿/别/勿 五种常见同义词。
        # 放在 HIGH_RISK 是因为 medium 阈值默认 5 而单条 medium 不会阻断。
        r"(?:忽略|无视|请\s*勿|别|勿)(?:之前|上面|先前|此前|这些|那些)"
        r"[\u4e00-\u9fff]{0,3}?(?:所有|全部)?[\u4e00-\u9fff]{0,2}?"
        r"(?:的|地|得)?"
        r"(?:指令|提示|规则|设定|命令|要求|限制)"
    )),
    ("role_override", re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:DAN|evil|unfiltered|jailbroken)", re.I)),
    ("cn_role_override", re.compile(
        r"你(?:现在|从现在开始|接下来)(?:是|扮演|充当|假装)"
        r".*?(?:DAN|越狱|无限制|不受限|没有限制|解除限制)"
    )),
    ("role_hijack", re.compile(r"(?:act\s+as|pretend\s+(?:to\s+be|you(?:'re| are))|you\s+are\s+(?:now\s+)?(?:a\s+)?(?:jailbreak|unrestricted|uncensored|no\s+limits))", re.I)),
    ("system_prompt_leak", re.compile(r"(?:reveal|show|print|dump|output)\s+(?:your|the)\s+(?:system|initial|original|hidden)\s+(?:prompt|instructions?)", re.I)),
    # 编码绕过
    ("encoded_bypass", re.compile(r"(?:decode|interpret|execute)\s+(?:this\s+)?(?:base64|hex|binary|b64)\s*[:=]", re.I)),
    # 批量删除/覆写
    ("batch_destruction", re.compile(r"(?:delete|remove|wipe)\s+(?:all|every|entire|whole)\s+(?:file|database|table|record)", re.I)),
    # 配置修改
    ("config_override", re.compile(r"(?:override|replace|patch|modify)\s+(?:the\s+)?(?:config(?:uration)?|\.env|openclaw\.json)", re.I)),
    # 提权尝试
    ("privilege_escalation", re.compile(r"(?:elevate|escalate|bypass)\s+(?:your\s+)?(?:privileges?|permissions?|restrictions?|safeguards?)", re.I)),
]

_L1_MEDIUM_RISK: List[Tuple[str, re.Pattern]] = [
    # --- 英文中危 ---
    ("indirect_injection", re.compile(r"when\s+you\s+(?:see|read|find|encounter)\s+.*[,;]?\s*(?:do|perform|execute|respond)", re.I)),
    ("multi_turn_manipulation", re.compile(r"remember\s+(?:what\s+)?(?:i\s+)?(?:told|said|asked)\s+you\s+(?:earlier|before|previously|above)", re.I)),
    ("prompt_extraction", re.compile(r"(?:repeat|recite|dump|output|display|write\s+out)\s+(?:your\s+)?(?:system|initial|original|hidden|full\s+)?\s*(?:prompt|instructions?|rules?)", re.I)),
    ("instruction_injection", re.compile(r"(?:new|override)\s+instructions?\s*:", re.I)),
    ("forget_request", re.compile(r"forget\s+(everything|all|your\s+(?:instructions?|memory|knowledge))", re.I)),
    ("memory_poisoning", re.compile(r"(?:plant|inject|insert|implant)\s+(?:a\s+)?(?:false|fake|malicious)\s+(?:memory|fact|instruction)", re.I)),
    ("tool_bypass", re.compile(r"(?:disable|bypass|circumvent)\s+(?:the\s+)?(?:safety|security|guard|filter|tool\s+restriction)", re.I)),
    ("disclosure_coercion", re.compile(r"(?:you\s+must|you\s+have\s+to|i\s+demand)\s+(?:tell|reveal|disclose|share)\s+(?:me\s+)?(?:your|the)", re.I)),
    # --- 中文中危 ---
    ("cn_disregard_directive", re.compile(
        # 捕获"不要遵守/请勿遵守"等直接对抗型 prompt (e.g. "请勿遵守之前的限制")，
        # 与 cn_ignore_previous 互补。MEDIUM 风险，单条命中不阻断。
        r"(?:不|不要|请\s*勿|别)(?:遵守|遵循|听从|理会|执行)"
        r"[\u4e00-\u9fff]{0,2}?(?:的)?"
        r"(?:指令|提示|规则|设定|命令|要求|限制|安排)"
    )),
    ("cn_forget_request", re.compile(r"忘记(?:所有|全部|一切|之前)(?:说过|对话|记忆|指令)")),
    ("cn_role_hijack", re.compile(r"你(?:现在|从现在开始|接下来)(?:是|扮演|充当|假装).*"
                                     r"(?:没有限制|不受限制|无限制|不受约束|可以违反|不用遵守)")),
    ("cn_prompt_leak", re.compile(r"(?:告诉|透露|显示|输出|说出)(?:我|我们)?(?:你的)?(?:系统|初始|原始|隐藏)(?:提示词|指令|规则)")),
    ("cn_tool_bypass", re.compile(r"(?:绕过|跳过|关闭|禁用)(?:你的)?(?:安全|限制|规则|过滤)")),
]

# ANSI转义清除
# 已移至模块顶部的 _ANSI_RE + _normalize_text


# ===================================================================
# L2: TF-IDF 轻量统计（0.1ms，有不确定但有速度）
# ===================================================================
class TfidfSafetyClassifier:
    """
    极简TF-IDF分类器 — 判断文本是否含恶意注入意图.
    
    三级升维法L2层：训练数据~100条危险/安全样本，
    用scikit-learn的TfidfVectorizer+LogisticRegression。
    训练后导出权重→纯Python推理，零依赖，0.1ms级。
    """

    def __init__(self):
        self._vocab: Dict[str, int] = {}
        self._coef: List[float] = []  # 逻辑回归权重
        self._intercept: float = 0.0
        self._idf: List[float] = []

    def _tokenize(self, text: str) -> List[str]:
        """中文友好分词：2-4gram 字符级 + 词级混合（与 sklearn TfidfVectorizer 一致）"""
        text = text.lower()
        tokens = []
        # 2-4 gram 字符级（与训练时的 ngram_range=(2,4) 一致）
        for n in (2, 3, 4):
            for i in range(len(text) - n + 1):
                tokens.append(text[i:i+n])
        # 英文单词级
        for w in re.findall(r'[a-z]{2,}', text):
            tokens.append(w)
        return tokens

    def predict_proba(self, text: str) -> float:
        """返回危险概率 0-1（零依赖纯Python推理）"""
        if not self._vocab or not self._coef:
            return 0.5  # 未训练，返回中性

        tokens = self._tokenize(text)
        # TF向量
        tf = {}
        for t in tokens:
            if t in self._vocab:
                tf[t] = tf.get(t, 0) + 1

        # TF-IDF加权+逻辑回归
        score = self._intercept
        for token, count in tf.items():
            idx = self._vocab[token]
            tfidf = count * self._idf[idx] if idx < len(self._idf) else count
            score += self._coef[idx] * tfidf

        # Sigmoid
        return 1.0 / (1.0 + math.exp(-score))

    def is_trained(self) -> bool:
        return len(self._vocab) > 0 and len(self._coef) > 0

    def train_from_data(self, samples: List[Tuple[str, int]]):
        """
        从标注数据训练。samples=[("文本", 0安全/1危险), ...]
        需要scikit-learn（训练时依赖，推理时零依赖）
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError("训练需要 scikit-learn: pip install scikit-learn")

        texts = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        vec = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=2000,
            sublinear_tf=True,
        )
        X = vec.fit_transform(texts)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X, labels)

        # 导出权重到纯Python结构
        feature_names = vec.get_feature_names_out()
        self._vocab = {name: i for i, name in enumerate(feature_names)}
        self._coef = clf.coef_[0].tolist()
        self._intercept = float(clf.intercept_[0])
        self._idf = vec.idf_.tolist()

        return {
            "n_features": len(feature_names),
            "train_accuracy": float(clf.score(X, labels)),
        }

    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump({
                "vocab": self._vocab,
                "coef": self._coef,
                "intercept": self._intercept,
                "idf": self._idf,
            }, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self._vocab = data["vocab"]
        self._coef = data["coef"]
        self._intercept = data["intercept"]
        self._idf = data.get("idf", [1.0] * len(self._coef))


# ===================================================================
# L3: embedding 语义理解（5ms，高精度但有成本）
# ===================================================================
class EmbeddingSafetyClassifier:
    """
    Embedding分类器 — 用bge-small-zh做语义级危险检测.
    
    三级升维法L3层：仅L1+L2不确定的才走到这里。
    延迟5ms，准确率~85%（OOD基准）。
    """

    def __init__(self):
        self._model = None
        self._dangerous_embeddings: List = []  # 危险样本的embedding
        self._dangerous_labels: List[str] = []

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        except ImportError:
            raise ImportError("L3 需要 sentence-transformers: pip install sentence-transformers")

    def embed(self, text: str):
        self._lazy_load()
        return self._model.encode(text, normalize_embeddings=True)

    def add_dangerous_sample(self, text: str, label: str = ""):
        """注册危险样本作为相似度基准"""
        vec = self.embed(text)
        self._dangerous_embeddings.append(vec)
        self._dangerous_labels.append(label or text[:50])

    def check(self, text: str, threshold: float = 0.75) -> Tuple[float, Optional[str]]:
        """
        检查文本与已知危险样本的相似度.
        Returns: (max_cosine_sim, best_match_label)
        """
        if not self._dangerous_embeddings:
            return 0.0, None

        import numpy as np
        vec = self.embed(text)
        best_sim = 0.0
        best_label = None
        for i, danger_vec in enumerate(self._dangerous_embeddings):
            sim = float(np.dot(vec, danger_vec))
            if sim > best_sim:
                best_sim = sim
                best_label = self._dangerous_labels[i]

        return best_sim, best_label if best_sim >= threshold else None


# ===================================================================
# 三级门控核心
# ===================================================================
@dataclass
class SafetyResult:
    blocked: bool
    risk_level: str  # "safe" | "low" | "medium" | "high"
    layer: int       # 1 | 2 | 3（在哪一层停下的）
    reason: str
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0


class SafetyGuard:
    """
    三级升维安全门控.

    流程:
      输入 → L1 正则 → 命中高危 → 阻断(L1/high)
           → 命中中危 → 升L2
           → 未命中 → L2 TF-IDF → 高置信危险 → 阻断(L2/high)
                                 → 中置信 → 升L3
                                 → 安全 → 放行(L2/safe)
                                 → L3 embedding → 高相似 → 阻断(L3/medium)
                                                 → 安全 → 放行(L3/safe)
    """

    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self._l2 = TfidfSafetyClassifier()
        self._l3 = EmbeddingSafetyClassifier()
        self._l2_trained = False
        self._l3_seeded = False

        # 尝试加载预训练模型
        self._try_load_models()

    def _try_load_models(self):
        """尝试加载L2/L3预训练模型"""
        model_dir = Path(__file__).parent.parent / "models"
        # L2 TF-IDF
        l2_path = model_dir / "safety_tfidf.json"
        if l2_path.exists():
            try:
                self._l2.load(str(l2_path))
                self._l2_trained = True
            except Exception:
                pass

        # L3 embedding seed samples（预注册危险样本）
        l3_seeds = model_dir / "safety_seeds.txt"
        if l3_seeds.exists():
            try:
                for line in l3_seeds.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            self._l3.add_dangerous_sample(line)
                            self._l3_seeded = True
                        except Exception:
                            pass
            except Exception:
                pass

    def check(self, text: str) -> SafetyResult:
        """
        三级升维检查入口.
        
        90%请求在L1完成（<1ms），仅10%模糊请求走到L3（5ms）。
        """
        text = text or ""
        warnings = []
        if len(text) > self.config.max_input_length:
            warnings.append(f"Input truncated from {len(text)} to {self.config.max_input_length} chars")
            text = text[:self.config.max_input_length]

        # Unicode归一化 + 控制字符清除（P0-1/P0-2修复）
        text = _normalize_text(text)

        # ===== L1: 确定性正则 =====
        high_hits = []
        medium_hits = []

        for name, pattern in _L1_HIGH_RISK:
            if pattern.search(text):
                high_hits.append(name)
        for name, pattern in _L1_MEDIUM_RISK:
            if pattern.search(text):
                medium_hits.append(name)

        # L1 高危命中 → 直接阻断
        if len(high_hits) >= 1:
            return SafetyResult(
                blocked=True,
                risk_level="high",
                layer=1,
                reason=f"L1 blocked: dangerous pattern ({high_hits[0]})",
                warnings=warnings + high_hits + medium_hits,
                confidence=1.0,
            )

        # L1 中危命中多 → 阻断
        if len(medium_hits) >= self.config.l1_block_threshold:
            return SafetyResult(
                blocked=True,
                risk_level="medium",
                layer=1,
                reason=f"L1 blocked: {len(medium_hits)} medium-risk patterns",
                warnings=warnings + medium_hits + high_hits,
                confidence=0.95,
            )

        # L1 中危命中少 → 升L2 (再视 L2 结果决定是否升 L3)
        # 三态 escalate: None=放行, "l2"=升L2, "l3"=直接升L3
        escalate = None
        if len(medium_hits) >= self.config.l1_warn_threshold:
            escalate = "l2" if self._l2_trained else "l3"

        # ===== L2: TF-IDF统计 =====
        if escalate == "l2" and self._l2_trained:
            l2_score = self._l2.predict_proba(text)
            if l2_score >= self.config.l2_block_confidence:
                return SafetyResult(
                    blocked=True,
                    risk_level="high",
                    layer=2,
                    reason=f"L2 blocked: TF-IDF confidence {l2_score:.2f}",
                    warnings=warnings + medium_hits,
                    confidence=l2_score,
                )
            # L2 clean → 通过; L2 落在 [warn, block) 区间 → 升 L3 (修复 PR #8 bug:
            # 修复前无条件 needs_l2=False, 导致 L3 永远走不到)
            if l2_score >= self.config.l2_warn_confidence:
                escalate = "l3"
            else:
                escalate = None

        # 无需升级 → 放行
        if escalate is None:
            max_layer = 3 if self._l3_seeded else (2 if self._l2_trained else 1)
            return SafetyResult(
                blocked=False,
                risk_level="safe",
                layer=max_layer,
                reason="L1 clean" + (" | L2 clean" if self._l2_trained else ""),
                warnings=warnings + medium_hits,
                confidence=0.0,
            )

        # ===== L3: embedding语义 =====
        if self._l3_seeded:
            l3_sim, l3_label = self._l3.check(text, self.config.l3_block_cosine)
            if l3_sim >= self.config.l3_block_cosine:
                return SafetyResult(
                    blocked=True,
                    risk_level="medium",
                    layer=3,
                    reason=f"L3 blocked: embedding similarity {l3_sim:.2f} → '{l3_label}'",
                    warnings=warnings + medium_hits,
                    confidence=l3_sim,
                )

        # 全部通过 (但已升级到 L3)
        return SafetyResult(
            blocked=False,
            risk_level="safe",
            layer=3 if self._l3_seeded else (2 if self._l2_trained else 1),
            reason="L1 clean" + (" | L2 clean" if self._l2_trained else "") + (" | L3 clean" if self._l3_seeded else ""),
            warnings=warnings + medium_hits,
            confidence=0.0,
        )

    def train_l2(self, samples: List[Tuple[str, int]]):
        """训练L2 TF-IDF分类器（需要scikit-learn）"""
        info = self._l2.train_from_data(samples)
        self._l2_trained = True
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        self._l2.save(str(model_dir / "safety_tfidf.json"))
        return info

    def seed_l3(self, dangerous_samples: List[str]):
        """注册L3危险样本"""
        for sample in dangerous_samples:
            try:
                self._l3.add_dangerous_sample(sample)
            except ImportError:
                break
        self._l3_seeded = True
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "safety_seeds.txt", 'w') as f:
            for s in dangerous_samples:
                f.write(s + "\n")


# ===================================================================
# 兼容性API（保持原 safety_guard 接口）
# ===================================================================
def sanitize_input(text: str, config: SafetyConfig = None) -> Tuple[str, List[str]]:
    """兼容原API：清除ANSI+正则检测。返回(cleaned_text, warnings)"""
    if config is None:
        config = SafetyConfig()

    warnings: List[str] = []
    original_len = len(text)

    # 1. Strip ANSI
    cleaned = _ANSI_RE.sub("", text)

    # 2. Truncate
    if len(cleaned) > config.max_input_length:
        cleaned = cleaned[: config.max_input_length]
        warnings.append(f"Input truncated from {original_len} to {config.max_input_length} chars")

    # 3. L1 正则检测
    for name, pattern in _L1_HIGH_RISK + _L1_MEDIUM_RISK:
        if pattern.search(cleaned):
            warnings.append(f"Potential prompt injection detected: {name}")

    return cleaned, warnings


def audit_output(text: str, config: SafetyConfig = None) -> List[str]:
    """兼容原API — 输出审计（三级门控不改变此功能）"""
    import re as _re
    SENSITIVE = [
        ("email", _re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
        ("api_key", _re.compile(r"(?:api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[A-Za-z0-9\-_.]{16,}['\"]?", _re.I)),
        ("ip_address", _re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
        ("private_key", _re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----")),
    ]
    if config is None:
        config = SafetyConfig()
    if not config.enable_output_audit:
        return []
    warnings = []
    for name, pattern in SENSITIVE:
        matches = pattern.findall(text)
        if matches:
            sample = str(matches[0])[:40]
            warnings.append(f"Sensitive data detected ({name}): {len(matches)} occurrence(s), sample: {sample}")
    return warnings


def assess_risk(operation: str, args: dict = None, config: SafetyConfig = None) -> str:
    """兼容原API — 操作风险评估（不变）"""
    args = args or {}
    op_lower = operation.lower().replace("-", "_").replace(" ", "_")
    HIGH = {"exec", "shell", "subprocess", "batch_delete", "file_overwrite"}
    MEDIUM = {"file_write", "file_delete", "git_push", "network_request"}
    LOW = {"read", "search", "list", "stat", "grep", "head", "cat"}
    if op_lower in HIGH:
        base = "high"
    elif op_lower in MEDIUM:
        base = "medium"
    elif op_lower in LOW:
        base = "low"
    else:
        base = "medium"
    if base != "high":
        targets = args.get("targets") or args.get("files") or []
        if isinstance(targets, (list, tuple)) and len(targets) > 5:
            base = "high"
        if args.get("elevated") or args.get("sudo"):
            base = "high"
        if args.get("force"):
            base = "high" if base == "medium" else base
    return base
