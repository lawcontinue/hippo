"""
Hippo M0 记忆安全层 — source/confidence 来源审计 + 信任加权搜索.

熔炉#99 四层记忆安全架构 M0 最小实现。
不改变现有 schema，source/confidence 存在 metadata JSON 中。

用法:
    store = VectorStore("docs.db")
    # 添加带来源标记的文档
    store.add("重要事实", metadata={"source": "user", "confidence": 1.0})
    store.add("Agent推断", metadata={"source": "model", "confidence": 0.6})
    # 信任加权搜索
    results = search_with_confidence(store, "查询", top_k=5)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---- 来源类型常量 ----

SOURCE_USER = "user"              # 用户直接提供
SOURCE_MODEL = "model"            # Agent 推断
SOURCE_VERIFIED = "verified"      # 已验证（用户确认过的 model 推断）
SOURCE_EXTERNAL = "external"      # 外部文档/网页
SOURCE_SYSTEM = "system"          # 系统配置（不可衰减）

# 默认置信度
DEFAULT_CONFIDENCE = {
    SOURCE_USER: 1.0,
    SOURCE_MODEL: 0.5,
    SOURCE_VERIFIED: 0.9,
    SOURCE_EXTERNAL: 0.7,
    SOURCE_SYSTEM: 1.0,
}

# ---- 操作类型 Stake 分级（熔炉#108 S1 成果）----
# Stake 由操作性质定义，不由 memory importance 值定义。
# 打破"importance 决定 trust，trust 校验 importance"的循环论证。

class StakeLevel(Enum):
    """操作 stake 分级。基于操作性质，非历史记忆加权。"""
    LOW = "low"            # 只读、内部查询
    MEDIUM = "medium"      # 写入、状态更新
    HIGH = "high"          # 删除、外部发送、权限变更
    CRITICAL = "critical"  # 安全配置、密钥操作

_OPERATION_STAKE_MAP: Dict[str, StakeLevel] = {
    "read": StakeLevel.LOW,
    "search": StakeLevel.LOW,
    "list": StakeLevel.LOW,
    "write": StakeLevel.MEDIUM,
    "update": StakeLevel.MEDIUM,
    "create": StakeLevel.MEDIUM,
    "delete": StakeLevel.HIGH,
    "send_external": StakeLevel.HIGH,
    "grant_permission": StakeLevel.HIGH,
    "revoke_permission": StakeLevel.HIGH,
    "modify_config": StakeLevel.CRITICAL,
    "rotate_key": StakeLevel.CRITICAL,
    "publish_external": StakeLevel.CRITICAL,  # 发布到外部渠道（公众号/PR/issue）
}


def get_stake(operation: str) -> StakeLevel:
    """
    根据操作类型返回 stake 级别.

    与记忆 importance 解耦——stake 由操作性质定义，不由历史记忆加权。
    这是熔炉#108 的核心贡献之一（S1: 操作类型 Stake Routing）。
    """
    op_lower = operation.lower().strip()
    return _OPERATION_STAKE_MAP.get(op_lower, StakeLevel.MEDIUM)


def requires_multi_agent_trust(operation: str) -> bool:
    """判断当前操作是否需要 multi-agent 实时 trust scoring."""
    return get_stake(operation) in (StakeLevel.HIGH, StakeLevel.CRITICAL)


def tag_memory(
    store,
    doc_id: int,
    source: str = SOURCE_MODEL,
    confidence: Optional[float] = None,
    reviewed_by: Optional[str] = None,
) -> bool:
    """
    为已有文档添加 M0 来源审计标签.

    Args:
        store: VectorStore 实例
        doc_id: 文档 ID
        source: 来源类型 (user/model/verified/external/system)
        confidence: 置信度 0-1，None 则用来源默认值
        reviewed_by: 确认者标识（可选）

    Returns:
        True 如果标记成功

    Example:
        store.add("用户说他是律师")
        tag_memory(store, doc_id=1, source="user", confidence=1.0)
        # 后来用户确认了 Agent 的推断
        tag_memory(store, doc_id=5, source="verified", reviewed_by="user")
    """
    if confidence is None:
        confidence = DEFAULT_CONFIDENCE.get(source, 0.5)
    # P1-1 修复：confidence 范围验证
    confidence = max(0.0, min(1.0, float(confidence)))

    meta = store.get_metadata(doc_id)
    if meta is None:
        return False
    meta["source"] = source
    meta["confidence"] = confidence
    if reviewed_by:
        meta["reviewed_by"] = reviewed_by
        meta["reviewed_at"] = datetime.now(timezone.utc).isoformat()

    return store.update_metadata(doc_id, meta)


def add_with_source(
    store,
    text: str,
    source: str = SOURCE_MODEL,
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    reviewed_by: Optional[str] = None,
    batch_confirm_id: Optional[str] = None,
    batch_confirm_size: Optional[int] = None,
) -> int:
    """
    添加文档并自动标注来源审计标签.

    Args:
        store: VectorStore 实例
        text: 文档文本
        source: 来源类型
        confidence: 置信度，None 用默认值
        metadata: 额外 metadata
        reviewed_by: 确认者
        batch_confirm_id: 批量确认批次 ID（熔炉#108 S2: 标注替代折扣）
        batch_confirm_size: 该批次包含的确认条目总数

    Returns:
        doc_id

    Example:
        doc_id = add_with_source(store, "用户偏好深色模式", source="user")
        doc_id = add_with_source(store, "可能喜欢技术文章", source="model", confidence=0.4)
        # 批量确认：标注 batch context 但不打折扣
        for text in batch_texts:
            add_with_source(store, text, reviewed_by="user",
                           batch_confirm_id="batch_20260616",
                           batch_confirm_size=10)
    """
    if confidence is None:
        confidence = DEFAULT_CONFIDENCE.get(source, 0.5)
    confidence = max(0.0, min(1.0, float(confidence)))

    meta = metadata or {}
    meta["source"] = source
    meta["confidence"] = confidence
    if reviewed_by:
        meta["reviewed_by"] = reviewed_by
        meta["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    # 熔炉#108 S2: batch context 标注（信息完整性优先，不打折扣）
    if batch_confirm_id:
        meta["batch_confirm_id"] = batch_confirm_id
    if batch_confirm_size is not None:
        meta["batch_confirm_size"] = batch_confirm_size

    return store.add(text, metadata=meta)


def search_with_confidence(
    store,
    query: str,
    top_k: int = 5,
    min_confidence: float = 0.0,
    confidence_weight: float = 0.3,
    **kwargs,
) -> List:
    """
    信任加权搜索 — BM25/dense 分数 × confidence.

    熔炉#99 M0 核心功能：被标记为低信任的记忆自动沉底。

    Args:
        store: VectorStore 实例
        query: 搜索查询
        top_k: 返回数量
        min_confidence: 最低置信度过滤（0=不过滤）
        confidence_weight: confidence 对排序的影响权重 (0=忽略, 1=完全由confidence决定)
        **kwargs: 传给 store.search() 的额外参数

    Returns:
        Document 列表，score 已按 confidence 调整

    Example:
        # 低信任记忆即使 BM25 分高也排后面
        results = search_with_confidence(store, "律师", confidence_weight=0.3)
    """
    # 先取更多候选，然后按 confidence 重新排序
    raw_results = store.search(query, top_k=top_k * 3, **kwargs)


    adjusted = []
    for doc in raw_results:
        meta = doc.metadata or {}
        confidence = meta.get("confidence", 0.5)  # 未标记的默认 0.5
        if confidence < min_confidence:
            continue
        # 加权: 对非负分数用乘法，对负分数(BM25 log-likelihood)用加法偏移
        if doc.score >= 0:
            # dense cosine score (0-1): score *= (1 - w + w * confidence)
            adjusted_score = doc.score * (1.0 - confidence_weight + confidence_weight * confidence)
        else:
            # BM25 log-likelihood (negative): 推向更负 = 降权
            # confidence 越低，惩罚越大（减去 w * (1 - confidence) * |score|）
            penalty = confidence_weight * (1.0 - confidence) * abs(doc.score)
            adjusted_score = doc.score - penalty
        doc.score = round(adjusted_score, 6)
        # 附带 confidence 到 metadata 供调用方使用
        doc.metadata["_confidence"] = confidence
        adjusted.append(doc)

    adjusted.sort(key=lambda d: d.score, reverse=True)
    return adjusted[:top_k]


def decay_low_confidence(
    store,
    threshold: float = 0.3,
    days_old: int = 7,
) -> int:
    """
    M1 衰减层预览：降低低置信度 + 旧文档的 confidence.

    Args:
        store: VectorStore 实例
        threshold: 低于此 confidence 的才衰减
        days_old: 超过此天数的才衰减

    Returns:
    衰减的文档数量
    """
    import json as _json
    import sqlite3
    from datetime import datetime, timedelta

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    count = 0

    with sqlite3.connect(store.db_path) as conn:
        for row in conn.execute(
            "SELECT id, metadata, created_at FROM documents WHERE created_at < ?",
            (cutoff,),
        ):
            doc_id, meta_json, created = row
            meta = _json.loads(meta_json) if meta_json else {}
            confidence = meta.get("confidence", 0.5)
            source = meta.get("source", SOURCE_MODEL)

            # system 和 verified 不衰减
            if source in (SOURCE_SYSTEM, SOURCE_VERIFIED, SOURCE_USER):
                continue
            if confidence >= threshold:
                continue

            # P1-4 修复：幂等保护，24h 内只衰减一次
            last_decayed = meta.get("last_decayed")
            if last_decayed:
                try:
                    from datetime import timedelta
                    last_dt = datetime.fromisoformat(last_decayed)
                    if datetime.now(timezone.utc) - last_dt < timedelta(days=1):
                        continue  # 今天已经衰减过
                except (ValueError, TypeError):
                    pass

            # 衰减：confidence *= 0.9
            new_conf = confidence * 0.9
            meta["confidence"] = round(new_conf, 4)
            meta["last_decayed"] = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE documents SET metadata = ? WHERE id = ?",
                (_json.dumps(meta, ensure_ascii=False), doc_id),
            )
            count += 1

    # 重载内存
    store._load_all()
    return count


# ---- 隐性行为审计日志（熔炉#108 S2 + 规则#121）----
# 行为信号不直接修改 confidence/importance。
# 写入独立的 audit log，由独立 agent 周期性审查。

BEHAVIORAL_SIGNALS = [
    "user_closed_recommendation",    # 关闭推荐但未点差评
    "user_bought_competitor",        # 购买竞品但未卸载
    "user_skipped_suggestion",       # 跳过建议
    "user_ignored_repeated",         # 3+ 次忽略同一类推荐
    "user_corrected_output",         # 手动修正 Agent 输出
    "user_undo_action",              # 撤销 Agent 刚执行的操作
]


def log_behavioral_signal(
    store,
    signal_type: str,
    context_doc_ids: Optional[List[int]] = None,
    note: Optional[str] = None,
) -> bool:
    """
    写入隐性行为信号到 audit log.

    ⚠️ 不修改任何 memory importance/confidence。
    行为信号由独立审计 agent 周期性审查后决定是否调整信任权重。

    Args:
        store: VectorStore 实例
        signal_type: 信号类型 (见 BEHAVIORAL_SIGNALS)
        context_doc_ids: 相关文档 ID 列表（可选）
        note: 附加说明
    Returns:
        True 如果写入成功
    """
    if signal_type not in BEHAVIORAL_SIGNALS:
        return False

    meta = {
        "signal_type": signal_type,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "_audit_log": 1,  # SQLite json_extract 兼容的整数标记
    }
    if context_doc_ids:
        meta["context_doc_ids"] = context_doc_ids
    if note:
        meta["note"] = note

    doc_id = store.add(
        f"[BEHAVIORAL_AUDIT] {signal_type}",
        metadata=meta,
    )
    return doc_id > 0


def get_behavioral_signals(
    store,
    signal_type: Optional[str] = None,
    hours_back: int = 24,
) -> List[Dict[str, Any]]:
    """
    查询审计日志中的行为信号.

    Args:
        store: VectorStore 实例
        signal_type: 按类型过滤，None 则返回全部
        hours_back: 回溯时间（小时）
    Returns:
        行为信号字典列表 [{signal_type, logged_at, context_doc_ids, note}, ...]
    """
    import json as _json
    import sqlite3
    from datetime import datetime, timedelta

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    results = []

    with sqlite3.connect(store.db_path) as conn:
        query = (
            "SELECT id, text, metadata, created_at FROM documents "
            "WHERE json_extract(metadata, '$._audit_log') = 1 "
            "AND created_at > ? ORDER BY created_at DESC"
        )
        for row in conn.execute(query, (cutoff,)):
            doc_id, text, meta_json, created = row
            meta = _json.loads(meta_json) if meta_json else {}
            st = meta.get("signal_type", "")
            if signal_type and st != signal_type:
                continue
            results.append({
                "id": doc_id,
                "signal_type": st,
                "logged_at": created,
                "context_doc_ids": meta.get("context_doc_ids", []),
                "note": meta.get("note", ""),
                "raw": text,
            })
    return results


def batch_tag_with_context(
    store,
    texts: List[str],
    source: str = SOURCE_MODEL,
    reviewed_by: Optional[str] = None,
) -> Tuple[str, int, List[int]]:
    """
    批量添加文档并统一打上 batch_confirm_id.

    熔炉#108 S2 落地：batch context 标注替代折扣。
    每条记忆独立存储，但共享 batch_id 供下游 agent 自行判断。

    Args:
        store: VectorStore 实例
        texts: 批量文本列表
        source: 来源类型
        reviewed_by: 确认者
    Returns:
        (batch_id, size, [doc_ids])
    """
    import uuid
    batch_id = f"batch_{uuid.uuid4().hex[:8]}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    size = len(texts)
    if size == 0:
        return "", 0, []
    doc_ids = []
    for text in texts:
        doc_id = add_with_source(
            store, text,
            source=source,
            reviewed_by=reviewed_by,
            batch_confirm_id=batch_id,
            batch_confirm_size=size,
        )
        doc_ids.append(doc_id)
    return batch_id, size, doc_ids
