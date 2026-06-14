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
from typing import Any, Dict, List, Optional

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

    Returns:
        doc_id

    Example:
        doc_id = add_with_source(store, "用户偏好深色模式", source="user")
        doc_id = add_with_source(store, "可能喜欢技术文章", source="model", confidence=0.4)
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
