"""Tests for hippo.embedding.memory_safety (M0 来源审计 + 信任加权搜索)."""

import os
import tempfile

import pytest

from embedding.store import VectorStore
from embedding.memory_safety import (
    SOURCE_USER,
    SOURCE_MODEL,
    SOURCE_VERIFIED,
    SOURCE_SYSTEM,
    add_with_source,
    tag_memory,
    search_with_confidence,
    decay_low_confidence,
)


@pytest.fixture
def store():
    """Create a fresh sparse VectorStore for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = VectorStore(path, mode="sparse")
    yield s
    # Close the SQLite connection before deleting the .db, otherwise Windows
    # raises WinError 32 because the file is still locked by the connection.
    s.close()
    if os.path.exists(path):
        os.unlink(path)


class TestAddWithSource:
    def test_add_user_source(self, store):
        doc_id = add_with_source(store, "用户说他是律师", source=SOURCE_USER)
        assert doc_id > 0
        entry = store._entry_map[doc_id]
        import json
        meta = json.loads(entry[2])
        assert meta["source"] == "user"
        assert meta["confidence"] == 1.0

    def test_add_model_source_default_confidence(self, store):
        doc_id = add_with_source(store, "可能喜欢技术文章", source=SOURCE_MODEL)
        entry = store._entry_map[doc_id]
        import json
        meta = json.loads(entry[2])
        assert meta["source"] == "model"
        assert meta["confidence"] == 0.5

    def test_add_with_custom_confidence(self, store):
        doc_id = add_with_source(store, "不确定的推断", source=SOURCE_MODEL, confidence=0.3)
        entry = store._entry_map[doc_id]
        import json
        meta = json.loads(entry[2])
        assert meta["confidence"] == 0.3

    def test_add_with_reviewed_by(self, store):
        doc_id = add_with_source(
            store, "已验证的事实", source=SOURCE_VERIFIED, reviewed_by="admin"
        )
        entry = store._entry_map[doc_id]
        import json
        meta = json.loads(entry[2])
        assert meta["reviewed_by"] == "admin"
        assert "reviewed_at" in meta


class TestTagMemory:
    def test_tag_existing_doc(self, store):
        doc_id = store.add("未标记文档")
        result = tag_memory(store, doc_id, source=SOURCE_USER, confidence=0.95)
        assert result is True
        entry = store._entry_map[doc_id]
        import json
        meta = json.loads(entry[2])
        assert meta["source"] == "user"
        assert meta["confidence"] == 0.95

    def test_tag_nonexistent_doc(self, store):
        result = tag_memory(store, 99999, source=SOURCE_USER)
        assert result is False

    def test_tag_upgrades_to_verified(self, store):
        """Agent 推断 → 用户确认 → 升级为 verified."""
        doc_id = add_with_source(store, "Agent 推断用户是程序员", source=SOURCE_MODEL)
        # 后来用户确认了
        tag_memory(store, doc_id, source=SOURCE_VERIFIED, reviewed_by="user")
        entry = store._entry_map[doc_id]
        import json
        meta = json.loads(entry[2])
        assert meta["source"] == "verified"
        assert meta["confidence"] == 0.9
        assert meta["reviewed_by"] == "user"


class TestSearchWithConfidence:
    def test_low_confidence_sinks(self, store):
        """低置信度记忆即使 BM25 匹配好也排后面."""
        add_with_source(store, "律师是法律专业人士，处理过很多案件", source=SOURCE_USER, confidence=1.0)
        add_with_source(store, "律师可能喜欢吃面条", source=SOURCE_MODEL, confidence=0.1)

        results = search_with_confidence(store, "律师", top_k=2, confidence_weight=0.8)
        assert len(results) == 2
        # 高置信度应该排前面
        assert results[0].metadata.get("source") == "user"
        assert results[0].score > results[1].score

    def test_min_confidence_filter(self, store):
        """min_confidence 过滤低信任结果."""
        add_with_source(store, "高信任文档", source=SOURCE_USER, confidence=0.9)
        add_with_source(store, "低信任文档", source=SOURCE_MODEL, confidence=0.2)

        results = search_with_confidence(
            store, "文档", top_k=5, min_confidence=0.5, confidence_weight=0.0
        )
        assert len(results) == 1
        assert results[0].metadata.get("source") == "user"

    def test_zero_weight_equals_raw_search(self, store):
        """confidence_weight=0 等于原始搜索."""
        add_with_source(store, "测试文档A", source=SOURCE_USER, confidence=1.0)
        add_with_source(store, "测试文档B", source=SOURCE_MODEL, confidence=0.1)

        raw = store.search("测试", top_k=5)
        weighted = search_with_confidence(store, "测试", top_k=5, confidence_weight=0.0)
        assert len(raw) == len(weighted)
        # scores should be equal
        for r, w in zip(raw, weighted):
            assert abs(r.score - w.score) < 0.001

    def test_untagged_defaults_to_0_5(self, store):
        """未标记 source 的文档默认 confidence=0.5."""
        store.add("未标记文档内容")
        results = search_with_confidence(store, "未标记", top_k=1)
        assert len(results) == 1
        assert results[0].metadata.get("_confidence") == 0.5


class TestDecayLowConfidence:
    def test_system_never_decays(self, store):
        """system 记忆永不衰减."""
        add_with_source(store, "系统规则：不做违法操作", source=SOURCE_SYSTEM, confidence=0.2)
        # 即使 confidence 低 + 老，也不应该衰减
        count = decay_low_confidence(store, threshold=0.5, days_old=0)
        assert count == 0

    def test_user_never_decays(self, store):
        """user 记忆不衰减."""
        add_with_source(store, "用户偏好", source=SOURCE_USER, confidence=0.1)
        count = decay_low_confidence(store, threshold=0.5, days_old=0)
        assert count == 0

    def test_low_confidence_model_decays(self, store):
        """低置信度 model 记忆会被衰减."""
        add_with_source(store, "不确定的推断", source=SOURCE_MODEL, confidence=0.2)
        count = decay_low_confidence(store, threshold=0.5, days_old=0)
        assert count == 1
        # 验证 confidence 被降低
        entry = list(store._entry_map.values())[0]
        import json
        meta = json.loads(entry[2])
        assert meta["confidence"] < 0.2
        assert meta["confidence"] == round(0.2 * 0.9, 4)
