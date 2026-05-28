"""
Hippo Tokenizer — whitespace + CJK single-char tokenization with stopword filtering.

Dependencies: none (stdlib only)
"""

from __future__ import annotations

import re
import string
from typing import List

__all__ = ["default_tokenizer"]

EN_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "am", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "so", "if", "as", "no", "just", "about", "up", "out", "over", "into",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their",
})

ZH_STOPWORDS = frozenset({
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
})

# CJK Unified Ideographs + Hiragana + Katakana + Hangul
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)

_ALL_STOPWORDS = EN_STOPWORDS | ZH_STOPWORDS


def default_tokenizer(text: str) -> List[str]:
    """Whitespace + CJK single-char tokenization with stopword removal."""
    if not text:
        return []

    tokens: List[str] = []

    # Extract CJK chars first (single char tokens)
    cjk_chars = set()
    for m in _CJK_RE.finditer(text):
        ch = m.group()
        cjk_chars.add(m.start())
        if ch not in ZH_STOPWORDS:
            tokens.append(ch)

    # For non-CJK portions: split on whitespace, strip punctuation, lowercase
    cleaned = []
    for i, ch in enumerate(text):
        if i in cjk_chars:
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    rest = "".join(cleaned)

    for word in rest.split():
        word = word.strip(string.punctuation).lower()
        if word and word not in EN_STOPWORDS:
            tokens.append(word)

    return tokens
