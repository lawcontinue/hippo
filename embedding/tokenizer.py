"""
Hippo Tokenizer — whitespace + CJK single-char + bigram tokenization with stopword filtering.

v2.1 (2026-06-22): Added HIPPO_BIGRAM=0 env rollback switch + CJK punctuation filtering.
v2 (2026-06-22): Added CJK bigram tokens to improve BM25 discrimination.
  Tokenize "四层分级架构" → ['四','层','分','级','架','构','四层','层分','分级','级架','架构']

Dependencies: none (stdlib only)
"""

from __future__ import annotations

import os
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
    "都", "一", "上", "也", "很", "到", "说", "要", "去",
})

# CJK punctuation (full-width, filtered to avoid leaking as tokens)
_CJK_PUNCT = frozenset("，。！？；：""''【】《》（）…—～「」『』、")

# Unicode blocks split by script to avoid cross-script bigrams
_CJK_HANZI = re.compile(r"[\u4e00-\u9fff]+")
_CJK_HIRAGANA = re.compile(r"[\u3040-\u309f]+")
_CJK_KATAKANA = re.compile(r"[\u30a0-\u30ff]+")
_CJK_HANGUL = re.compile(r"[\uac00-\ud7af]+")

# Combined pattern for Phase 2 (non-CJK whitespace splitting)
_CJK_COMBINED = re.compile(
    r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+"
)

# Bigram rollback switch: set HIPPO_BIGRAM=0 to revert to v1 unigrams-only
_BIGRAM_ENABLED = os.environ.get("HIPPO_BIGRAM", "1") != "0"


def _process_cjk_segment(chars: List[str]) -> List[str]:
    """Generate unigrams and optionally bigrams from a CJK character sequence.
    
    Filters CJK punctuation and stopwords.
    """
    filtered = [ch for ch in chars if ch not in ZH_STOPWORDS and ch not in _CJK_PUNCT]
    if not filtered:
        return []
    
    tokens = list(filtered)  # unigrams
    
    if _BIGRAM_ENABLED and len(filtered) >= 2:
        for i in range(len(filtered) - 1):
            tokens.append(filtered[i] + filtered[i + 1])
    
    return tokens


def default_tokenizer(text: str) -> List[str]:
    """Whitespace + CJK unigram + bigram tokenization with stopword/punctuation filtering.
    
    Transformation: "四层分级架构" →
      unigrams: ['四','层','分','级','架','构']  (minus stopwords)
      bigrams:  ['四层','层分','分级','级架','架构']
    
    Bigrams dramatically improve BM25 IDF for Chinese compound terms
    because "分级" co-occurs in far fewer docs than "分" or "级" alone.
    
    Rollback: Set HIPPO_BIGRAM=0 to revert to v1 unigrams-only mode.
    """
    if not text:
        return []

    tokens: List[str] = []

    # Phase 1: CJK unigrams + bigrams, per-script to avoid cross-script pairs
    for regex in (_CJK_HANZI, _CJK_HIRAGANA, _CJK_KATAKANA, _CJK_HANGUL):
        for match in regex.finditer(text):
            tokens.extend(_process_cjk_segment(list(match.group())))

    # Phase 2: whitespace tokens for non-CJK portions
    # Note: _CJK_COMBINED already covers CJK spans; remaining CJK punctuation
    #   in the whitespace-split portion is stripped below.
    _punct_set = string.punctuation + "，。！？；：“”‘’《》（）…—～「」『』、"
    rest = _CJK_COMBINED.sub(" ", text)
    for word in rest.split():
        word = word.strip(_punct_set).lower()
        if word and word not in EN_STOPWORDS:
            tokens.append(word)

    return tokens
