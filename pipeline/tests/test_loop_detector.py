#!/usr/bin/env python3
"""Tests for loop_detector.py — no real model needed."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loop_detector import LoopDetector


def test_no_loop():
    """Normal varied text should not trigger."""
    d = LoopDetector(window=20, threshold=3, similarity=0.7)
    lines = [
        "The quick brown fox jumps over the lazy dog\n",
        "Python is a versatile programming language\n",
        "Machine learning models require large datasets\n",
        "The weather is nice and sunny today\n",
        "Database optimization improves query performance\n",
    ]
    for line in lines:
        result = d.feed(line)
        assert result is None, f"False positive on: {line}"
    print("[OK] test_no_loop")


def test_exact_repeat():
    """Exact same line repeated 3+ times should trigger."""
    d = LoopDetector(window=20, threshold=3, similarity=0.7)
    repeated = "I need to check the configuration settings carefully\n"
    # Feed varied lines first to fill window
    for _ in range(5):
        d.feed("Some varied text that is different each time number " + str(_) + "\n")
    # Now repeat the same line
    for i in range(4):
        result = d.feed(repeated)
    # After 3+ repeats, should trigger
    assert result is not None and result["loop"], "Should detect exact repetition"
    print("[OK] test_exact_repeat")


def test_semantic_similarity():
    """Semantically similar but differently phrased lines should trigger."""
    d = LoopDetector(window=20, threshold=3, similarity=0.5)
    variants = [
        "I think we need to analyze this problem more carefully now\n",
        "I think we should analyze this problem more carefully now\n",
        "I think we must analyze this problem more carefully now\n",
        "I think we ought to analyze this problem more carefully now\n",
    ]
    result = None
    for line in variants:
        result = d.feed(line)
    assert result is not None and result["loop"], f"Should detect semantic loop: {result}"
    print("[OK] test_semantic_similarity")


def test_enumeration_no_false_positive():
    """Enumerated list items should NOT trigger."""
    d = LoopDetector(window=20, threshold=3, similarity=0.7)
    items = [
        "1. Set up the development environment and tools\n",
        "2. Configure the database connection parameters\n",
        "3. Deploy the application to production servers\n",
        "4. Monitor system performance and user feedback\n",
        "5. Optimize code based on profiling results gathered\n",
    ]
    for item in items:
        result = d.feed(item)
        assert result is None, f"False positive on enumeration: {item}"
    print("[OK] test_enumeration_no_false_positive")


def test_short_lines_ignored():
    """Lines < 20 chars should not trigger."""
    d = LoopDetector(window=20, threshold=3, similarity=0.7)
    shorts = ["ok\n", "yes\n", "sure\n", "done\n"] * 5
    for s in shorts:
        result = d.feed(s)
        assert result is None, f"Short line triggered: {s}"
    print("[OK] test_short_lines_ignored")


def test_reset_clears_buffer():
    """After reset, previous lines should not cause detection."""
    d = LoopDetector(window=20, threshold=3, similarity=0.7)
    repeated = "This is a repeated line for testing purposes here\n"
    for _ in range(5):
        d.feed(repeated)

    d.reset()

    # After reset, same line repeated once should not trigger (need threshold=3)
    for i in range(2):
        result = d.feed(repeated)
    assert result is None, "Should not trigger after reset with only 2 repeats"
    print("[OK] test_reset_clears_buffer")


def test_token_by_token():
    """Feeding token by token with \\n boundary should work."""
    d = LoopDetector(window=20, threshold=3, similarity=0.7)
    repeated = "We need to reconsider our approach to the problem"
    # Feed line 1 token by token
    tokens = repeated.split()
    for i, tok in enumerate(tokens):
        result = d.feed(tok + (" " if i < len(tokens) - 1 else ""))
        assert result is None
    d.feed("\n")

    # Feed same line 3 more times
    for rep in range(3):
        for i, tok in enumerate(tokens):
            result = d.feed(tok + (" " if i < len(tokens) - 1 else ""))
        result = d.feed("\n")

    # On the 3rd or 4th repeat, should trigger
    assert d._triggered, "Should detect loop when feeding token by token"
    print("[OK] test_token_by_token")


if __name__ == "__main__":
    test_no_loop()
    test_exact_repeat()
    test_semantic_similarity()
    test_enumeration_no_false_positive()
    test_short_lines_ignored()
    test_reset_clears_buffer()
    test_token_by_token()
    print("\nAll 7 tests passed!")
