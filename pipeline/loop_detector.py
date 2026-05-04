#!/usr/bin/env python3
"""
loop_detector.py — Line-level thinking loop detector for LLM streaming output.

Detects semantic repetition at line boundaries — catches patterns that
token-level repeat_penalty misses (same meaning, different phrasing).

Independent module, no backend dependencies.
"""

from __future__ import annotations

# Minimal English stop words — enough for Jaccard filtering
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "and", "or", "but", "of", "in", "on", "at", "by", "for",
    "with", "as", "from", "that", "this", "it", "its", "also", "need",
    "we", "our", "us", "i", "my", "me", "he", "she", "they", "them",
    "their", "his", "her", "has", "have", "had", "do", "does", "did",
    "not", "no", "so", "if", "then", "than", "can", "will", "would",
})


class LoopDetector:
    """Line-level thinking loop detector for LLM streaming output.

    Detects semantic repetition at line boundaries — catches patterns that
    token-level repeat_penalty misses (same meaning, different phrasing).

    Default config: window=20 lines, threshold=3 matches, similarity=0.7
    """

    def __init__(self, window: int = 20, threshold: int = 3,
                 similarity: float = 0.7, action: str = "escape"):
        """
        Args:
            window: sliding window of recent lines to check against
            threshold: number of similar lines to trigger detection
            similarity: Jaccard similarity threshold (0-1) for fuzzy matching
            action: what to do on detection — "escape", "stop", "warn"
        """
        self.window = window
        self.threshold = threshold
        self.similarity = similarity
        self.action = action
        self._lines: list[str] = []       # completed lines
        self._line_tokens: list[set] = []  # tokenized sets for completed lines
        self._buffer: str = ""             # current incomplete line
        self._triggered = False

    def feed(self, token_text: str) -> dict | None:
        """Feed a token's text. Returns detection info dict if loop detected.

        Call this for every token in the stream. Internally buffers lines
        (flushes on \\n boundary).

        Returns on detection:
            {"loop": True, "line": "...", "count": N, "action": "escape"|"stop"|"warn"}
        """
        if self._triggered and self.action == "stop":
            return None  # already stopped, don't re-trigger

        self._buffer += token_text

        # Flush complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line or len(line) < 20:
                continue  # skip empty / short lines
            if self._check_line(line):
                self._triggered = True
                return {
                    "loop": True,
                    "line": line,
                    "count": self.threshold,
                    "action": self.action,
                }

        return None

    def _tokenize(self, text: str) -> set:
        """Simple whitespace tokenization for Jaccard similarity.
        Remove common stop words, lowercase, take set."""
        words = text.lower().split()
        return {w for w in words if w not in _STOP_WORDS and len(w) > 1}

    def _jaccard(self, set_a: set, set_b: set) -> float:
        """Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    def _check_line(self, line: str) -> bool:
        """Check if this line is semantically similar to >= threshold lines in window."""
        tokens = self._tokenize(line)
        if not tokens:
            return False

        # Check against window
        window_lines = self._line_tokens[-self.window:] if self._line_tokens else []
        match_count = 0
        for existing_tokens in window_lines:
            if self._jaccard(tokens, existing_tokens) >= self.similarity:
                match_count += 1

        # Add to buffer
        self._lines.append(line)
        self._line_tokens.append(tokens)

        # Only trigger if we found enough matches AND there are enough lines
        # (match_count >= threshold means this line is similar to `threshold` previous lines)
        return match_count >= self.threshold

    def reset(self):
        """Clear buffer. Call between requests."""
        self._lines.clear()
        self._line_tokens.clear()
        self._buffer = ""
        self._triggered = False

    def check_text(self, text: str) -> dict | None:
        """Post-hoc check on a complete text (for non-streaming mode).

        Returns detection info if loop found, None otherwise.
        """
        self.reset()
        # Feed line by line
        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) < 20:
                continue
            if self._check_line(line):
                self._triggered = True
                return {
                    "loop": True,
                    "line": line,
                    "count": self.threshold,
                    "action": self.action,
                }
        return None
