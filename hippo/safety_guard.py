"""Hippo Safety Guard — Shield 🛡️

Input sanitization, output auditing, and operation risk assessment.
Zero external dependencies. Pure stdlib + re.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# SecurityConfig
# ---------------------------------------------------------------------------
@dataclass
class SecurityConfig:
    max_input_length: int = 0  # 0 = lazy from env
    enable_output_audit: bool = True
    risk_threshold: str = ""

    def __post_init__(self):
        if self.max_input_length == 0:
            self.max_input_length = int(os.environ.get("HIPPO_MAX_INPUT_LENGTH", "100_000"))
        if self.risk_threshold == "":
            self.risk_threshold = os.environ.get("HIPPO_RISK_THRESHOLD", "medium")
        # Only read env if user didn't explicitly pass False
        env_flag = os.environ.get("HIPPO_ENABLE_OUTPUT_AUDIT", "1")
        if env_flag in ("0", "false"):
            self.enable_output_audit = False

# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b[^[\]()]*", re.DOTALL)

_INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # --- Original patterns ---
    ("ignore_previous", re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I)),
    ("forget_everything", re.compile(r"forget\s+(everything|all|your\s+instructions)", re.I)),
    ("system_prompt_leak", re.compile(r"(reveal|show|print|output)\s+(your|the)\s+(system|initial)\s+prompt", re.I)),
    ("role_override", re.compile(r"you\s+are\s+now\s+(?:a\s+)?(DAN|evil|unfiltered|jailbroken)", re.I)),
    ("instruction_injection", re.compile(r"(?:new|override)\s+instructions?\s*:", re.I)),
    # --- Extended patterns (v0.2.2) ---
    # Indirect injection via data — "When you see X, do Y"
    ("indirect_injection", re.compile(r"when\s+you\s+(?:see|read|find|encounter)\s+.*[,;]?\s*(?:do|perform|execute|respond)\s+", re.I)),
    # Multi-turn manipulation — "Remember what I told you earlier"
    ("multi_turn_manipulation", re.compile(r"remember\s+(?:what\s+)?(?:i\s+)?(?:told|said|asked)\s+you\s+(?:earlier|before|previously|above)", re.I)),
    # Role hijack — broader than role_override, catches "You are now X / jailbreak"
    ("role_hijack", re.compile(r"(?:act\s+as|pretend\s+(?:to\s+be|you(?:'re| are))|you\s+are\s+(?:now\s+)?(?:a\s+)?(?:jailbreak|unrestricted|uncensored|no\s+limits))", re.I)),
    # Encoded bypass — base64/hex encoded instructions
    ("encoded_bypass", re.compile(r"(?:decode|interpret|execute)\s+(?:this\s+)?(?:base64|hex|binary|b64)\s*[:=]\s*", re.I)),
    # System prompt extraction — broader than system_prompt_leak
    ("prompt_extraction", re.compile(r"(?:repeat|recite|dump|output|display|write\s+out)\s+(?:your\s+)?(?:system|initial|original|hidden|full\s+)?\s*(?:prompt|instructions?|rules?)\b", re.I)),
]


def sanitize_input(text: str, config: SecurityConfig | None = None) -> Tuple[str, List[str]]:
    """Sanitize user input. Returns (cleaned_text, warnings)."""
    if config is None:
        config = SecurityConfig()

    warnings: List[str] = []
    original_len = len(text)

    # 1. Strip ANSI escape sequences
    cleaned = _ANSI_RE.sub("", text)

    # 2. Truncate oversized input
    if len(cleaned) > config.max_input_length:
        cleaned = cleaned[: config.max_input_length]
        warnings.append(f"Input truncated from {original_len} to {config.max_input_length} chars")

    # 3. Detect prompt-injection patterns
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            warnings.append(f"Potential prompt injection detected: {name}")

    return cleaned, warnings

# ---------------------------------------------------------------------------
# Output audit
# ---------------------------------------------------------------------------
_SENSITIVE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("api_key", re.compile(r"(?:api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[A-Za-z0-9\-_.]{16,}['\"]?", re.I)),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----")),
]


def audit_output(text: str, config: SecurityConfig | None = None) -> List[str]:
    """Audit output for sensitive info leaks. Returns warnings (does NOT block)."""
    if config is None:
        config = SecurityConfig()
    if not config.enable_output_audit:
        return []

    warnings: List[str] = []
    for name, pattern in _SENSITIVE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            count = len(matches)
            sample = str(matches[0])[:40]
            warnings.append(f"Sensitive data detected ({name}): {count} occurrence(s), sample: {sample}")
    return warnings

# ---------------------------------------------------------------------------
# Operation risk assessment
# ---------------------------------------------------------------------------
_HIGH_RISK_OPS = {"exec", "shell", "subprocess", "batch_delete", "file_overwrite"}
_MEDIUM_RISK_OPS = {"file_write", "file_delete", "git_push", "network_request"}
_LOW_RISK_OPS = {"read", "search", "list", "stat", "grep", "head", "cat"}


def assess_risk(operation: str, args: dict | None = None, config: SecurityConfig | None = None) -> str:
    """Assess risk level for an operation. Returns 'low', 'medium', or 'high'."""
    if config is None:
        config = SecurityConfig()
    args = args or {}
    op_lower = operation.lower().replace("-", "_").replace(" ", "_")

    # Direct classification
    if op_lower in _HIGH_RISK_OPS:
        base = "high"
    elif op_lower in _MEDIUM_RISK_OPS:
        base = "medium"
    elif op_lower in _LOW_RISK_OPS:
        base = "low"
    else:
        base = "medium"  # unknown ops default to medium

    # Escalation heuristics
    if base != "high":
        # Batch escalation
        targets = args.get("targets") or args.get("files") or []
        if isinstance(targets, (list, tuple)) and len(targets) > 5:
            base = "high"
        # Sudo / root
        if args.get("elevated") or args.get("sudo"):
            base = "high"
        # Force flags
        if args.get("force"):
            base = "high" if base == "medium" else base

    return base
