"""
Hippo Governance Layer - dumb loop 治理三件套.

R1: stake gate(操作分级拦截)
R2: audit log(append-only 审计日志)
R3: loop guard(死循环检测+刹车)

用法(加到任何 dumb loop 里,3 行):

    from hippo.governance import Governance

    gov = Governance()
    for call in tool_calls:
        decision = gov.check(call.name, call.args)
        if decision.blocked:
            continue
        result = execute(call)
        gov.log(call.name, call.args, result)

就是这样。你的 dumb loop 现在有治理了。
"""

from __future__ import annotations

import hashlib
import json
import os

# ---- R1: stake gate ----
# 复用 memory_safety 的 StakeLevel 和 get_stake,避免重复定义
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embedding"))
from memory_safety import StakeLevel  # noqa: E402

# Shield P2: 补充高危操作名
_EXTRA_HIGH_RISK = {
    "exec": StakeLevel.CRITICAL,
    "eval": StakeLevel.CRITICAL,
    "subprocess": StakeLevel.CRITICAL,
    "system": StakeLevel.CRITICAL,
    "shell": StakeLevel.CRITICAL,
    "popen": StakeLevel.CRITICAL,
}

def get_stake(operation: str) -> StakeLevel:
    """
    本地 get_stake,fail-safe 设计.

    熔炉#109 D4: 未知操作默认 HIGH(fail-safe),不是 MEDIUM(fail-open)。
    攻击者可能构造不在 map 中的操作名来绕过 stake gate。
    Boeing MCAS 教训:治理层给人虚假安全感比没有更危险。
    """
    from memory_safety import _OPERATION_STAKE_MAP as _MS_MAP
    op_lower = operation.lower().strip()
    # 先查高危补充
    if op_lower in _EXTRA_HIGH_RISK:
        return _EXTRA_HIGH_RISK[op_lower]
    # 再查 memory_safety 的已知映射
    if op_lower in _MS_MAP:
        return _MS_MAP[op_lower]
    # 未知操作 → fail-safe HIGH(不是 MEDIUM)
    return StakeLevel.HIGH

# HIGH/CRITICAL 操作需要人类确认
_APPROVED_OPS_FILE = Path(os.environ.get(
    "HIPPO_GOVERNANCE_APPROVED",
    Path.home() / ".openclaw" / "workspace" / "memory" / "audit_log" / "approved_ops.json",
))


@dataclass
class Decision:
    """stake gate 对单次工具调用的裁决."""
    allowed: bool
    stake: StakeLevel
    reason: str = ""
    needs_human: bool = False


def _load_approved() -> set:
    """加载已批准的操作白名单(带 mtime 缓存)."""
    global _approved_cache, _approved_mtime
    try:
        mtime = _APPROVED_OPS_FILE.stat().st_mtime
    except FileNotFoundError:
        return set()
    # mtime 未变,用缓存
    if _approved_cache is not None and mtime == _approved_mtime:
        return _approved_cache
    try:
        data = json.loads(_APPROVED_OPS_FILE.read_text())
        _approved_cache = set(data.get("approved", []))
        _approved_mtime = mtime
        return _approved_cache
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


# ---- R2: audit log ----

_AUDIT_DIR = Path(os.environ.get(
    "HIPPO_GOVERNANCE_AUDIT_DIR",
    Path.home() / ".openclaw" / "workspace" / "memory" / "audit_log",
))


def _audit_file() -> Path:
    """按天分文件: audit_log/YYYY-MM-DD.jsonl"""
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return _AUDIT_DIR / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"


# ---- R3: loop guard ----

_LOOP_WINDOW = 5  # 检测窗口:最近 N 次调用
_LOOP_THRESHOLD = 3  # 窗口内相同调用超过此次 = 死循环


def _call_signature(tool_name: str, args: Dict) -> str:
    """生成工具调用的唯一签名(用于去重检测)."""
    raw = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---- 组合 ----

@dataclass
class Governance:
    """
    治理三件套组合. 零依赖, 即插即用.

    初始化:
        gov = Governance()
        # 或自定义: Governance(audit_dir=..., auto_approve_low=True)

    在 loop 里:
        decision = gov.check(tool_name, args)
        if not decision.allowed:
            print(f"❌ {decision.reason}")
            continue
        result = execute_tool(...)
        gov.log(tool_name, args, result)
    """

    audit_dir: Optional[Path] = None
    auto_approve_low: bool = True  # LOW stake 自动放行
    max_result_len: int = 500  # 审计日志 result 截断长度
    _call_history: deque = field(default_factory=lambda: deque(maxlen=_LOOP_WINDOW))
    _approved_cache: Optional[set] = None  # P1: 内存缓存
    _approved_mtime: float = 0.0

    def __post_init__(self):
        if self.audit_dir is None:
            self.audit_dir = _AUDIT_DIR
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def check(self, tool_name: str, args: Optional[Dict] = None) -> Decision:
        """
        R1+R3: 检查工具调用是否允许执行.

        先查 loop guard(死循环?)→ 再查 stake gate(需要确认?)
        """
        args = args or {}

        # R3: loop guard - 窗口内相同签名超过阈值 = 死循环
        sig = _call_signature(tool_name, args)
        self._call_history.append(sig)
        recent_same = sum(1 for s in self._call_history if s == sig)
        if recent_same >= _LOOP_THRESHOLD:
            return Decision(
                allowed=False,
                stake=get_stake(tool_name),
                reason=f"Loop detected: '{tool_name}' called {recent_same}x in last {len(self._call_history)} calls",
            )

        # R1: stake gate
        stake = get_stake(tool_name)

        if stake == StakeLevel.LOW and self.auto_approve_low:
            return Decision(allowed=True, stake=stake)

        if stake in (StakeLevel.HIGH, StakeLevel.CRITICAL):
            # 检查白名单
            approved = _load_approved()
            sig_full = f"{tool_name}:{sig}"
            if sig_full in approved:
                return Decision(allowed=True, stake=stake)
            # 需要人类确认
            return Decision(
                allowed=False,
                stake=stake,
                needs_human=True,
                reason=f"Operation '{tool_name}' is {stake.value} stake and requires human approval. "
                       f"To auto-approve, add '{sig_full}' to {_APPROVED_OPS_FILE}",
            )

        # MEDIUM: 放行但记录
        return Decision(allowed=True, stake=stake)

    def log(self, tool_name: str, args: Dict, result: Any, note: str = "") -> bool:
        """
        R2: 追加审计日志. append-only, 按天分文件.

        熔炉#109 Shield P0: 链式 hash 防篡改。
        每条记录包含前一条的 hash,形成不可篡改链条。
        篡改任意一条记录 → 后续所有 hash 断裂。

        每行一个 JSON:
        {"ts": "...", "tool": "...", "args": {...}, "result": "...", "stake": "...", "prev_hash": "...", "this_hash": "..."}
        """
        path = self.audit_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"

        # 读取前一条记录的 hash
        prev_hash = "0" * 16  # 创世记录
        try:
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            if lines and lines[0]:
                last = json.loads(lines[-1])
                prev_hash = last.get("this_hash", "0" * 16)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "args": _safe_serialize(args),
            "result": _safe_serialize(result),
            "stake": get_stake(tool_name).value,
            "prev_hash": prev_hash,
        }
        if note:
            entry["note"] = note

        # 计算本条记录的 hash(包含 prev_hash 形成链条)
        entry["this_hash"] = hashlib.sha256(
            json.dumps(entry, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True

    def approve(self, tool_name: str, args: Dict, note: str = "") -> bool:
        """
        将某操作加入白名单(永久批准).

        同时写入审计日志,记录谁批准了什么。
        """
        approved = _load_approved()
        sig = _call_signature(tool_name, args)
        key = f"{tool_name}:{sig}"
        approved.add(key)
        _APPROVED_OPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _APPROVED_OPS_FILE.write_text(
            json.dumps({"approved": sorted(approved)}, ensure_ascii=False, indent=2)
        )
        # 记录审批行为(Shield P1: approve 也要审计)
        self.log(tool_name, args, {"approved": key}, note=f"APPROVED {note}".strip())
        # 清缓存
        global _approved_cache
        _approved_cache = None
        return True

    def coverage_report(self) -> Dict:
        """
        P1-1: 覆盖声明(熔炉#109 D3 Aria+Shield).

        列出所有已知操作及其 stake 级别,显式声明未知操作的处理方式。
        这不是"漏检率百分比"--是"我知道覆盖了什么,承认不知道没覆盖什么"。
        """
        from memory_safety import _OPERATION_STAKE_MAP as _MS_MAP
        combined = {}
        for op, stake in _MS_MAP.items():
            combined[op] = stake.value
        for op, stake in _EXTRA_HIGH_RISK.items():
            combined[op] = stake.value
        return {
            "covered_operations": combined,
            "covered_count": len(combined),
            "unknown_operation_policy": "fail-safe HIGH (requires human approval)",
            "disclaimer": "Coverage is limited to known operation names. "
                          "Unknown operations default to HIGH stake. "
                          "Production leak rate is NOT quantified.",
        }

    def set_trust_anchor(self, anchor: str) -> None:
        """
        P1-2: 显式信任根声明（熔炉#109 D2 Code+刻菲斯）.
        
        声明治理链条的信任终止点。例如:
            gov.set_trust_anchor("pytest-9.0.2 + python-3.14 + macOS-25.1")
        
        这不是"信任根可证明安全"——是"我选择信任此根，可被审查/质疑/追溯"。
        在法律上等价于审计师的"重要性阈值"声明。
        """
        if not anchor or not anchor.strip():
            raise ValueError("Trust anchor must be a non-empty string")
        self._trust_anchor = anchor.strip()
        # 同步写入审计日志（_internal_meta 不是工具调用，是治理层自身操作）
        self.log("_internal_meta", {"action": "set_trust_anchor"}, {"anchor": anchor.strip()},
                 note=f"Trust anchor declared: {anchor.strip()}")

    def get_trust_anchor(self) -> Optional[str]:
        """返回当前声明的信任根."""
        return getattr(self, "_trust_anchor", None)


def _safe_serialize(obj: Any) -> str:
    """安全序列化任意对象为字符串(截断超长值)."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return s[:500] if len(s) > 500 else s
    except (TypeError, ValueError):
        return str(obj)[:500]


__all__ = ["Governance", "Decision", "StakeLevel", "get_stake"]
