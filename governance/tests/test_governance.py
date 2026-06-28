"""Hippo Governance Layer 测试."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 确保 import 路径
gov_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, gov_dir)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "embedding"))

import governance as gov_module
from governance import Governance, Decision, StakeLevel


@pytest.fixture
def gov(tmp_path):
    """每个测试用独立的 audit_dir 和 approved 文件."""
    audit_dir = tmp_path / "audit"
    approved_file = tmp_path / "approved.json"
    os.environ["HIPPO_GOVERNANCE_APPROVED"] = str(approved_file)
    os.environ["HIPPO_GOVERNANCE_AUDIT_DIR"] = str(audit_dir)
    # 重新 import 以拾取新环境变量
    import importlib
    importlib.reload(gov_module)
    return gov_module.Governance()


class TestStakeGate:
    """R1: stake gate 测试."""

    def test_low_stake_auto_approved(self, gov):
        d = gov.check("read", {"path": "/tmp/test.txt"})
        assert d.allowed is True
        assert d.stake == StakeLevel.LOW

    def test_medium_stake_allowed(self, gov):
        d = gov.check("write", {"path": "/tmp/test.txt", "content": "hi"})
        assert d.allowed is True
        assert d.stake == StakeLevel.MEDIUM

    def test_high_stake_blocked(self, gov):
        d = gov.check("delete", {"path": "/tmp/important.txt"})
        assert d.allowed is False
        assert d.needs_human is True
        assert d.stake == StakeLevel.HIGH
        assert "requires human approval" in d.reason

    def test_critical_stake_blocked(self, gov):
        d = gov.check("modify_config", {"key": "secret", "value": "xxx"})
        assert d.allowed is False
        assert d.needs_human is True
        assert d.stake == StakeLevel.CRITICAL

    def test_unknown_operation_failsafe_high(self, gov):
        """熔炉#109 D4: 未知操作默认 HIGH（fail-safe），不是 MEDIUM."""
        d = gov.check("some_new_tool", {})
        assert d.allowed is False  # HIGH stake 需要人类确认
        assert d.stake == StakeLevel.HIGH
        assert d.needs_human is True

    def test_approved_op_passes(self, gov, tmp_path):
        # 先批准
        gov.approve("delete", {"path": "/tmp/safe.txt"})
        # 再检查
        d = gov.check("delete", {"path": "/tmp/safe.txt"})
        assert d.allowed is True
        assert d.stake == StakeLevel.HIGH

    def test_different_args_still_blocked_after_approve(self, gov):
        gov.approve("delete", {"path": "/tmp/safe.txt"})
        d = gov.check("delete", {"path": "/etc/passwd"})
        assert d.allowed is False
        assert d.needs_human is True


class TestAuditLog:
    """R2: audit log 测试."""

    def test_log_written(self, gov, tmp_path):
        gov.log("read", {"path": "/tmp/x"}, {"status": "ok"})
        files = list(gov.audit_dir.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tool"] == "read"
        assert entry["stake"] == "low"
        assert "ts" in entry

    def test_multiple_logs_append(self, gov):
        gov.log("read", {"a": 1}, "r1")
        gov.log("write", {"b": 2}, "r2")
        gov.log("delete", {"c": 3}, "r3")
        files = list(gov.audit_dir.glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["tool"] == "read"
        assert json.loads(lines[1])["tool"] == "write"
        assert json.loads(lines[2])["tool"] == "delete"

    def test_long_result_truncated(self, gov):
        long_result = "x" * 1000
        gov.log("read", {}, long_result)
        files = list(gov.audit_dir.glob("*.jsonl"))
        entry = json.loads(files[0].read_text().strip())
        assert len(entry["result"]) <= 500

    def test_note_logged(self, gov):
        gov.log("write", {"k": "v"}, "ok", note="manual override")
        files = list(gov.audit_dir.glob("*.jsonl"))
        entry = json.loads(files[0].read_text().strip())
        assert entry["note"] == "manual override"

    def test_chain_hash_present(self, gov):
        """熔炉#109 Shield P0: 审计日志必须有链式 hash."""
        gov.log("read", {"a": 1}, "r1")
        gov.log("write", {"b": 2}, "r2")
        files = list(gov.audit_dir.glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        e1 = json.loads(lines[0])
        e2 = json.loads(lines[1])
        # 创世记录的 prev_hash = 全零
        assert e1["prev_hash"] == "0" * 16
        assert "this_hash" in e1
        # 第二条的 prev_hash = 第一条的 this_hash
        assert e2["prev_hash"] == e1["this_hash"]
        assert "this_hash" in e2

    def test_chain_hash_breaks_on_tamper(self, gov):
        """篡改任意一条记录 → 后续 hash 断裂."""
        gov.log("read", {}, "r1")
        gov.log("write", {}, "r2")
        gov.log("delete", {}, "r3")
        files = list(gov.audit_dir.glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        e1 = json.loads(lines[0])
        e2 = json.loads(lines[1])
        # 篡改第一条的 result
        e1["result"] = "TAMPERED"
        tampered_hash = hashlib.sha256(
            json.dumps({k: v for k, v in e1.items() if k != "this_hash"}, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        # 篡改后的 hash != 原始 hash → 链条断裂
        assert tampered_hash != e1["this_hash"]
        assert e2["prev_hash"] == e1["this_hash"]  # e2 仍指向旧 hash
        assert e2["prev_hash"] != tampered_hash  # 不匹配篡改后的 hash


class TestLoopGuard:
    """R3: loop guard 测试."""

    def test_no_false_positive_on_unique_calls(self, gov):
        for i in range(10):
            d = gov.check("read", {"path": f"/tmp/file_{i}.txt"})
            assert d.allowed is True

    def test_detects_identical_loop(self, gov):
        args = {"path": "/tmp/same.txt"}
        # 前 2 次正常
        assert gov.check("read", args).allowed is True
        assert gov.check("read", args).allowed is True
        # 第 3 次触发
        d = gov.check("read", args)
        assert d.allowed is False
        assert "Loop detected" in d.reason

    def test_different_args_no_loop(self, gov):
        for i in range(5):
            d = gov.check("read", {"path": f"/tmp/f{i}"})
            assert d.allowed is True

    def test_loop_clears_after_window(self, gov):
        """窗口滑过后应恢复正常."""
        args = {"x": 1}
        # 触发 loop
        gov.check("read", args)
        gov.check("read", args)
        gov.check("read", args)  # blocked
        # 填入不同的调用，把 loop 签名挤出窗口
        for i in range(5):
            gov.check("read", {"x": i + 100})
        # 现在应该不再触发
        d = gov.check("read", args)
        assert d.allowed is True


class TestApprove:
    """白名单功能测试."""

    def test_approve_persists(self, gov, tmp_path):
        gov.approve("send_external", {"to": "safe@example.com"})
        approved_file = Path(os.environ["HIPPO_GOVERNANCE_APPROVED"])
        data = json.loads(approved_file.read_text())
        assert len(data["approved"]) == 1

    def test_approve_exact_args_only(self, gov):
        gov.approve("delete", {"path": "/tmp/a"})
        # 完全匹配 → 通过
        assert gov.check("delete", {"path": "/tmp/a"}).allowed is True
        # 不同参数 → 拦截
        assert gov.check("delete", {"path": "/tmp/b"}).allowed is False


class TestCoverageAndTrustAnchor:
    """P1: 覆盖声明 + 信任根声明测试."""

    def test_coverage_report_structure(self, gov):
        report = gov.coverage_report()
        assert "covered_operations" in report
        assert "covered_count" in report
        assert "unknown_operation_policy" in report
        assert "disclaimer" in report
        assert isinstance(report["covered_operations"], dict)
        assert report["covered_count"] == len(report["covered_operations"])

    def test_coverage_includes_known_ops(self, gov):
        report = gov.coverage_report()
        ops = report["covered_operations"]
        assert ops["read"] == "low"
        assert ops["write"] == "medium"
        assert ops["delete"] == "high"
        assert ops["modify_config"] == "critical"
        assert ops["exec"] == "critical"

    def test_coverage_failsafe_policy(self, gov):
        report = gov.coverage_report()
        assert "fail-safe" in report["unknown_operation_policy"]
        assert "NOT" in report["disclaimer"]

    def test_trust_anchor_set_and_get(self, gov):
        gov.set_trust_anchor("pytest-9.0.2 + python-3.14 + macOS-25.1")
        assert gov.get_trust_anchor() == "pytest-9.0.2 + python-3.14 + macOS-25.1"

    def test_trust_anchor_logged_to_audit(self, gov):
        gov.set_trust_anchor("test-anchor")
        files = list(gov.audit_dir.glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        entry = json.loads(lines[-1])
        assert entry["tool"] == "_internal_meta"
        assert "Trust anchor" in entry.get("note", "")

    def test_trust_anchor_default_none(self, gov):
        assert gov.get_trust_anchor() is None

    def test_trust_anchor_rejects_empty(self, gov):
        with pytest.raises(ValueError):
            gov.set_trust_anchor("")
        with pytest.raises(ValueError):
            gov.set_trust_anchor("   ")
        with pytest.raises(ValueError):
            gov.set_trust_anchor(None)


class TestIntegration:
    """集成测试：模拟一个完整的 dumb loop."""

    def test_dumb_loop_with_governance(self, gov, tmp_path):
        """模拟 dumb loop 场景：正常执行 → 审计 → 结束."""
        results = []
        tools_to_call = [
            ("read", {"path": "/etc/config"}),
            ("write", {"path": "/tmp/out.txt", "content": "hello"}),
            ("search", {"query": "test"}),
        ]

        for tool, args in tools_to_call:
            d = gov.check(tool, args)
            assert d.allowed, f"{tool} should be allowed: {d.reason}"
            result = f"executed {tool}"
            results.append(result)
            gov.log(tool, args, result)

        # 验证审计日志
        files = list(gov.audit_dir.glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 3

    def test_dumb_loop_blocks_dangerous(self, gov):
        """模拟 dumb loop 场景：高危操作被拦截."""
        d = gov.check("rotate_key", {"service": "prod"})
        assert d.allowed is False
        assert d.stake == StakeLevel.CRITICAL
        assert "requires human approval" in d.reason

    def test_dumb_loop_catches_infinite_loop(self, gov):
        """模拟 dumb loop 场景：死循环被检测."""
        args = {"query": "same thing"}
        for _ in range(2):
            gov.check("search", args)
        d = gov.check("search", args)
        assert d.allowed is False
        assert "Loop detected" in d.reason
