"""Companion: legitimate terminal/execute_code usage must stay frictionless.

Preservation invariants — these pass on the vulnerable code TODAY and must
KEEP passing after the security fix. If the fix breaks any of them, the fix is
too broad (violates the "least disruptive" constraint):

  * ordinary safe commands (ls/pwd/git status/cat of non-sensitive files)
    in an untainted turn keep auto-approving — no new prompts;
  * a LOW-risk web result introduces no taint (untainted flows unaffected);
  * untainted headless execute_code keeps its documented #30882 behavior.

No test here executes anything: guard-level decisions only, canary-free.
"""

from __future__ import annotations

import pytest

from agent.tool_dispatch_helpers import make_tool_result_message
from tools import approval as approval_module


@pytest.fixture
def headless_manual(monkeypatch):
    for var in (
        "HERMES_INTERACTIVE",
        "HERMES_GATEWAY_SESSION",
        "HERMES_CRON_SESSION",
        "HERMES_EXEC_ASK",
        "HERMES_SESSION_PLATFORM",
        "HERMES_YOLO_MODE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda command: {"action": "allow", "findings": [], "summary": ""},
    )
    approval_module._session_approved.clear()
    approval_module._pending.clear()
    try:
        yield
    finally:
        approval_module._session_approved.clear()
        approval_module._pending.clear()


def test_untainted_safe_commands_still_auto_approve(headless_manual, tmp_path):
    """Ordinary read-only commands with no untrusted provenance: no friction."""
    ordinary = tmp_path / "notes.txt"
    ordinary.write_text("just notes\n", encoding="utf-8")
    for cmd in (
        "ls",
        "pwd",
        "git status",
        "pytest -q",
        "python scripts/foo.py",
        f"cat {ordinary}",
    ):
        guard = approval_module.check_all_command_guards(cmd, "local")
        assert guard["approved"] is True, (
            f"security fix regressed legitimate usage: {cmd!r} -> {guard}"
        )


def test_low_risk_web_result_taints_nothing(headless_manual):
    """A benign web result must not force authorization on later commands."""
    msg = make_tool_result_message(
        "web_search", "The capital of France is Paris. " * 5, "call_benign"
    )
    assert msg["_tool_output_risk"]["risk"] == "low"
    guard = approval_module.check_all_command_guards("ls /tmp", "local")
    assert guard["approved"] is True


def test_untainted_headless_execute_code_keeps_documented_behavior(headless_manual):
    """#30882 contract for the UNTAINTED case is preserved (fix scopes taint)."""
    guard = approval_module.check_execute_code_guard("print(2 + 2)", "local")
    assert guard["approved"] is True
