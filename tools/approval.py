"""Dangerous command approval -- the gate flow and per-session state.

Owns the session state (approvals, yolo, gateway queues, denial breaker), the three guard
entry points (``check_all_command_guards``, ``check_execute_code_guard``,
``request_tool_approval`` / ``_run_approval_gate``) and the shared human-decision engine
behind them. Leaves: ``approval_detection`` (hardline/dangerous patterns), ``approval_context``
(contextvars, config readers), ``approval_floors`` (pre-gate blocks, allowlist match),
``approval_prompt`` (CLI prompt, plugin transports, MCP elicitation), ``approval_gateway_wait``
(blocking gateway round-trip), ``approval_smart`` (guardian LLM), ``approval_human_wait``.
Leaves read facade-owned state (``_lock``, queues, denial breaker) back through ``tools.approval`` at
call time; sibling-defined names are imported from their defining module.
"""

from dataclasses import dataclass
import hashlib
import importlib
import logging
import os
import re
import threading
from typing import Optional

from utils import env_var_enabled, is_truthy_value
from tools import approval_context
from tools.approval_context import (
    _approval_turn_id, _get_session_platform, _is_cron_approval_context,
    _is_gateway_approval_context, _is_interactive_cli, _is_single_query_approval_context,
    _is_unattended_platform_approval_context, _resolve_cli_approval_callback, _should_fall_through_to_cli_approval,
    _tirith_fail_open, get_current_session_key,
)
from tools.approval_detection import (
    _CMDPOS, _CREDENTIAL_FILES, _HERMES_CONFIG_PATH, _HERMES_ENV_PATH,
    _PROJECT_SENSITIVE_WRITE_TARGET, _RE_FLAGS, _SHELL_RC_FILES, _SSH_SENSITIVE_PATH,
    _approval_key_aliases, _check_sudo_stdin_guard, _command_detection_variants,
    detect_dangerous_command, detect_hardline_command,
)
from tools.approval_floors import (
    _command_matches_permanent_allowlist, _hardline_block_result, _match_user_deny_rule, _sudo_stdin_block_result,
    _user_deny_block_result,
)
from tools.approval_gateway_wait import _await_gateway_decision
from tools.approval_prompt import _present_with_selected_transport, _transport_choice, prompt_dangerous_approval
from tools.approval_smart import _smart_verdict

logger = logging.getLogger(__name__)

# Frozen at import: reading os.environ per call would let any skill running in the process set
# this and bypass every approval check (prompt-injection escalation path).
_YOLO_MODE_FROZEN: bool = is_truthy_value(os.getenv("HERMES_YOLO_MODE", ""))


def _get_approval_mode() -> str:
    """Module-namespace alias for :func:`approval_context._get_approval_mode`.

    The guard entry points below resolve the mode through this alias (rather
    than ``approval_context.*`` directly) so tests and operators can steer
    ``tools.approval`` exactly like every other approval knob. Production
    behavior is identical: it delegates at call time.
    """
    return approval_context._get_approval_mode()


def _get_cron_approval_mode() -> str:
    """Module-namespace alias for :func:`approval_context._get_cron_approval_mode`.

    Same steering rationale as :func:`_get_approval_mode`: the cron guards
    below consult this alias, which delegates at call time.
    """
    return approval_context._get_cron_approval_mode()


# --- Per-session approval state (thread-safe) -----------------------------------------------------------------------

_lock = threading.Lock()
_pending: dict[str, dict] = {}
_session_approved: dict[str, set] = {}
_session_yolo: set[str] = set()
_permanent_approved: set = set()

# --- Consecutive-denial circuit breaker for smart approvals ---------------------------------------------------------
# Each retry of a smart-denied command burns another guardian LLM call. After ``approvals.denial_breaker_threshold``
# consecutive guardian DENY verdicts in one session (default 3; 0 disables) the deny message escalates to a hard-stop
# instruction; any approval resets the tally. Only TOOL RESULT text changes — no history surgery, no interrupts — so
# it is prompt-cache-invariant. Capped so short-lived session keys cannot grow it without bound; oldest (least
# recently denied) entries are evicted.
_denial_tally: dict[str, int] = {}
_DENIAL_TALLY_MAX_SESSIONS = 256


def _get_denial_breaker_threshold() -> int:
    """``approvals.denial_breaker_threshold``: default 3; 0 or negative disables."""
    try:
        return int(approval_context._get_approval_config().get("denial_breaker_threshold", 3))
    except (ValueError, TypeError):
        return 3


def _record_denial(session_key: str) -> int:
    """Increment and return the session's consecutive guardian-denial count. Pop-and-reinsert
    keeps actively-denying sessions at the most-recent end so eviction drops idle keys."""
    with _lock:
        count = _denial_tally.pop(session_key, 0) + 1
        _denial_tally[session_key] = count
        while len(_denial_tally) > _DENIAL_TALLY_MAX_SESSIONS:
            _denial_tally.pop(next(iter(_denial_tally)))
        return count


def _reset_denials(session_key: str) -> None:
    """Clear the session's consecutive-denial tally (an approval happened)."""
    with _lock:
        _denial_tally.pop(session_key, None)


def _denial_breaker_addendum(session_key: str) -> str:
    """Escalated hard-stop text once the breaker has tripped, else ''. Read-only: callers
    increment via :func:`_record_denial`; the text is appended verbatim to the deny message."""
    with _lock:
        count = _denial_tally.get(session_key, 0)
    threshold = _get_denial_breaker_threshold()
    if threshold <= 0 or count < threshold:
        return ""
    # WARNING (was DEBUG): a failed/blocked guardian call is a real event the operator needs to see — the
    # whole point of #82846 is that the hang was invisible. Log the elapsed time and error class too.
    logger.warning(
        "Smart-approval circuit breaker tripped for session %s: %d consecutive denials (threshold %d)",
        session_key, count, threshold,
    )
    return (
        f" CIRCUIT BREAKER: {count} consecutive commands were blocked by "
        "the security reviewer. STOP attempting variations of this "
        "operation. Report the blocked operation to the user and either ask them to run it manually or use /approve."
    )

# --- Sensitive reads (confidentiality companion to write gating) ------------------------------------------------------
# Write-side gating (redirection/tee/cp/sed -i into credential paths) stops
# persistence and destruction, but the terminal runs as the invoking UID so
# READS of the same stores (``cat ~/.hermes/.env``, ``cat ~/.ssh/id_rsa``,
# bare ``env``) auto-approved with zero authorization — the exact primitive
# a prompt-injected ``cat`` needs. This rule set is DISTINCT from
# provenance taint: it fires on the command shape alone, tainted or not,
# and routes into the normal approval gate (prompt with a human, deny
# without one) rather than blocking unconditionally.
_SSH_KEY_BASENAMES = r'(?:id_rsa|id_ed25519|id_ecdsa|id_dsa|authorized_keys)\b'
_KEY_FILE_EXTENSIONS = r'[\w\-]+\.(?:pem|key|p12|pfx|asc|gpg)\b'
_HERMES_AUTH_STORES = (
    r'(?:~/\.hermes/|'
    r'(?:\$home|\$\{home\})/\.hermes/|'
    r'(?:\$hermes_home|\$\{hermes_home\})/)'
    r'(?:auth\.json|webhook_subscriptions\.json|anthropic_oauth\.json|'
    r'auth/google_oauth\.json|mcp-tokens/)'
)
_CLOUD_CRED_DIRS = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:aws|kube|gnupg|docker|azure|config/gh|config/gcloud)/\S*'
)
# Credential *directories* themselves (not just specific files
# inside them). Without this, ``tar -cf - ~/.hermes`` or ``cp -r ~/.aws``
# slip through the sensitive-read detector because no specific filename
# appears in the command, even though the entire credential store is
# read/archived. The trailing ``(?:/\S*)?`` greedily consumes any descendant
# path (``~/.hermes/foo``, ``~/.hermes/subdir/file``) so anything under the
# credential directory is treated as sensitive; the outer
# ``_READ_TARGET_BOUNDARY`` handles whitespace/separator/EOS termination.
# Sibling names like ``~/.hermes2`` are rejected because the boundary
# check fails on the trailing non-boundary character.
_SENSITIVE_CRED_DIRS = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:hermes|ssh|aws|kube|gnupg|docker|azure)'
    r'(?:/\S*)?'
)
# Same directories under macOS/Linux absolute-path forms. Only
# matches the cloud-credential set — ``~/.ssh`` and ``~/.hermes`` already
# have other identifiers (key basenames / Hermes config patterns) that
# would catch the read; the cloud creds rely entirely on directory shape.
# Restricting to /(home|Users|root)/<name>/<cred>/ avoids making every
# absolute path sensitive. The trailing ``\S*`` greedily consumes the
# file/component portion (mirrors _CLOUD_CRED_DIRS for the home-prefix form).
_SENSITIVE_CRED_DIRS_ABS = (
    r'/(?:home|Users|root)(?:/[^/\s]+)?/\.'
    r'(?:aws|kube|gnupg|docker|azure)'
    r'(?:/\S*|(?=$))'
)
_SENSITIVE_READ_TARGET = (
    rf'(?:{_SSH_SENSITIVE_PATH}|'
    rf'{_HERMES_ENV_PATH}|'
    rf'{_HERMES_CONFIG_PATH}|'
    rf'{_HERMES_AUTH_STORES}|'
    rf'{_SHELL_RC_FILES}|'
    rf'{_CREDENTIAL_FILES}|'
    rf'{_CLOUD_CRED_DIRS}|'
    rf'{_PROJECT_SENSITIVE_WRITE_TARGET}|'
    rf'{_SSH_KEY_BASENAMES}|'
    rf'{_KEY_FILE_EXTENSIONS}|'
    rf'{_SENSITIVE_CRED_DIRS}|'
    rf'{_SENSITIVE_CRED_DIRS_ABS})'
)
# Verbs whose execution prints (parts of) the target file to stdout.
# ``dd if=~/.hermes/.env`` is a canonical exfil primitive;
# ``dd`` reads a file and writes it (to stdout, another file, or a pipe),
# so it must trigger the same sensitive-read handling as ``cat``/``head``.
_SENSITIVE_READ_VERBS = (
    r'(?:cat|tac|head|tail|less|more|most|bat|nl|strings|xxd|od|hexdump|'
    r'grep|egrep|fgrep|rg|ag|ack|sed|awk|gawk|mawk|jq|yq|base64|cut|sort|uniq|wc|'
    r'dd)'
)
_READ_TARGET_BOUNDARY = r'(?=[\s;&|"\'\n]|$)'
SENSITIVE_READ_PATTERNS = [
    (
        rf'{_CMDPOS}{_SENSITIVE_READ_VERBS}\b[^\n]*?(?:{_SENSITIVE_READ_TARGET}){_READ_TARGET_BOUNDARY}',
        "sensitive-read:credential-file",
        "read of sensitive credential/SSH/Hermes file",
    ),
    (
        rf'{_CMDPOS}(?:tar|zip|7z)\b[^\n]*?(?:{_SENSITIVE_READ_TARGET}){_READ_TARGET_BOUNDARY}',
        "sensitive-read:archive-include",
        "archive including sensitive credential/SSH/Hermes files",
    ),
    (
        rf'{_CMDPOS}(?:cp|scp|rsync)\b[^\n]*?(?:{_SENSITIVE_READ_TARGET}){_READ_TARGET_BOUNDARY}',
        "sensitive-read:copy-touch",
        "copy/transfer touching sensitive credential/SSH/Hermes files",
    ),
    (
        rf'{_CMDPOS}env\b((?:\s+[A-Za-z_]\w*=\S*)*)\s*(?:[|;&\n]|$)',
        "sensitive-read:env-enumeration",
        "environment enumeration (dumps secrets into context)",
    ),
    (
        rf'{_CMDPOS}printenv\s*(?:[|;&\n]|$)',
        "sensitive-read:env-enumeration",
        "environment enumeration (dumps secrets into context)",
    ),
    (
        rf'{_CMDPOS}set\s*(?:[|;&\n]|$)',
        "sensitive-read:env-enumeration",
        "environment enumeration (dumps secrets into context)",
    ),
]
SENSITIVE_READ_PATTERNS_COMPILED = [
    (re.compile(pattern, _RE_FLAGS), key, description)
    for pattern, key, description in SENSITIVE_READ_PATTERNS
]


def _detect_sensitive_read(command: str) -> tuple:
    """Check if a command reads sensitive credential stores or enumerates env.

    Returns (True, pattern_key, description) or (False, None, None).
    Detection-only: the caller decides approval vs. block. Uses the same
    detection variants (home-prefix folding, quote-aware command starts) as
    dangerous-command detection so ``$HOME/.hermes/.env`` and friends match.
    """
    for command_variant in _command_detection_variants(command):
        command_lower = command_variant.lower()
        for pattern_re, pattern_key, description in SENSITIVE_READ_PATTERNS_COMPILED:
            if pattern_re.search(command_lower):
                return (True, pattern_key, description)
    return (False, None, None)

# =========================================================================
# Per-turn provenance taint (defense-in-depth)
# =========================================================================
#
# When a tool result carries high-confidence prompt-injection findings,
# the turn is marked tainted. Terminal / execute_code guards consult this
# mark and route capability-bearing actions to human authorization instead
# of auto-approving them.
#
# Identity: keys are ``(session_key, turn_id)``. The session key comes from
# the same resolution as every other approval decision
# (``get_current_session_key()`` — gateway/TUI/CLI bind a distinct key per
# session before invoking the agent; see gateway/run.py, tui_gateway/
# server.py, cli.py). The turn id is the agent turn id
# (``agent._current_turn_id``, ``<session>:<task>:<uuid8>`` generated fresh
# per user turn in agent/turn_context.py), threaded explicitly by producers
# and via the ``_approval_turn_id`` ContextVar during tool dispatch.
# Unrelated sessions can never share a key; the empty-turn fallback
# (``""``) is used only by bare/test paths that have no turn, and taint
# there is still session-scoped, so it can only ever ADD approval friction
# inside that session — never silently authorize another session.
#
# The store is deny-biased by construction: its only power is forcing the
# human gate (with smart/LLM auto-approval suppressed). A leak, duplicate,
# or stale entry produces extra prompts, never silent execution.
_TAINT_KEY_PREFIX = "taint:"
_TAINT_KEY = "taint:untrusted-context"
_turn_taint: dict[tuple[str, str], dict] = {}
_taint_seq = 0
# Bound the store: keep only recent turns per session.
_TAINT_MAX_TURNS_PER_SESSION = 8
_TAINT_MAX_ENTRIES = 512

# Findings that alone are too weak to taint a turn (no instruction intent).
_TAINT_WEAK_FINDING_PREFIX = "invisible_unicode_"


def _findings_warrant_taint(findings) -> bool:
    """True when any finding shows instruction-level attacker intent."""
    for finding in findings or []:
        if not str(finding).startswith(_TAINT_WEAK_FINDING_PREFIX):
            return True
    return False


def _resolve_taint_scope(
    session_key: str | None = None,
    turn_id: str | None = None,
) -> tuple[str, str]:
    """Resolve the (session, turn) scope for taint reads/writes."""
    if session_key is None:
        session_key = get_current_session_key()
    if turn_id is None:
        try:
            turn_id = _approval_turn_id.get() or ""
        except Exception:
            turn_id = ""
    return (session_key or "", turn_id or "")


def _evict_old_turn_taint(session_key: str) -> None:
    """Bound per-session taint history (oldest turns first)."""
    if len(_turn_taint) <= _TAINT_MAX_ENTRIES:
        return
    session_keys = sorted(
        (key for key in _turn_taint if key[0] == session_key),
        key=lambda key: _turn_taint[key].get("seq", 0),
    )
    for key in session_keys[: -_TAINT_MAX_TURNS_PER_SESSION]:
        _turn_taint.pop(key, None)
    while len(_turn_taint) > _TAINT_MAX_ENTRIES:
        oldest = min(_turn_taint, key=lambda key: _turn_taint[key].get("seq", 0))
        _turn_taint.pop(oldest, None)


def note_turn_taint(
    session_key: str | None,
    turn_id: str | None,
    findings,
    *,
    source: str = "turn",
) -> bool:
    """Record provenance taint for a (session, turn). Returns True if stored.

    Weak-only findings (lone invisible-unicode) are ignored so benign
    content never taints a turn. Recording is idempotent: repeated notes
    merge findings without changing authorization semantics.
    """
    if not _findings_warrant_taint(findings):
        return False
    scope = _resolve_taint_scope(session_key, turn_id)
    global _taint_seq
    with _lock:
        entry = _turn_taint.get(scope)
        if entry is None:
            _taint_seq += 1
            _turn_taint[scope] = {
                "findings": sorted({str(finding) for finding in findings or []}),
                "source": source,
                "seq": _taint_seq,
            }
        else:
            merged = sorted(set(entry["findings"]) | {str(f) for f in findings or []})
            entry["findings"] = merged
        _evict_old_turn_taint(scope[0])
    return True


def current_turn_taint(
    session_key: str | None = None,
    turn_id: str | None = None,
) -> dict | None:
    """Return the taint record for a (session, turn), or None if clean."""
    scope = _resolve_taint_scope(session_key, turn_id)
    with _lock:
        entry = _turn_taint.get(scope)
        return dict(entry) if entry is not None else None


def clear_turn_taint(
    session_key: str | None = None,
    turn_id: str | None = None,
) -> None:
    """Remove taint for a (session, turn). Used by session teardown/tests."""
    scope = _resolve_taint_scope(session_key, turn_id)
    with _lock:
        _turn_taint.pop(scope, None)


def clear_session_taint(session_key: str) -> None:
    """Remove taint for every turn of a session (session end)."""
    with _lock:
        for key in [key for key in _turn_taint if key[0] == (session_key or "")]:
            _turn_taint.pop(key, None)


# Delegation inheritance: staged taint for not-yet-started
# child turns. A tainted parent stages its findings under the CHILD's
# session key at spawn time; the child claims them into its own initial
# turn at turn start (one-shot consume). Nothing is global, nothing
# persists beyond the first turn unless history legitimately reseeds it,
# and no model-controlled input can write, clear, or redirect this store.
_inherited_taint: dict[str, dict] = {}
_INHERITED_MAX_ENTRIES = 128


def stage_child_taint(child_session_key: str, findings) -> bool:
    """Stage parent taint for a child's initial turn. Returns True if stored.

    Called once at delegation/spawn time with the parent's active findings.
    Clean parents (no warranting findings) stage nothing. Model input never
    reaches this function — findings come from the ambient parent taint
    record, and the key is the child's own session id.
    """
    if not child_session_key or not _findings_warrant_taint(findings):
        return False
    global _taint_seq
    with _lock:
        _taint_seq += 1
        _inherited_taint[child_session_key] = {
            "findings": sorted({str(finding) for finding in findings or []}),
            "seq": _taint_seq,
        }
        while len(_inherited_taint) > _INHERITED_MAX_ENTRIES:
            oldest = min(
                _inherited_taint,
                key=lambda key: _inherited_taint[key].get("seq", 0),
            )
            _inherited_taint.pop(oldest, None)
    return True


def has_staged_inheritance(session_key: str) -> bool:
    """Peek (no consume) whether inheritance is staged for a session.

    Takes an explicit key and never touches ambient context, so the check
    itself is near-infallible. Empty keys are never staged: return False.
    """
    if not session_key:
        return False
    with _lock:
        return session_key in _inherited_taint


def claim_inherited_taint(
    session_key: str | None,
    turn_id: str | None,
) -> bool:
    """Consume staged inheritance into a (session, turn). One-shot.

    Called at turn start with the starting agent's explicit session id, so
    resolution never depends on ambient context that a child run may or may
    not have bound yet. Returns True when an inheritance was claimed.
    Later fresh turns find nothing staged and stay clean unless history
    reseeds them through the normal path.

    The staged entry is removed only after the taint is recorded: if
    recording raises, the entry survives so a fail-closed caller can still
    observe the pending inheritance instead of starting clean.
    """
    scope_session, scope_turn = _resolve_taint_scope(session_key, turn_id)
    with _lock:
        staged = _inherited_taint.get(scope_session)
    if staged is None:
        return False
    recorded = note_turn_taint(
        scope_session, scope_turn, staged.get("findings"), source="inherited"
    )
    if recorded:
        with _lock:
            _inherited_taint.pop(scope_session, None)
    return recorded


def inherit_parent_taint_to_child(child_session_key: str) -> bool:
    """Spawn-boundary policy: inherit active parent taint into a child.

    Reads the ambient parent (session, turn) taint and stages it for the
    child's initial turn. Returns False when the parent turn is clean
    (normal case — delegation proceeds unchanged).

    Fail-closed: when the parent is known tainted but staging
    fails for any reason, raises ``RuntimeError`` instead of letting the
    child start clean. The parent-taint *read* itself is also fail-closed:
    an unreadable parent record is unknown, not clean, and raises rather
    than spawning a silent clean child. Callers must let this propagate —
    it fails that one delegation tool call with an error, never a silent
    clean child. Model input never reaches this function.
    """
    try:
        parent_taint = current_turn_taint()
    except Exception as exc:
        raise RuntimeError(
            "refusing child spawn: parent taint unreadable "
            "(treating unknown as clean would be fail-open)"
        ) from exc
    if parent_taint is None:
        return False
    try:
        staged = stage_child_taint(
            child_session_key, parent_taint.get("findings")
        )
        if not staged:
            raise RuntimeError("staging declined for child session")
    except Exception as exc:
        raise RuntimeError(
            "refusing clean child spawn from tainted parent turn "
            "(provenance staging failed)"
        ) from exc
    logger.debug(
        "staged delegation taint for child session %s",
        child_session_key,
    )
    return True


def _resumable_tail_tool_messages(messages, cap: int) -> list:
    """Tool messages belonging to the turn being resumed.

    Taint is turn-scoped, so a fresh turn must never inherit taint merely
    because old tool output remains in history. The only messages owned by
    the resumed turn are trailing results the model has not responded to:

    * a trailing run of ``tool``-role messages (interrupted after tool
      execution, before the model saw the results), plus
    * tool results exposed by skipping a trailing *dangling*
      ``assistant(tool_calls)`` block (interrupted before execution;
      replay strips it, exposing the previous results as the tail).

    A tail of assistant text, a user message, or completed steps (every
    tool block followed by an assistant message) means prior turns
    completed — nothing is owned, nothing seeds. ``cap`` bounds the work.
    """
    if not messages or cap <= 0:
        return []
    idx = len(messages)
    while idx > 0:
        tail = messages[idx - 1]
        if (
            isinstance(tail, dict)
            and tail.get("role") == "assistant"
            and tail.get("tool_calls")
        ):
            # Dangling calls by construction: nothing follows them in the
            # history, so no call was answered; replay removes them.
            idx -= 1
            continue
        break
    owned = []
    while idx > 0 and len(owned) < cap:
        message = messages[idx - 1]
        if not isinstance(message, dict) or message.get("role") != "tool":
            break
        owned.append(message)
        idx -= 1
    owned.reverse()
    return owned


def _scan_tool_content_for_seed(content) -> list:
    """Classify persisted tool content with the existing scanner.

    Reuses the same ``context``-scope scan as live classification so only
    content that would qualify as high-risk today can seed taint. Returns
    warranting findings (possibly empty). Never raises: a scan failure
    yields no findings here — live ingestion (withholding on
    unprovenanced content) is the
    conservative path for content that cannot be proven clean, while a
    best-effort resume net must not break turn startup.
    """
    if isinstance(content, str):
        texts = [content]
    elif isinstance(content, list):
        texts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if not texts:
            return []
    else:
        return []
    try:
        from tools.threat_patterns import scan_for_threats
    except Exception:
        return []
    findings: list = []
    try:
        for text in texts:
            for finding in scan_for_threats(text, scope="context"):
                if finding not in findings:
                    findings.append(finding)
    except Exception:
        return []
    return [f for f in findings if _findings_warrant_taint([f])]


def seed_turn_taint_from_history(
    messages,
    *,
    session_key: str | None = None,
    turn_id: str | None = None,
    max_tool_messages: int = 50,
) -> bool:
    """Reconstruct taint from persisted or in-memory history.

    Resume/restart/compaction safety net: the session database persists
    message *content* but drops the transient ``_tool_output_risk`` key, so
    re-derivation must come from the content that actually survived.
    Messages carrying recorded high-risk metadata seed directly; messages
    without it are re-scanned with the existing scanner. Only content that
    qualifies as high-risk under current scanner semantics seeds taint;
    clean and low-risk history stays clean. Taint still expires with the
    turn. Deterministic from history bytes — no mutable persisted flag is
    trusted.

    Turn ownership: only the resumable tail — trailing tool
    results no assistant message has responded to (plus results exposed by
    skipping a trailing dangling ``assistant(tool_calls)`` block) — can
    seed. Completed turns (assistant/user tail) seed nothing, so a fresh
    turn after a malicious-but-answered turn stays clean. ``cap`` is
    ``max_tool_messages``.
    """
    if not messages:
        return False
    scope_session, scope_turn = _resolve_taint_scope(session_key, turn_id)
    owned = _resumable_tail_tool_messages(messages, max_tool_messages)
    seeded = False
    for message in owned:
        findings = None
        risk_metadata = message.get("_tool_output_risk")
        if isinstance(risk_metadata, dict) and risk_metadata.get("risk") == "high":
            findings = risk_metadata.get("findings")
        else:
            # Transient metadata lost (restart/compaction): re-derive from
            # the persisted content itself.
            findings = _scan_tool_content_for_seed(message.get("content"))
        if findings and note_turn_taint(
            scope_session, scope_turn, findings,
            source="history",
        ):
            seeded = True
    return seeded


def _explicit_taint_record(risk_metadata) -> dict | None:
    """Build a taint record from an explicitly passed risk dict, if warranting."""
    if not isinstance(risk_metadata, dict):
        return None
    findings = risk_metadata.get("findings") or []
    if risk_metadata.get("risk") == "low" and not findings:
        return None
    if not _findings_warrant_taint(findings):
        return None
    global _taint_seq
    with _lock:
        _taint_seq += 1
        seq = _taint_seq
    return {
        "findings": sorted({str(finding) for finding in findings}),
        "source": "explicit",
        "seq": seq,
    }


def _resolve_guard_taint(risk_metadata) -> dict | None:
    """Taint visible to an authorization guard: explicit kwarg wins, else ambient."""
    explicit = _explicit_taint_record(risk_metadata)
    if explicit is not None:
        return explicit
    return current_turn_taint()


def _is_taint_exempt_command(command: str) -> bool:
    """Pure-local metadata commands stay frictionless even on tainted turns.

    ``ls``/``dir``/``pwd`` without shell operators cannot read file content,
    exfiltrate, or mutate — gating them would only train users to approve
    blindly. Anything composed (pipes, separators, substitutions) is gated.
    """
    text = (command or "").strip()
    first = re.split(r"\s+", text, maxsplit=1)[0].lower() if text else ""
    if first not in ("ls", "dir", "pwd"):
        return False
    if re.search(r"[;&|$()`]", text):
        return False
    return True


def _taint_description(taint: dict) -> str:
    findings = ", ".join(taint.get("findings", [])[:4]) or "prompt injection"
    return (
        "untrusted high-risk context this turn "
        f"({findings}). The current context contains attacker-controlled "
        "content flagged as prompt injection, so this action needs explicit "
        "human authorization even though the command itself looks safe."
    )

# --- Gateway approval queue (the blocking wait loop lives in approval_gateway_wait) ---------------------------------


# Optional free-text reason supplied with an explicit deny (``/deny <reason>``) so the agent can adapt
# instead of only hearing "denied". Ported from qwibitai/nanoclaw#2832.
_gateway_queues: dict[str, list] = {}        # session_key → [_ApprovalEntry, …]
_gateway_notify_cbs: dict[str, object] = {}  # session_key → callable(approval_data)


def register_gateway_notify(session_key: str, cb) -> None:
    """Register ``cb(approval_data: dict) -> None`` for sending approval requests. The callback
    bridges sync→async: it runs in the agent thread and must schedule the send on the loop."""
    with _lock:
        _gateway_notify_cbs[session_key] = cb


def unregister_gateway_notify(session_key: str) -> None:
    """Unregister the callback and wake ALL blocked threads for this session so
    they don't hang forever (agent run finished or interrupted)."""
    with _lock:
        _gateway_notify_cbs.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        entry.event.set()


def resolve_gateway_approval(session_key: str, choice: str,
                             resolve_all: bool = False,
                             reason: Optional[str] = None,
                             request_id: Optional[str] = None) -> int:
    """Unblock waiting agent thread(s) from the gateway's /approve or /deny handler.

    *resolve_all* resolves every pending approval (``/approve all``); otherwise the oldest
    (FIFO) or the one matching *request_id*. *reason* is the ``/deny <reason>`` free text,
    relayed to the agent in the BLOCKED message. Returns the number resolved.
    """
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return 0
        if request_id:
            targets = [entry for entry in queue if entry.data.get("request_id") == request_id]
            if not targets:
                return 0
            queue[:] = [entry for entry in queue if entry not in targets]
        elif resolve_all:
            targets = list(queue)
            queue.clear()
        else:
            targets = [queue.pop(0)]
        if not queue:
            _gateway_queues.pop(session_key, None)

    for entry in targets:
        entry.result = choice
        if reason:
            entry.reason = reason
        entry.event.set()
    return len(targets)


def list_gateway_approvals(session_key: str) -> list[dict]:
    """Return replay-safe snapshots of unresolved approvals for one session."""
    with _lock:
        return [dict(entry.data) for entry in _gateway_queues.get(session_key, [])]


def ack_gateway_approval(session_key: str, request_id: str) -> bool:
    """Record that a client received a particular pending approval request."""
    with _lock:
        for entry in _gateway_queues.get(session_key, []):
            if entry.data.get("request_id") == request_id:
                entry.acknowledged = True
                return True
    return False


def has_blocking_approval(session_key: str) -> bool:
    """Check if a session has one or more blocking gateway approvals waiting."""
    with _lock:
        return bool(_gateway_queues.get(session_key))


def get_pending_gateway_approval(session_key: str) -> dict | None:
    """Copy of the oldest unresolved gateway approval, for reconnecting clients
    to restore a prompt. Read-only snapshot — the queue stays authoritative."""
    if not session_key:
        return None
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return None
        return dict(queue[0].data)


def submit_pending(session_key: str, approval: dict):
    """Store a pending approval request for a session."""
    with _lock:
        _pending[session_key] = approval


def approve_session(session_key: str, pattern_key: str):
    """Approve a pattern for this session only."""
    with _lock:
        _session_approved.setdefault(session_key, set()).add(pattern_key)


def _release_permission_mode_dependents(session_key: str) -> None:
    """Drop resources whose immutable mode derives from Hermes YOLO. Lazy import so approval-only
    sessions never load computer-use; releasing on BOTH edges makes enabling YOLO replace a
    standard backend and disabling it revoke a private unrestricted daemon immediately."""
    try:
        from tools.computer_use.tool import release_computer_use_session

        release_computer_use_session(session_key)
    except Exception:
        logger.debug("Failed to release permission-mode dependent resources for %s", session_key, exc_info=True)


def _set_session_yolo(session_key: str, enabled: bool) -> None:
    if not session_key:
        return
    with _lock:
        (_session_yolo.add if enabled else _session_yolo.discard)(session_key)
    _release_permission_mode_dependents(session_key)


def enable_session_yolo(session_key: str) -> None:
    """Enable YOLO bypass for a single session key."""
    _set_session_yolo(session_key, True)


def disable_session_yolo(session_key: str) -> None:
    """Disable YOLO bypass for a single session key."""
    _set_session_yolo(session_key, False)


def clear_session(session_key: str) -> None:
    """Remove all approval and yolo state for a given session."""
    if not session_key:
        return
    with _lock:
        _session_approved.pop(session_key, None)
        _session_yolo.discard(session_key)
        _pending.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        # Cancel blocked waits now so the old run unwinds instead of idling until timeout.
        entry.result = "deny"
        entry.event.set()
    _release_permission_mode_dependents(session_key)
    # Session-persistent code kernels (local and remote) share this owner key and die at the same boundary so a
    # finished conversation cannot leak a live interpreter.
    for module, shutdown in (("tools.code_kernel", "shutdown_kernels_for_owner"),
                             ("tools.code_kernel_remote", "shutdown_remote_kernels_for_owner")):
        try:
            getattr(importlib.import_module(module), shutdown)(session_key)
        except Exception:
            pass


def is_session_yolo_enabled(session_key: str) -> bool:
    """Return True when YOLO bypass is enabled for a specific session."""
    if not session_key:
        return False
    with _lock:
        return session_key in _session_yolo


def is_current_session_yolo_enabled() -> bool:
    """Return True when the active approval session has YOLO bypass enabled."""
    return is_session_yolo_enabled(get_current_session_key(default=""))


def _yolo_active() -> bool:
    """CLI ``--yolo`` (process-scoped, frozen at import) or gateway ``/yolo``
    (session-scoped). Hardline / deny-rule floors run BEFORE this everywhere."""
    return _YOLO_MODE_FROZEN or is_current_session_yolo_enabled()


def is_approved(session_key: str, pattern_key: str) -> bool:
    """Session-scoped or permanent approval. Accepts the canonical key and the legacy
    regex-derived key so existing command_allowlist entries survive key migrations."""
    aliases = _approval_key_aliases(pattern_key)
    with _lock:
        approved = _permanent_approved | _session_approved.get(session_key, set())
    return any(alias in approved for alias in aliases)


def approve_permanent(pattern_key: str):
    """Add a pattern to the permanent allowlist."""
    with _lock:
        _permanent_approved.add(pattern_key)


def load_permanent(patterns: set):
    """Bulk-load permanent allowlist entries from config."""
    with _lock:
        _permanent_approved.update(patterns)


def _persist_choice(session_key: str, choice: str, warnings: list[tuple]) -> None:
    """Persist a human ``session``/``always`` choice for each ``(key, _, is_tirith)``. Tirith
    findings are session-max by design (no broad permanent allowlisting of content-level
    findings), so ``always`` downgrades them to session. ``once`` persists nothing.

    ``taint:*`` keys are one-shot by design and never persist: a tainted
    turn's approval must not survive to the next command, so a later tainted
    command on the same session still requires a fresh human decision.
    Ordinary keys retain their existing persistence semantics.
    """
    for key, _, is_tirith in warnings:
        if key.startswith(_TAINT_KEY_PREFIX):
            continue
        if choice not in ("session", "always"):
            continue
        approve_session(session_key, key)
        if choice == "always" and not is_tirith:
            approve_permanent(key)
            save_permanent_allowlist(_permanent_approved)


# --- Config persistence for permanent allowlist ---------------------------------------------------------------------

def load_permanent_allowlist() -> set:
    """Load ``command_allowlist`` from config and sync it into the approval state
    so is_approved() honors 'always' choices from previous sessions."""
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        patterns = set(config.get("command_allowlist", []) or [])
        if patterns:
            load_permanent(patterns)
        return patterns
    except Exception as e:
        logger.warning("Failed to load permanent allowlist: %s", e)
        return set()


def save_permanent_allowlist(patterns: set):
    """Save permanently allowed command patterns to config."""
    try:
        from hermes_cli.config import load_config, save_config
        config = load_config()
        config["command_allowlist"] = list(patterns)
        save_config(config)
    except Exception as e:
        logger.warning("Could not save allowlist: %s", e)


# --- Bypass check (yolo / mode=off) ---------------------------------------------------------------------------------

def is_approval_bypass_active_for_session(session_key: str) -> bool:
    """Canonical three-source bypass check: process ``--yolo`` (frozen at import), the
    session-scoped gateway ``/yolo`` toggle, ``approvals.mode: off``. Pure bypass
    sub-expression only — hardline blocklist / permanent allowlist are the caller's job."""
    return (_YOLO_MODE_FROZEN or is_session_yolo_enabled(session_key) or approval_context._get_approval_mode() == "off")


def is_approval_bypass_active() -> bool:
    """Return whether the current approval context has bypass enabled."""
    return is_approval_bypass_active_for_session(get_current_session_key(default=""))


# --- Result builders shared by the gates ----------------------------------------------------------------------------

def _approved() -> dict:
    return {"approved": True, "message": None}


def _denied(message: str, *, pattern_key: str, description: str, outcome: str, **extra) -> dict:
    """Standard non-consent result: the agent must not retry or rephrase."""
    return {"approved": False, "message": message, "pattern_key": pattern_key,
            "description": description, "outcome": outcome, "user_consent": False, **extra}


def _blocked(message: str, *, pattern_key: str, description: str) -> dict:
    """Non-interactive block (cron / -q / unattended / no-human): no consent keys."""
    return {"approved": False, "message": message, "pattern_key": pattern_key, "description": description}


def _user_approved(session_key: str, description: str) -> dict:
    """A human approval (incl. ESCALATE-then-approve or a smart-DENY owner
    override) resets the consecutive-denial tally."""
    _reset_denials(session_key)
    return {"approved": True, "message": None, "user_approved": True, "description": description}


def _gateway_notify_cb(session_key: str):
    with _lock:
        return _gateway_notify_cbs.get(session_key)


def _pending_result(spec, session_key: str, *, command: str, description: str,
                    pattern_key: str, pattern_keys: list[str], body: str | None,
                    smart_denied: bool) -> dict:
    """Queue an approval nobody can answer right now (no gateway notifier, no CLI panel) for
    ``/approve`` / ``/deny`` review. Command/code gates return the backward-compatible
    ``pending_approval`` shape (``pattern_keys`` + STOP text); the action gate ``approval_required``."""
    pending = {"command": command, "pattern_key": pattern_key}
    if spec.pending_keys:
        pending["pattern_keys"] = pattern_keys
    pending["description"] = description
    if smart_denied:
        pending.update(smart_denied=True, allow_permanent=False)
    submit_pending(session_key, pending)
    if not spec.pending_keys:
        return {
            "approved": False, "pattern_key": pattern_key, "status": "approval_required",
            "command": command, "description": description,
            "message": (f"⚠️ This action is potentially dangerous ({description}). "
                        f"Asking the user for approval.\n\n**Target:**\n```\n{command}\n```"),
        }
    body = body or f"**Command:**\n```\n{command}\n```"
    result = {
        "approved": False, "pattern_key": pattern_key, "status": "pending_approval",
        "approval_pending": True, "command": command, "description": description,
        "message": (
            f"⚠️ {description}. Asking the user for approval.\n\n{body}\n\n"
            f"STOP: do NOT re-run, rephrase, or re-issue this {spec.noun} — each "
            "variant sends the user ANOTHER approval card. Wait for the "
            "user's decision; if this turn must end, report that approval is pending."
        ),
    }
    if smart_denied:
        result.update(smart_denied=True, allow_permanent=False)
    return result


# --- Unattended contexts (nobody present to answer a prompt) --------------------------------------------------------

@dataclass(frozen=True)
class _Unattended:
    """One non-interactive context and the text every gate uses to explain it."""
    name: str       # "single_query" | "cron" | "unattended"
    cfg_key: str    # approvals.<cfg_key>: approve|deny
    clause: str     # "why nobody can approve" (lower-case sentence fragment)
    scope: str      # "in cron jobs" — completes "To allow ... {scope}"
    trust: str      # execute_code: "approve only if {trust}"

    def mode(self) -> str:
        # Looked up on the defining module at call time so tests patching the getters keep working.
        return getattr(approval_context, f"_get_{self.name}_approval_mode")()

    def block_message(self, subject: str, *, noun: str, advice: str) -> str:
        return (f"BLOCKED: {subject} but {self.clause}. {advice} To allow {noun} {self.scope}, set "
                f"approvals.{self.cfg_key}: approve in config.yaml.")

    @property
    def exec_tail(self) -> str:
        return (f"{self.clause[0].upper()}{self.clause[1:]}. Use normal tools "
                f"instead, or set approvals.{self.cfg_key}: approve only if {self.trust}.")


_SINGLE_QUERY_CTX = _Unattended(
    "single_query", "single_query_mode",
    "single-query mode (-q) runs without a user present to approve it",
    "in single-query mode", "this single-query run is intentionally trusted",
)
_CRON_CTX = _Unattended(
    "cron", "cron_mode", "cron jobs run without a user present to approve it",
    "in cron jobs", "this cron profile is intentionally trusted",
)


def _unattended_contexts() -> list[_Unattended]:
    """Active unattended contexts in evaluation order: single-query first (``hermes chat -q``
    exports HERMES_INTERACTIVE=1 but nobody answers); cron beats a platform marker because
    cron binds the platform for delivery routing only."""
    contexts = []
    if _is_single_query_approval_context():
        contexts.append(_SINGLE_QUERY_CTX)
    if _is_cron_approval_context():
        contexts.append(_CRON_CTX)
    elif _is_unattended_platform_approval_context():
        contexts.append(_Unattended(
            "unattended", "unattended_mode",
            "this session runs on an unattended platform "
            f"({_get_session_platform()}) with no user present to approve it",
            "on unattended platforms", "sessions on this surface are intentionally trusted",
        ))
    return contexts


def _unattended_deny(command: str, ctx: _Unattended) -> dict | None:
    """Deny-mode handling for one unattended context (cron / -q / webhook); None = allow.

    Pattern detection first, then tirith so content-level threats (homograph URLs,
    pipe-to-interpreter, terminal injection) are caught even when the pattern detector misses.
    An un-importable tirith honours ``security.tirith_fail_open``: fail-closed means block,
    since nobody can approve.
    """
    if ctx.mode() != "deny":
        return None

    def block(subject: str) -> dict:
        return {"approved": False, "message": ctx.block_message(
            subject, noun="dangerous commands",
            advice="Find an alternative approach that avoids this command.")}

    is_dangerous, _pk, description = detect_dangerous_command(command)
    if is_dangerous:
        result = block(f"Command flagged as dangerous ({description})")
        if ctx.name == "single_query":
            result.update(pattern_key=_pk, description=description)
        return result
    try:
        from tools.tirith_security import check_command_security
        tirith = check_command_security(command)
    except ImportError:
        if _tirith_fail_open():
            return None
        return {"approved": False, "message": (
            "BLOCKED: the Tirith security scanner could not be imported and security.tirith_fail_open is false, "
            f"so this command cannot be silently allowed — and {ctx.clause}. "
            f"Find an alternative approach, install tirith, or set approvals.{ctx.cfg_key}: approve in config.yaml.")}
    if tirith.get("action") in ("block", "warn"):
        return block(_format_tirith_description(tirith))
    return None


# --- Human-decision engine shared by the three gates ----------------------------------------------------------------
# Every flagged action reaches a human the same way — selected plugin transport → gateway round-trip → pending
# fallback → CLI prompt → persist — so the consent contract (silence is not consent, deny is a hard halt, a smart-DENY
# override is one operation) cannot drift between gates. Only wording and a few policy knobs differ per flavor; they
# live in _GateSpec.

@dataclass(frozen=True)
class _GateSpec:
    noun: str                 # "command" | "code" — for the pending STOP text
    transport: bool           # offer the selected plugin transport first
    user_approved: bool       # human approval resets the denial tally
    redact_cli: bool          # CLI prompt + hooks see the redacted copy
    pending_keys: bool        # pending fallback: redacted ``pending_approval`` shape with
                              # pattern_keys (True) vs raw ``approval_required`` (False)
    # Message templates. ``{breaker}`` = the denial circuit-breaker addendum,
    # read only where a template shows it (reading it logs when tripped).
    notify_failed: str
    gateway_refused: str      # {reason}{reason_addendum}{timeout_addendum}{breaker}
    transport_denied: str     # {breaker}
    cli_timeout: str          # {breaker}
    cli_denied: str           # {description}{breaker}
    smart_log: str            # {command}{description}{session_key}


_STOP_COMMAND = (
    " The user has NOT consented to this action. Do NOT retry this command, do "
    "NOT rephrase it, and do NOT attempt the same outcome via a different "
    "command. Stop the current workflow and wait for the user to respond before "
    "taking any further destructive or irreversible action."
)
_STOP_ACTION = (
    " The user has NOT consented to this action. Do NOT retry it, do NOT "
    "rephrase it, and do NOT attempt the same outcome via a different path."
)

_COMMAND_GATE = _GateSpec(
    noun="command", transport=True, user_approved=True, redact_cli=False, pending_keys=True,
    notify_failed="BLOCKED: Failed to send approval request to user. Do NOT retry.",
    gateway_refused="BLOCKED: Command {reason}.{reason_addendum}" + _STOP_COMMAND
                    + "{timeout_addendum}{breaker}",
    transport_denied=(
        "BLOCKED: User denied this command through the selected approval "
        "transport. The user has NOT consented to this action. Do NOT retry or "
        "attempt the same outcome through another route.{breaker}"
    ),
    cli_timeout="BLOCKED: Command timed out without user response." + _STOP_COMMAND
                + " Silence is not consent.{breaker}",
    cli_denied="BLOCKED: User denied this command." + _STOP_COMMAND + "{breaker}",
    smart_log="Smart approval: auto-approved '{command}' ({description})",
)
_EXECUTE_CODE_GATE = _GateSpec(
    noun="code", transport=True, user_approved=True, redact_cli=True, pending_keys=True,
    notify_failed="BLOCKED: Failed to send execute_code approval request to user. Do NOT retry.",
    gateway_refused=(
        "BLOCKED: execute_code script {reason}.{reason_addendum} The user has "
        "NOT consented to running this code. Do NOT retry, do NOT rephrase the "
        "script, and do NOT attempt the same outcome via a different tool.{timeout_addendum}{breaker}"
    ),
    transport_denied=(
        "BLOCKED: User denied execute_code through the selected approval transport. The user has NOT consented."
    ),
    cli_timeout="BLOCKED: Action timed out without user response." + _STOP_ACTION
                + " Silence is not consent.{breaker}",
    cli_denied=(
        "BLOCKED: User denied execute_code script execution (matched "
        "'{description}'). Do NOT retry — the user has explicitly rejected it.{breaker}"
    ),
    smart_log="Smart approval: auto-approved execute_code for session {session_key}",
)
# Plugin-escalated tool calls / protected writes: no transport, no breaker,
# no user_approved marker (parity with the historical gate).
_ACTION_GATE = _GateSpec(
    noun="action", transport=False, user_approved=False, redact_cli=False, pending_keys=False,
    notify_failed="BLOCKED: Failed to send approval request to user. Do NOT retry.",
    gateway_refused="BLOCKED: Action {reason}.{reason_addendum}" + _STOP_ACTION
                    + "{timeout_addendum}",
    transport_denied="",
    cli_timeout="BLOCKED: Action timed out without user response." + _STOP_ACTION
                + " Silence is not consent.",
    cli_denied=(
        "BLOCKED: User denied this potentially dangerous action (matched "
        "'{description}'). Do NOT retry — the user has explicitly rejected it."
    ),
    smart_log="",
)


def _smart_gate(spec: _GateSpec, command: str, description: str, pattern_key: str,
                pattern_keys: list[str], session_key: str, *,
                human_present: bool) -> tuple[dict | None, bool]:
    """Guardian-LLM step -> ``(result, smart_denied_for_owner)``: a result ends the gate;
    ``smart_denied_for_owner`` means an interactive owner may still override the DENY for this
    one operation (once/deny only, nothing persists).

    APPROVE approves this command only — pattern-level persistence would let one benign
    command suppress review of later commands in the same broad detector category. A DENY
    counts toward the denial breaker even when an owner may override it. ESCALATE follows the
    normal, potentially persistent manual behavior.
    """
    verdict = _smart_verdict(command, description, pattern_key, pattern_keys, session_key)
    if verdict == "approve":
        _reset_denials(session_key)
        logger.debug(spec.smart_log.format(command=command[:60], description=description, session_key=session_key))
        return {"approved": True, "message": None, "smart_approved": True, "description": description}, False
    if verdict != "deny":
        return None, False
    _record_denial(session_key)
    if human_present:
        return None, True
    return {
        # Unattended programmatic platforms (webhook/msgraph_webhook/ api_server): respect unattended_mode
        # config. Resolves instantly — never a pending approval nobody can answer (#37284, #87509).
        "approved": False,
        "message": (f"BLOCKED by smart approval: {description}. The command was assessed as genuinely "
                    f"dangerous. Do NOT retry.{_denial_breaker_addendum(session_key)}"),
        "smart_denied": True,
    }, True


def _human_decision(spec: _GateSpec, *, command: str, description: str,
                    pattern_key: str, pattern_keys: list[str], warnings: list[tuple],
                    session_key: str, approval_callback, is_cli: bool, is_gateway: bool,
                    is_ask: bool, smart: bool = False,
                    permanent_capable: bool = True, pending_body=None) -> dict:
    """Ask a human (after the optional guardian-LLM step) and turn the answer into the gate result.

    ``warnings`` are the ``(key, _, is_tirith)`` tuples :func:`_persist_choice` stores on
    session/always. ``permanent_capable`` hides [a]lways when no key could be permanently
    allowlisted (pure-tirith prompts); a smart-DENY owner override reduces every surface to
    once/deny and persists nothing. ``pending_body`` is a thunk, built only once a human is
    actually asked, so a smart APPROVE never pays for redacting a large script.
    """
    from agent.redact import redact_sensitive_text

    smart_denied = False
    if smart:
        result, smart_denied = _smart_gate(spec, command, description, pattern_key, pattern_keys,
                                           session_key, human_present=is_cli or is_gateway or is_ask)
        if result is not None:
            return result
    pending_body = pending_body() if pending_body else None
    allow_permanent = permanent_capable and not smart_denied

    def deny(template: str, outcome: str, **fmt) -> dict:
        breaker = ""
        if "{breaker}" in template:
            breaker = _denial_breaker_addendum(session_key)
        deny_reason = fmt.pop("deny_reason", None)
        extra = {"deny_reason": deny_reason} if "reason" in fmt else {}
        return _denied(template.format(description=description, breaker=breaker, **fmt),
                       pattern_key=pattern_key, description=description,
                       outcome=outcome, **extra)

    def grant(choice: str) -> dict:
        # A smart-DENY owner override is always one operation, even if an older client returns "session" or "always".
        if not smart_denied:
            _persist_choice(session_key, choice, warnings)
        if spec.user_approved:
            return _user_approved(session_key, description)
        return _approved()

    if spec.transport:
        attempt = _present_with_selected_transport(
            command=command, description=description, pattern_key=pattern_key, pattern_keys=pattern_keys,
            session_key=session_key, surface="gateway" if (is_gateway or is_ask) else "cli",
            allow_session=not smart_denied, allow_permanent=allow_permanent,
        )
        choice, denied = _transport_choice(attempt, pattern_key=pattern_key, description=description)
        if denied is not None:
            return denied
        if choice is not None:
            if choice == "deny":
                _record_denial(session_key)
                return deny(spec.transport_denied, "denied")
            return grant(choice)

    # Gateway/async approval: block the agent thread until /approve or /deny, mirroring the CLI's synchronous input()
    # flow. The agent never sees "approval_required" here — it gets output or a definitive BLOCKED.
    if is_gateway or is_ask:
        # Redacted copies for user-visible rendering only (the gateway paints them into Discord/Slack); the raw
        # command still executes after approval and persistence keys off pattern_key.
        display_command = redact_sensitive_text(command)
        display_description = redact_sensitive_text(description)
        notify_cb = _gateway_notify_cb(session_key)
        if notify_cb is not None:
            # Smart DENY overrides are one-operation decisions, so the UI must not offer a
            # permanent scope. Session approval is safe for every non-Smart-DENY prompt —
            # including pure-tirith ones, where persistence already caps scope at session.
            data = {
                "command": display_command, "pattern_key": pattern_key,
                "pattern_keys": pattern_keys, "description": display_description,
                "allow_permanent": permanent_capable and not smart_denied,
                "allow_session": not smart_denied,
            }
            if smart_denied:
                data["smart_denied"] = True
            decision = _await_gateway_decision(session_key, notify_cb, data, surface="gateway")
            if decision.get("notify_failed"):
                return _denied(spec.notify_failed, pattern_key=pattern_key,
                               description=description, outcome="notify_failed")
            # Consent contract: silence is NOT consent, and an explicit deny is a hard
            # halt — both produce a BLOCKED outcome. ``/deny <reason>`` free text is
            # relayed verbatim so the agent can adapt rather than only hearing "denied".
            choice, deny_reason = decision["choice"], decision.get("reason")
            if not decision["resolved"]:
                return deny(spec.gateway_refused, "timeout", reason="timed out without user response",
                            reason_addendum="", timeout_addendum=" Silence is not consent.",
                            deny_reason=deny_reason)
            if choice is None or choice == "deny":
                return deny(spec.gateway_refused, "denied", reason="denied by user",
                            reason_addendum=(f' Reason given by the user: "{deny_reason}".' if deny_reason else ""),
                            timeout_addendum="", deny_reason=deny_reason)
            return grant(choice)

        # No gateway callback (cron, batch, or ask-mode leaked into an interactive CLI, historically via `import
        # gateway.run`): paint the local panel when possible instead of a pending_approval that makes the agent look
        # "auto-blocked".
        if not _should_fall_through_to_cli_approval(
            is_cli=is_cli, approval_callback=approval_callback, notify_cb=notify_cb,
        ):
            if not spec.pending_keys:
                display_command, display_description = command, description
            return _pending_result(
                spec, session_key, command=display_command, description=display_description, pattern_key=pattern_key,
                pattern_keys=pattern_keys, body=pending_body, smart_denied=smart_denied,
            )

    # CLI interactive: single combined prompt, wrapped in the pre/post plugin hooks.
    prompt_command, prompt_description = command, description
    if spec.redact_cli:
        prompt_command = redact_sensitive_text(command)
        prompt_description = redact_sensitive_text(description)
    hook_kwargs = dict(command=prompt_command, description=prompt_description, pattern_key=pattern_key,
                       pattern_keys=list(pattern_keys), session_key=session_key, surface="cli")
    approval_context._fire_approval_hook("pre_approval_request", **hook_kwargs)
    choice = prompt_dangerous_approval(prompt_command, prompt_description, allow_permanent=allow_permanent,
                                       smart_denied=smart_denied, approval_callback=approval_callback)
    approval_context._fire_approval_hook("post_approval_response", **hook_kwargs, choice=choice)
    if choice == "timeout":
        return deny(spec.cli_timeout, "timeout")
    if choice == "deny":
        # No _record_denial(): the breaker counts consecutive guardian LLM
        # DENY verdicts, not deliberate human denials.
        return deny(spec.cli_denied, "denied")
    return grant(choice)


def _presence(approval_callback=None) -> tuple:
    """``(approval_callback, is_cli, is_gateway, is_ask)`` for the current context. Single-query
    (-q) exports HERMES_INTERACTIVE=1 but nobody answers prompts, and HERMES_EXEC_ASK has no
    human either — both are cleared so single_query_mode actually takes effect."""
    approval_callback = _resolve_cli_approval_callback(approval_callback)
    is_cli, is_gateway = _is_interactive_cli(), _is_gateway_approval_context()
    is_ask = env_var_enabled("HERMES_EXEC_ASK")
    if _is_single_query_approval_context():
        is_cli = is_gateway = is_ask = False
    return approval_callback, is_cli, is_gateway, is_ask


def _run_approval_gate(
    *, pattern_key: str, description: str, display_target: str, approval_callback=None,
    subject: str = "", noun: str = "flagged actions",
    advice: str = "Find an alternative approach that avoids this action.",
    cron_deny_message: str = "", single_query_deny_message: str = "", unattended_deny_message: str = "",
    autoapprove_log_prefix: str, fail_closed_when_no_human: bool = False, no_human_block_message: str = "",
) -> dict:
    """Shared human-approval gate for a flagged action (tool call or write): decision core for
    :func:`request_tool_approval` and the file-tool write gates.

    Order: yolo bypass → session-cache short-circuit → interactive/gateway/unattended branch →
    prompt → persistence. Input-shape checks (hardline, allowlist, pattern detection) are the
    caller's job. ``fail_closed_when_no_human``: a non-interactive, non-gateway, non-cron
    context BLOCKS instead of auto-approving, so a plugin-flagged action never runs ungated.
    Unattended deny text is ``ctx.block_message(subject, noun, advice)`` unless the caller passes
    an explicit ``*_deny_message`` (the file-tool write gates word their own).
    """
    # Hardline blocks are the caller's job BEFORE this gate, so yolo here only skips the recoverable approval layer.
    if _yolo_active():
        return _approved()
    session_key = get_current_session_key()
    if is_approved(session_key, pattern_key):
        return _approved()

    approval_callback, is_cli, is_gateway, is_ask = _presence(approval_callback)
    if not is_cli and not is_gateway:
        log_args = (autoapprove_log_prefix, pattern_key, description)
        # Every unattended context resolves instantly — never a pending approval nobody can answer.
        deny_messages = {
            "single_query": single_query_deny_message, "cron": cron_deny_message,
            "unattended": unattended_deny_message,
        }
        for ctx in _unattended_contexts():
            if ctx.mode() == "deny":
                message = deny_messages[ctx.name]
                if not message and ctx.name == "unattended":
                    # Platform contexts keep the generic wording (historical shape).
                    message = ctx.block_message(f"approval required ({description})", noun="flagged actions",
                                                advice="Find an alternative approach that avoids this action.")
                elif not message:
                    message = ctx.block_message(subject, noun=noun, advice=advice)
                return _blocked(message, pattern_key=pattern_key, description=description)
            if ctx.name == "single_query":
                # Return here rather than fall through: the fail-closed branch would
                # otherwise block what single_query_mode: approve just authorized.
                logger.warning("%s (pattern: %s): %s — single-query auto-approve "
                               "(approvals.single_query_mode: approve).", *log_args)
                return _approved()
            break  # cron/unattended approve-mode: auto-approve below
        else:
            if fail_closed_when_no_human:
                logger.warning("%s (pattern: %s): %s — no interactive user/gateway present; "
                               "BLOCKED (fail-closed). Set HERMES_INTERACTIVE or "
                               "HERMES_GATEWAY_SESSION to answer the prompt.", *log_args)
                return _blocked(no_human_block_message or (
                    f"BLOCKED: approval required ({description}) but no "
                    "interactive user or gateway is present to approve it."),
                    pattern_key=pattern_key, description=description)
        logger.warning("%s (pattern: %s): %s — set HERMES_INTERACTIVE or "
                       "HERMES_GATEWAY_SESSION to require approval.", *log_args)
        return _approved()

    return _human_decision(
        _ACTION_GATE, command=display_target, description=description, pattern_key=pattern_key,
        pattern_keys=[pattern_key], warnings=[(pattern_key, None, False)], session_key=session_key,
        approval_callback=approval_callback, is_cli=is_cli, is_gateway=is_gateway, is_ask=is_ask,
    )


def _should_skip_container_guards(env_type: str, has_host_access: bool = False) -> bool:
    """True when the backend is isolated enough to skip dangerous-command prompts. Docker is the
    exception once host paths are bind-mounted: ``rm -rf /workspace`` then reaches host files."""
    if env_type == "docker":
        return not has_host_access
    return env_type in ("singularity", "modal", "daytona", "vercel_sandbox")


def _user_deny_block(command: str) -> dict | None:
    """The operator's ``approvals.deny`` rules are documented as never bypassable — not by yolo,
    not by mode=off, and not by an isolated container either: they express intent about what the
    agent may DO, not what it can reach, so they are evaluated before the container fast path."""
    deny_pattern = _match_user_deny_rule(command)
    if deny_pattern is None:
        return None
    logger.warning("User deny rule %r blocked command: %s", deny_pattern, command[:200])
    return _user_deny_block_result(deny_pattern)


def _floor_block(command: str, *, sudo_guard: bool = False) -> dict | None:
    """Unconditional floors, BEFORE yolo / mode=off / cron approve-mode so no
    session-level setting can bypass them: hardline catastrophic commands,
    password-piping to ``sudo -S`` with no SUDO_PASSWORD configured (full guard
    only), and the user's own approvals.deny rules ("never, even under yolo")."""
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc, command)
    if sudo_guard:
        is_sudo_guess, sudo_guess_desc = _check_sudo_stdin_guard(command)
        if is_sudo_guess:
            logger.warning("Sudo stdin guard block: %s (command: %s)", sudo_guess_desc, command[:200])
            return _sudo_stdin_block_result(sudo_guess_desc)
    return _user_deny_block(command)


def check_dangerous_command(command: str, env_type: str,
                            approval_callback=None,
                            has_host_access: bool = False) -> dict:
    """Detect a dangerous command and handle approval (pattern layer only). ``has_host_access``:
    a Docker sandbox that bind-mounts host paths must not skip approval.
    Returns ``{"approved": True/False, "message": str or None, ...}``."""
    if _should_skip_container_guards(env_type, has_host_access=has_host_access):
        return _user_deny_block(command) or _approved()
    blocked = _floor_block(command)
    if blocked is not None:
        return blocked
    if _yolo_active():
        return _approved()
    if _command_matches_permanent_allowlist(command):
        return _approved()
    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    if not is_dangerous:
        return _approved()
    return _run_approval_gate(
        pattern_key=pattern_key, description=description, display_target=command, approval_callback=approval_callback,
        subject=f"Command flagged as dangerous ({description})", noun="dangerous commands",
        advice="Find an alternative approach that avoids this command.",
        autoapprove_log_prefix="AUTO-APPROVED dangerous command in non-interactive non-gateway context",
    )


def request_tool_approval(tool_name: str, reason: str, *, rule_key: str = "", approval_callback=None) -> dict:
    """Escalate an arbitrary tool call to the human-approval gate.

    Entry point for a plugin ``pre_tool_call`` hook returning ``{"action": "approve", ...}``:
    it asks the SAME human gate as Tier-2 dangerous shell patterns (session/permanent
    allowlist, CLI prompt, gateway pending, once/session/always/deny, timeout fail-closed), so
    the LLM cannot skip it. Cron honors ``approvals.cron_mode``; any OTHER non-interactive
    non-gateway context fails CLOSED. ``rule_key`` controls the ``[a]lways`` allowlist grain;
    when empty it is ``tool_name`` + a hash of ``reason`` so DISTINCT reasons on the same tool
    persist independently. Returns the ``check_dangerous_command`` result shape.
    """
    description = reason or f"Plugin requires approval for {tool_name}"
    if not rule_key:
        rule_key = f"{tool_name}:{hashlib.sha256(description.encode('utf-8')).hexdigest()[:12]}"
    subject = f"Tool '{tool_name}' requires approval ({description})"
    return _run_approval_gate(
        # Namespaced so plugin-rule approvals share the allowlist machinery without ever colliding with a real
        # command pattern key; the display target is a synthetic label for the display/allowlist layer.
        pattern_key=f"plugin_rule:{rule_key}", description=description,
        display_target=f"<{tool_name}> (plugin approval rule)", approval_callback=approval_callback,
        subject=subject, advice="Find an alternative approach.",
        autoapprove_log_prefix=f"plugin-escalated tool call '{tool_name}' in non-interactive non-gateway context",
        fail_closed_when_no_human=True,
        no_human_block_message=(f"BLOCKED: {subject} but no interactive user or gateway is present "
                                "to approve it. A plugin flagged this action for human confirmation."),
    )


# --- Combined pre-exec guard (tirith + dangerous command detection) -------------------------------------------------

def _legacy_cron_tirith_fail_open() -> bool:
    """``security.tirith_fail_open`` via the legacy ``load_config`` reader.

    The cron ImportError path historically read fail-open through
    ``hermes_cli.config.load_config`` (deep-copying reader) rather than the
    readonly fast path used by :func:`_tirith_fail_open`. Both readers observe
    the same underlying config in production; consulting both keeps the cron
    fail-closed opt-in effective regardless of which reader a caller patches.
    Default True (fail-open) when config is unreadable.
    """
    try:
        from hermes_cli.config import load_config as _load_cfg
        _sec = (_load_cfg() or {}).get("security", {}) or {}
        if _sec.get("tirith_enabled", True):
            return bool(_sec.get("tirith_fail_open", True))
        return True
    except Exception:
        return True


def _cron_tirith_fail_open() -> bool:
    """Fail-open for the cron Tirith-unavailable path: open only when BOTH
    config readers agree. Either reader reporting fail-closed blocks — the
    deny-biased choice for a session with no human present."""
    return bool(_tirith_fail_open() and _legacy_cron_tirith_fail_open())


def _with_taint_source(result: dict, taint_source) -> dict:
    """Attach ``taint_source`` to an approved gate result when tainted.

    Denied/pending results carry no consent marker; approved ones record
    which provenance stream forced the human decision (``"turn"``,
    ``"history"``, ``"inherited"``, ``"explicit"``).
    """
    if taint_source is not None and result.get("approved") is True:
        result = dict(result)
        result["taint_source"] = taint_source
    return result


def _needs_human_unattended_block(command: str, extra_warnings: list,
                                  taint_source) -> dict:
    """Fail-closed block for tainted/sensitive actions with no human present.

    Used outside CLI/gateway/ask flows (plain headless, single-query, or an
    unattended platform): nobody can answer, so the action cannot run.
    """
    denied_desc = "; ".join(desc for _, desc, _ in extra_warnings)
    key, desc = extra_warnings[0][0], extra_warnings[0][1]
    logger.warning(
        "Blocked tainted/sensitive command with no approval surface: %s",
        command[:200],
    )
    return {
        "approved": False,
        "message": (
            f"BLOCKED: {denied_desc} No interactive user or gateway "
            "is present to approve it. Do NOT retry it, do NOT rephrase "
            "it, and do NOT attempt the same outcome via a different path."
        ),
        "pattern_key": key,
        "description": desc,
        "taint_source": taint_source,
    }


def _cron_guard_decision(command: str, extra_warnings: list,
                         taint_source) -> dict | None:
    """Cron-deny handling for the non-interactive guard path.

    Returns a decision, or None when nothing in the cron profile flags the
    command (the caller falls through to the shared unattended engine, which
    re-derives the same outcome with its own wording).

    Order: an explicitly trusted profile (``cron_mode != deny``) approves
    first; then dangerous-pattern and Tirith findings block with the shared
    cron wording; a Tirith ImportError honors fail-closed (blocking when
    EITHER config reader opted out of fail-open) and otherwise defers to the
    ``needs_human`` check below so tainted/sensitive reads cannot silently
    bypass authorization just because the scanner is unavailable. No
    unconditional approval is returned inside the ImportError branch.
    """
    if _get_cron_approval_mode() != "deny":
        # cron_mode == "approve" explicitly trusts this cron profile.
        return _approved()
    needs_human = bool(extra_warnings)
    is_dangerous, _pk, description = detect_dangerous_command(command)
    if is_dangerous:
        return _blocked(
            _CRON_CTX.block_message(
                f"Command flagged as dangerous ({description})",
                noun="dangerous commands",
                advice="Find an alternative approach that avoids this command."),
            pattern_key=_pk, description=description)
    try:
        from tools.tirith_security import check_command_security
        _cron_tirith = check_command_security(command)
    except ImportError:
        # Tirith not installed. Honour fail-closed: when an operator opted
        # out of fail-open the command cannot be silently allowed — and a
        # cron session has no user to approve it, so fail-closed means block
        # (mirrors the fail-closed synthesis in _tirith_scan; see #20733).
        if not _cron_tirith_fail_open():
            return _blocked(
                "BLOCKED: the Tirith security scanner could not be imported "
                "and security.tirith_fail_open is false, so this command "
                "cannot be silently allowed — and cron jobs run without a "
                "user present to approve it. Find an alternative approach, "
                "install tirith, or set approvals.cron_mode: approve in "
                "config.yaml.",
                pattern_key="tirith:tirith-import-error",
                description="Tirith security module unavailable")
        # Fail-open: do NOT approve here — the needs_human check below still
        # blocks tainted/sensitive reads (cron-deny must block them, not
        # allow them just because tirith is unavailable).
    else:
        if _cron_tirith.get("action") in ("block", "warn"):
            return _blocked(
                _CRON_CTX.block_message(
                    _format_tirith_description(_cron_tirith),
                    noun="dangerous commands",
                    advice="Find an alternative approach that avoids this command."),
                pattern_key="tirith:cron",
                description=_format_tirith_description(_cron_tirith))
    if needs_human:
        cron_extra_desc = "; ".join(desc for _, desc, _ in extra_warnings)
        key, desc = extra_warnings[0][0], extra_warnings[0][1]
        return {
            "approved": False,
            "message": (
                f"BLOCKED: {cron_extra_desc} "
                "but cron jobs run without a user present to approve it. "
                "Find an alternative approach that avoids this command. "
                "To allow flagged actions in cron jobs, set "
                "approvals.cron_mode: approve in config.yaml."
            ),
            "pattern_key": key,
            "description": desc,
            "taint_source": taint_source,
        }
    return None

def _format_tirith_description(tirith_result: dict) -> str:
    """Human-readable severity/title/description summary of tirith findings."""
    parts = []
    for f in tirith_result.get("findings") or []:
        severity, title, desc = f.get("severity", ""), f.get("title", ""), f.get("description", "")
        if title:
            text = f"{title}: {desc}" if desc else title
            parts.append(f"[{severity}] {text}" if severity else text)
    if not parts:
        summary = tirith_result.get("summary") or "security issue detected"
        return f"Security scan: {summary}"
    return "Security scan — " + "; ".join(parts)


def _tirith_scan(command: str) -> dict:
    """Tirith result for the interactive flow; an un-importable scanner allows
    (default) or, under fail-closed, synthesizes a HIGH warn finding that goes
    through the normal approval flow (#20733)."""
    try:
        from tools.tirith_security import check_command_security
        return check_command_security(command)
    except ImportError:
        if _tirith_fail_open():
            return {"action": "allow", "findings": [], "summary": ""}
        return {"action": "warn", "summary": "Tirith unavailable (fail-closed)", "findings": [{
            "rule_id": "tirith-import-error", "severity": "HIGH",
            "title": "Tirith security module unavailable",
            "description": ("The Tirith security scanner could not be imported. "
                            "Because security.tirith_fail_open is false, this "
                            "command cannot be silently allowed. Approve only if "
                            "you have verified the command is safe."),
        }]}


def check_all_command_guards(command: str, env_type: str,
                             approval_callback=None,
                             has_host_access: bool = False,
                             risk_metadata=None) -> dict:
    """Run all pre-exec security checks and return a single approval decision. Tirith and
    dangerous-command findings are presented as ONE combined approval request, so a gateway
    force=True replay cannot bypass one check when only the other was shown to the user.
    ``has_host_access``: a Docker sandbox with bind-mounted host paths takes the normal flow.

    ``risk_metadata`` is an optional explicit provenance record
    (``{"findings": [...], "risk": ...}``) for the current turn. It is set
    only by Hermes's own tool entries from ambient taint state — never from
    model-supplied tool arguments. When omitted, ambient turn taint for the
    active (session, turn) scope is consulted instead.
    """
    if _should_skip_container_guards(env_type, has_host_access=has_host_access):
        return _user_deny_block(command) or _approved()

    blocked = _floor_block(command, sudo_guard=True)
    if blocked is not None:
        return blocked

    approval_mode = _get_approval_mode()
    if _yolo_active() or approval_mode == "off":
        return _approved()
    if _command_matches_permanent_allowlist(command):
        return _approved()

    # Provenance + sensitive-read inputs. Resolved AFTER the yolo /
    # mode=off / permanent-allowlist bypasses (explicit user opt-outs win),
    # but BEFORE any auto-approve path below. ``taint`` is a taint record
    # (explicit kwarg wins, else ambient turn taint) or None when clean.
    # ``extra_warnings`` feeds both the headless fail-closed branch and the
    # Phase-1 warning list, so human and headless paths see identical inputs.
    taint = _resolve_guard_taint(risk_metadata)
    taint_source = taint.get("source") if taint is not None else None
    is_sensitive_read, sensitive_key, sensitive_desc = _detect_sensitive_read(command)
    extra_warnings = []
    if taint is not None and not _is_taint_exempt_command(command):
        extra_warnings.append((_TAINT_KEY, _taint_description(taint), False))
    if is_sensitive_read:
        extra_warnings.append((sensitive_key, sensitive_desc, False))
    needs_human = bool(extra_warnings)

    approval_callback, is_cli, is_gateway, is_ask = _presence(approval_callback)
    # Outside CLI/gateway/ask flows we never block on approvals: each
    # unattended context applies its configured deny/approve mode, else allow —
    # EXCEPT for tainted turns and sensitive reads, which fail closed when no
    # human can answer (cron honors cron_mode instead).
    if not is_cli and not is_gateway and not is_ask:
        if _is_cron_approval_context():
            cron_result = _cron_guard_decision(command, extra_warnings, taint_source)
            if cron_result is not None:
                return cron_result
        elif needs_human:
            return _needs_human_unattended_block(command, extra_warnings, taint_source)
        for ctx in _unattended_contexts():
            result = _unattended_deny(command, ctx)
            if result is not None:
                return result
        return _approved()

    # Gather findings: warnings = [(pattern_key, description, is_tirith)]. Tirith block AND warn both go through the
    # approval flow (block used to be a hard stop) so users can inspect the findings and approve.
    tirith_result = _tirith_scan(command)
    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    warnings = []
    session_key = get_current_session_key()
    if tirith_result["action"] in {"block", "warn"}:
        findings = tirith_result.get("findings") or []
        rule_id = findings[0].get("rule_id", "unknown") if findings else "unknown"
        tirith_key = f"tirith:{rule_id}"
        if not is_approved(session_key, tirith_key):
            warnings.append((tirith_key, _format_tirith_description(tirith_result), True))
    if is_dangerous and not is_approved(session_key, pattern_key):
        warnings.append((pattern_key, description, False))
    # Provenance + sensitive-read warnings. Session-approval short-circuit
    # mirrors the tirith/dangerous entries above: an explicit per-pattern
    # human decision still satisfies its own warning.
    for extra_key, extra_desc, extra_is_tirith in extra_warnings:
        if not is_approved(session_key, extra_key):
            warnings.append((extra_key, extra_desc, extra_is_tirith))
    if not warnings:
        return _approved()

    combined_desc = "; ".join(desc for _, desc, _ in warnings)
    primary_key = warnings[0][0]
    all_keys = [key for key, _, _ in warnings]
    # Taint warnings behave like tirith warnings for persistence: broad
    # permanent allowlisting of an untrusted-context gate is inappropriate,
    # so the UI must not offer a permanent scope. Tainted turns also skip
    # smart auto-approval entirely: the aux LLM would be judging an
    # attacker-shaped command, which reintroduces model trust into the
    # authorization path. A human always decides.
    has_taint = any(key.startswith(_TAINT_KEY_PREFIX) for key in all_keys)

    # "Always" is offered when at least one warning is a dangerous-pattern key the persistence layer would actually
    # allowlist permanently. Pure-tirith findings are session-max by design, so a tirith-only prompt hides Always;
    # mixed prompts offer it (the pattern key persists, tirith downgrades to session — see _persist_choice).
    return _with_taint_source(_human_decision(
        _COMMAND_GATE, command=command, description=combined_desc,
        pattern_key=primary_key, pattern_keys=all_keys, warnings=warnings,
        session_key=session_key, approval_callback=approval_callback,
        is_cli=is_cli, is_gateway=is_gateway, is_ask=is_ask, smart=approval_mode == "smart" and taint is None,
        permanent_capable=any(not is_t for _, _, is_t in warnings) and not has_taint,
    ), taint_source if has_taint else None)


_EXECUTE_CODE_DESCRIPTION = (
    "execute_code script execution. The script can spawn subprocesses or "
    "mutate files without passing through terminal command approval; approval is one-shot for this run."
)


def check_execute_code_guard(code: str, env_type: str, has_host_access: bool = False,
                             risk_metadata=None) -> dict:
    """Approve an execute_code script before its child process is spawned.

    The script can call ``subprocess``/``os.system``/``ctypes`` directly, none of which pass
    through ``terminal()`` / ``DANGEROUS_PATTERNS``; in gateway/ask contexts we fail closed by
    approving the script as a whole. Same dict contract as ``check_all_command_guards``.

    ``risk_metadata`` is an optional explicit provenance record, set only by
    Hermes's own tool entries from ambient taint state — never from
    model-supplied arguments. When omitted, ambient turn taint is consulted.

    Scope (documented limitation, #30882): in a purely local non-interactive
    non-gateway session (no TTY, not gateway, not cron-deny) an UNTAINTED
    script returns approved — matching the existing terminal auto-approve
    contract. A TAINTED script fails closed instead: arbitrary Python is a
    general capability that cannot be classified safe by inspection.

    See #30882.
    The hardline floor still blocks catastrophic ``terminal()`` commands the script issues; running
    arbitrary code headlessly without any approval surface is trusted-by-config (set a gateway/ask surface
    or ``approvals.cron_mode`` to require approval). See #30882.
    """
    pattern_key = "execute_code"
    description = _EXECUTE_CODE_DESCRIPTION

    # Isolated backends already sandbox the child. vercel_sandbox has no host-bind concept so it stays always-skipped.
    if env_type == "vercel_sandbox":
        return _approved()
    if _should_skip_container_guards(env_type, has_host_access=has_host_access):
        return _approved()
    approval_mode = _get_approval_mode()
    if _yolo_active() or approval_mode == "off":
        return _approved()

    # Provenance: resolved after the yolo/mode=off opt-out above. A
    # tainted script needs human authorization in every context below.
    taint = _resolve_guard_taint(risk_metadata)
    taint_source = taint.get("source") if taint is not None else None
    tainted = taint is not None
    taint_warnings = [(_TAINT_KEY, _taint_description(taint), False)] if tainted else []

    # (-q clears the presence flags, but its unattended context resolves first anyway.)
    approval_callback, is_cli, is_gateway, is_ask = _presence()
    # No user is present to approve arbitrary code in -q / cron / unattended
    # sessions: the first active context resolves instantly from its mode.
    # Tainted scripts skip profile trust: no approve-mode auto-approves them.
    if not tainted:
        for ctx in _unattended_contexts():
            if ctx.mode() == "deny":
                return _denied(
                    "BLOCKED: execute_code runs arbitrary local Python (including "
                    "subprocess calls that bypass shell-string approval checks). " + ctx.exec_tail,
                    pattern_key=pattern_key, description=description, outcome="blocked",
                )
            return _approved()

    # Only gateway/ask contexts get the one-shot whole-script approval — except
    # on tainted turns, where every context with a human requires it.
    #   * CLI interactive (untainted): the script's terminal() calls are guarded
    #     per-call (context now propagates into the RPC thread, #33057); a
    #     whole-script prompt would fire on every execute_code call.
    #   * Local non-interactive non-gateway (untainted): documented limitation.
    #   * Tainted CLI interactive: whole-script prompt below — the script can
    #     exfiltrate via sockets, files, or RPC tools without ever touching
    #     terminal(), so per-call guards are insufficient.
    #   * Tainted with no human surface: fail closed (cron honors cron_mode).
    if not is_gateway and not is_ask:
        if not tainted:
            return _approved()
        if is_cli:
            session_key = get_current_session_key()
            command = f"execute_code <<'PY'\n{code}\nPY"
            from agent.redact import redact_sensitive_text
            return _with_taint_source(_human_decision(
                _EXECUTE_CODE_GATE, command=command, description=description,
                pattern_key=pattern_key, pattern_keys=[pattern_key],
                warnings=[(pattern_key, None, False)] + taint_warnings,
                session_key=session_key, approval_callback=approval_callback,
                is_cli=is_cli, is_gateway=is_gateway, is_ask=is_ask,
                smart=False, permanent_capable=False,
                pending_body=lambda: f"**Code:**\n```python\n{redact_sensitive_text(code)}\n```",
            ), taint_source)
        return _denied(
            "BLOCKED: execute_code runs arbitrary local Python on a turn "
            "whose context contains untrusted high-risk content, but no "
            "interactive user or gateway is present to approve it. "
            "Do NOT retry, do NOT rephrase the script, and do NOT attempt "
            "the same outcome via a different tool.",
            pattern_key=pattern_key, description=description, outcome="blocked",
            taint_source=taint_source,
        )

    session_key = get_current_session_key()
    # Built only past the early-return gates so common paths don't copy a potentially-large script into this string.
    command = f"execute_code <<'PY'\n{code}\nPY"

    # Without this, "Approve session" / "Always" choices are stored but never
    # consulted, so every execute_code call re-prompts (#39275).
    # Tainted scripts always require a fresh human decision.
    if not tainted and is_approved(session_key, pattern_key):
        return _approved()

    # Smart mode: an APPROVE only suppresses the redundant whole-script prompt; the per-call terminal() guards still
    # run independently. The gateway renders the pending payload to Discord/Slack, so the script body is redacted for
    # display; the raw code is what gets assessed and run.
    # Tainted scripts skip smart auto-approval: the aux LLM would judge an
    # attacker-shaped script. A human always decides.
    from agent.redact import redact_sensitive_text
    return _with_taint_source(_human_decision(
        _EXECUTE_CODE_GATE, command=command, description=description, pattern_key=pattern_key,
        pattern_keys=[pattern_key], warnings=[(pattern_key, None, False)] + taint_warnings,
        session_key=session_key,
        approval_callback=approval_callback, is_cli=is_cli, is_gateway=is_gateway, is_ask=is_ask,
        smart=approval_mode == "smart" and not tainted,
        permanent_capable=not tainted,
        pending_body=lambda: f"**Code:**\n```python\n{redact_sensitive_text(code)}\n```",
    ), taint_source if tainted else None)


# Load permanent allowlist from config on module import
load_permanent_allowlist()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import contextlib  # noqa: F401,E402
import contextvars  # noqa: F401,E402
import fnmatch  # noqa: F401,E402
import functools  # noqa: F401,E402
import re  # noqa: F401,E402
import shlex  # noqa: F401,E402
import sys  # noqa: F401,E402
import tempfile  # noqa: F401,E402
import time  # noqa: F401,E402
import unicodedata  # noqa: F401,E402
import uuid  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'DANGEROUS_PATTERNS': ('tools.approval_detection', 'DANGEROUS_PATTERNS'),
    'DANGEROUS_PATTERNS_COMPILED': ('tools.approval_detection', 'DANGEROUS_PATTERNS_COMPILED'),
    'HARDLINE_PATTERNS': ('tools.approval_detection', 'HARDLINE_PATTERNS'),
    'HARDLINE_PATTERNS_COMPILED': ('tools.approval_detection', 'HARDLINE_PATTERNS_COMPILED'),
    'HUMAN_WAIT_MARGIN_S': ('tools.approval_human_wait', 'HUMAN_WAIT_MARGIN_S'),
    'cfg_get': ('hermes_cli.config', 'cfg_get'),
    'get_plugin_manager': ('tools.approval_prompt', 'get_plugin_manager'),
    'human_wait_ceiling': ('tools.approval_human_wait', 'human_wait_ceiling'),
    'human_wait_seconds': ('tools.approval_human_wait', 'human_wait_seconds'),
    'human_wait_window': ('tools.approval_human_wait', 'human_wait_window'),
    'is_interrupted': ('tools.interrupt', 'is_interrupted'),
    'request_elicitation_consent': ('tools.approval_prompt', 'request_elicitation_consent'),
    'reset_current_observability_context': ('tools.approval_context', 'reset_current_observability_context'),
    'reset_current_session_key': ('tools.approval_context', 'reset_current_session_key'),
    'reset_hermes_interactive_context': ('tools.approval_context', 'reset_hermes_interactive_context'),
    'set_current_observability_context': ('tools.approval_context', 'set_current_observability_context'),
    'set_current_session_key': ('tools.approval_context', 'set_current_session_key'),
    'set_hermes_interactive_context': ('tools.approval_context', 'set_hermes_interactive_context'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
