"""Fan-out resolution failures must surface as delivery errors, not silent success.

Regression coverage: ``_resolve_delivery_targets`` used to discard unresolvable
fan-out members (unknown platform, missing home channel, deleted bot-chat
profile) with no error accumulation, so ``_deliver_result`` returned ``None``
whenever a surviving sibling delivered fine — and the job recorded
``last_status="ok"`` while a requested target never received anything.
"""

import pytest

from cron.jobs import _record_run_outcome
from cron.scheduler import _deliver_result, _resolve_delivery_targets


@pytest.fixture()
def slack_sender(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with slack enabled; standalone sends recorded."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "D0HOME")
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

    send_calls = []

    async def fake_sender(pconfig, chat_id, message, *, thread_id=None,
                          media_files=None, force_document=False, caption=None):
        send_calls.append({"chat_id": chat_id, "thread_id": thread_id})
        return {"success": True, "chat_id": chat_id, "message_id": "1.2"}

    import hermes_cli.plugins as hp
    import gateway.platform_registry as reg

    entry = reg.platform_registry.get("slack")
    if entry is None:
        hp.discover_plugins()
        entry = reg.platform_registry.get("slack")
    if entry is None:
        pytest.skip("slack platform entry not registered")
    monkeypatch.setattr(entry, "standalone_sender_fn", fake_sender)
    monkeypatch.setattr(hp, "discover_plugins", lambda *a, **k: None)
    return send_calls


def _classify(job, delivery_error):
    recorded = dict(job)
    _record_run_outcome(recorded, True, None, delivery_error, None, "now")
    return recorded["last_status"]


class TestFanoutResolutionErrors:
    def test_valid_plus_unknown_target_returns_error(self, slack_sender):
        job = {
            "id": "j1", "name": "fanout",
            "deliver": "slack:D0GOOD,nosuchplatformXYZ", "origin": None,
        }
        err = _deliver_result(job, "hello", adapters=None, loop=None)
        assert err is not None, "dropped fan-out member must fail delivery accounting"
        assert "nosuchplatformXYZ" in err
        assert [c["chat_id"] for c in slack_sender] == ["D0GOOD"]
        assert _classify(job, err) == "delivery_failed"

    def test_single_invalid_target_still_fails_as_before(self, slack_sender):
        job = {"id": "j2", "name": "bad", "deliver": "nosuchplatformXYZ",
               "origin": None}
        err = _deliver_result(job, "hello", adapters=None, loop=None)
        assert err == "no delivery target resolved for deliver=nosuchplatformXYZ"
        assert slack_sender == []
        assert _classify(job, err) == "delivery_failed"

    def test_local_remains_intentional_noop(self, slack_sender):
        job = {"id": "j3", "name": "local", "deliver": "local", "origin": None}
        errors: list = []
        assert _resolve_delivery_targets(job, resolution_errors=errors) == []
        assert errors == []
        err = _deliver_result(job, "hello", adapters=None, loop=None)
        assert err is None
        assert slack_sender == []
        assert _classify(job, err) == "ok"

    def test_valid_fanout_unchanged(self, slack_sender):
        job = {
            "id": "j4", "name": "good-fanout",
            "deliver": "slack:D0A,slack:D0B", "origin": None,
        }
        err = _deliver_result(job, "hello", adapters=None, loop=None)
        assert err is None
        assert sorted(c["chat_id"] for c in slack_sender) == ["D0A", "D0B"]
        assert _classify(job, err) == "ok"

    def test_missing_bot_chat_profile_reported(self, slack_sender):
        job = {
            "id": "j5", "name": "bot-chat-fanout",
            "deliver": "slack:D0GOOD,bot-chat:deleted-profile-xyz",
            "origin": None,
        }
        err = _deliver_result(job, "hello", adapters=None, loop=None)
        assert err is not None
        assert "deleted-profile-xyz" in err
        assert [c["chat_id"] for c in slack_sender] == ["D0GOOD"]
        assert _classify(job, err) == "delivery_failed"

    def test_bare_platform_without_home_reported(self, slack_sender):
        job = {
            "id": "j6", "name": "no-home-fanout",
            "deliver": "slack:D0GOOD,discord", "origin": None,
        }
        err = _deliver_result(job, "hello", adapters=None, loop=None)
        assert err is not None
        assert "discord" in err
        assert [c["chat_id"] for c in slack_sender] == ["D0GOOD"]
        assert _classify(job, err) == "delivery_failed"

    def test_origin_without_origin_stays_silent(self, slack_sender, monkeypatch):
        """Origin-less ``origin`` is the documented no-op lane (#43014), not a
        genuine resolution failure — it must not flip a succeeding fan-out."""
        monkeypatch.delenv("SLACK_HOME_CHANNEL")
        job = {
            "id": "j7", "name": "origin-fanout",
            "deliver": "origin,slack:D0GOOD", "origin": None,
        }
        errors: list = []
        targets = _resolve_delivery_targets(job, resolution_errors=errors)
        assert [t["chat_id"] for t in targets] == ["D0GOOD"]
        assert errors == []
        err = _deliver_result(job, "hello", adapters=None, loop=None)
        assert err is None
        assert [c["chat_id"] for c in slack_sender] == ["D0GOOD"]

    def test_accumulator_defaults_to_legacy_behavior(self):
        """Existing callers pass no accumulator: list return, no error surfacing."""
        job = {"id": "j8", "deliver": "slack:D0X,nosuchplatformXYZ",
               "origin": None}
        targets = _resolve_delivery_targets(job)
        assert [(t["platform"], t["chat_id"]) for t in targets] == [
            ("slack", "D0X")]
