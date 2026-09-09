"""Cron live-lane integration with DeadTargetRegistry (deleted groups, blocked bots).

The gateway's ``DeliveryRouter.deliver()`` short-circuits confirmed-dead targets, but the
cron live lane sends through ``router._deliver_to_platform()`` directly (it needs the
Telegram three-mode topic routing, #22773) — bypassing the registry lifecycle. These tests
drive the REAL ``_deliver_result`` path with a REAL raising adapter (``DeliveryRouter``
itself is never mocked; only the platform adapter and scheduler plumbing are stubbed) and
assert the lifecycle against a temporary persistent registry:

  tick 1: live send attempted -> permanent Forbidden -> target recorded dead,
          no standalone resend to the same unreachable chat
  tick 2: same target -> registry says dead -> no live send, no standalone resend

A thread/topic-level ``not_found`` must NOT poison the whole chat (46f45104c4): the live
lane keeps trying every tick and the standalone fallback is preserved.
"""

import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig
from gateway.dead_targets import DeadTargetRegistry


CHAT_ID = "-1001234567890"


class RaisingAdapter:
    """Fake Telegram adapter: records sends, always raises a fixed error."""

    def __init__(self, message):
        self.message = message
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append(chat_id)
        raise RuntimeError(self.message)


def _job():
    # Fresh dict per tick — the scheduler reloads the job row on every fire.
    return {
        "id": "dead-target-cron-test",
        "name": "Dead Target Cron",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": CHAT_ID},
    }


def _gateway_config():
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    return config


@pytest.fixture
def registry_home(tmp_path, monkeypatch):
    """Temporary persistent registry path shared by every DeadTargetRegistry built
    on the delivery path (mirrors tests/gateway/test_dead_targets.py::isolate)."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("gateway.dead_targets.get_hermes_home", lambda: tmp_path)
    return tmp_path


def _run_tick(job, adapter, standalone_calls):
    """One cron tick through the real live-adapter + standalone lanes."""
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    async def fake_standalone(platform, pconfig, chat_id, text, **kwargs):
        standalone_calls.append(chat_id)
        return {}

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("tools.send_message_tool._send_to_platform", side_effect=fake_standalone),
        patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro),
    ):
        return _deliver_result(job, "hello", adapters={Platform.TELEGRAM: adapter}, loop=loop)


def test_permanent_forbidden_marks_dead_and_suppresses_resend(registry_home):
    adapter = RaisingAdapter("Forbidden: the group chat was deleted")
    standalone_calls = []

    # Tick 1: the live lane reaches the real adapter and fails permanently.
    error = _run_tick(_job(), adapter, standalone_calls)
    assert adapter.calls == [CHAT_ID]
    assert error is not None  # honestly failed, never silent success
    # A FRESH registry instance over the same backing file sees the death.
    assert DeadTargetRegistry().is_dead("telegram", CHAT_ID) is True
    # The standalone lane targets the same unreachable chat — no doomed resend.
    assert standalone_calls == []

    # Tick 2: the confirmed-dead target is skipped on every lane.
    error = _run_tick(_job(), adapter, standalone_calls)
    assert adapter.calls == [CHAT_ID]  # no second live send
    assert standalone_calls == []
    assert error is not None and "unreachable" in error


def test_thread_level_not_found_neither_marks_nor_suppresses_fallback(registry_home):
    adapter = RaisingAdapter("Bad Request: message thread not found")
    standalone_calls = []

    for _ in range(2):
        error = _run_tick(_job(), adapter, standalone_calls)
        assert error is None  # standalone fallback still delivers

    # Live lane retried every tick; fallback preserved; chat NOT marked dead.
    assert adapter.calls == [CHAT_ID, CHAT_ID]
    assert standalone_calls == [CHAT_ID, CHAT_ID]
    assert DeadTargetRegistry().is_dead("telegram", CHAT_ID) is False
