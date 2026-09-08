"""Regression tests: profile runtime scope in ephemeral agent threads (#50233).

Why: normal prompt turns bind ``session['profile_home']`` via
``set_hermes_home_override`` before ``run_conversation`` so the turn runs against
the correct profile home. The two ephemeral RPC paths — ``prompt.background`` and
``preview.restart`` — spawn a fresh ``AIAgent`` on a NEW thread, and the
``HERMES_HOME`` ContextVar set on the session-create thread does NOT propagate to
those threads. Without an explicit re-bind, a background/preview-restart turn under
a non-default profile would run against the wrong home. This module locks in:

  1. ``prompt.background`` re-binds ``profile_home`` for the ephemeral turn.
  2. ``preview.restart`` re-binds ``profile_home`` for the ephemeral turn AND does
     NOT close the ephemeral agent (a task-wide ``AIAgent.close()`` would kill the
     background server the restart just started — maintainer problem #1).
  3. Both paths RESTORE the override after the turn (reset token from set), exactly
     like the normal prompt turn, and skip the bind entirely when no profile is set.
  4. Both paths bind the FULL profile runtime scope (HERMES_HOME + secret scope +
     terminal scope), not HOME alone: HOME without the secret scope resolves the
     side agent's credentials from the launch profile's ambient environment, and
     HOME without the terminal scope resolves its execution backend/policy from
     there too.

How to test: run this module with pytest; each test drives the real RPC handler
from ``tui_gateway.server._methods`` with ``threading.Thread`` patched to run the
target inline, then asserts on the recorded override set/reset calls and the agent.
The scope tests (4) capture real ``get_hermes_home()`` / ``get_secret()`` /
``terminal_env()`` values inside the ephemeral turn: no scope machinery is mocked
away, so they fail against the old HOME-only implementation.
"""

from unittest.mock import MagicMock, patch

import pytest

from tui_gateway import server as srv


PROFILE_HOME = "/home/user/.hermes/profiles/work"


class _InlineThread:
    """Drop-in for ``threading.Thread`` that runs the target synchronously.

    Why: the ephemeral RPC handlers do their real work inside ``run()`` on a
    spawned thread; running it inline makes the override set/reset observable
    within the test without racing a real background thread.
    """

    def __init__(self, target=None, daemon=None, **_kwargs):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()


@pytest.fixture
def fake_session():
    """A minimal session carrying a non-default ``profile_home``."""
    agent = MagicMock()
    return {"agent": agent, "session_key": "sess_k", "profile_home": PROFILE_HOME}


@pytest.fixture
def override_calls():
    """Patch set/reset override + AIAgent + emit/context helpers; record calls.

    Returns a dict with the mocks so each test can assert on set/reset ordering
    and on whether the ephemeral agent was closed.
    """
    agent_instance = MagicMock()
    # run_conversation returns a plain dict like the real agent does.
    agent_instance.run_conversation.return_value = {"final_response": "done"}

    with patch("tui_gateway.server.threading.Thread", _InlineThread), \
        patch("tui_gateway.server.set_hermes_home_override", return_value="TOK") as m_set, \
        patch("tui_gateway.server.reset_hermes_home_override") as m_reset, \
        patch("tui_gateway.server._background_agent_kwargs", return_value={}), \
        patch("tui_gateway.server._ephemeral_preview_agent_kwargs", return_value={}), \
        patch("tui_gateway.server._preview_restart_callbacks", return_value={}), \
        patch("tui_gateway.server._preview_restart_history", return_value=[]), \
        patch("tui_gateway.server._set_session_context", return_value=None), \
        patch("tui_gateway.server._clear_session_context"), \
        patch("tui_gateway.server._session_cwd", return_value="/tmp"), \
        patch("tui_gateway.server._emit"), \
        patch("run_agent.AIAgent", return_value=agent_instance) as m_agent:
        yield {
            "set": m_set,
            "reset": m_reset,
            "agent_cls": m_agent,
            "agent": agent_instance,
        }


def _run(method_name, params, session):
    """Invoke a registered RPC handler with ``_sess`` patched to our session."""
    handler = srv._methods[method_name]
    with patch("tui_gateway.server._sess", return_value=(session, None)):
        return handler("rid1", params)


class TestBackgroundProfileOverride:
    def test_background_binds_and_restores_profile_home(self, fake_session, override_calls):
        """prompt.background binds profile_home for the ephemeral turn and restores it."""
        _run("prompt.background", {"text": "hi", "session_id": "ui1"}, fake_session)

        override_calls["set"].assert_called_once_with(PROFILE_HOME)
        override_calls["reset"].assert_called_once_with("TOK")
        override_calls["agent"].run_conversation.assert_called_once()

    def test_background_no_profile_skips_override(self, override_calls):
        """With no profile_home the background path never touches the override."""
        session = {"agent": MagicMock(), "session_key": "sess_k", "profile_home": None}
        _run("prompt.background", {"text": "hi", "session_id": "ui1"}, session)

        override_calls["set"].assert_not_called()
        override_calls["reset"].assert_not_called()
        override_calls["agent"].run_conversation.assert_called_once()

    def test_background_restores_override_on_error(self, fake_session, override_calls):
        """A failing turn must still restore the override (finally-block parity)."""
        override_calls["agent"].run_conversation.side_effect = RuntimeError("boom")
        _run("prompt.background", {"text": "hi", "session_id": "ui1"}, fake_session)

        override_calls["set"].assert_called_once_with(PROFILE_HOME)
        override_calls["reset"].assert_called_once_with("TOK")


class TestPreviewRestartProfileOverride:
    def test_preview_binds_and_restores_profile_home(self, fake_session, override_calls):
        """preview.restart binds profile_home for the ephemeral turn and restores it."""
        _run(
            "preview.restart",
            {"url": "http://localhost:5173", "cwd": "", "session_id": "ui1"},
            fake_session,
        )

        override_calls["set"].assert_called_once_with(PROFILE_HOME)
        override_calls["reset"].assert_called_once_with("TOK")
        override_calls["agent"].run_conversation.assert_called_once()

    def test_preview_does_not_close_agent(self, fake_session, override_calls):
        """The restarted preview server must survive: the ephemeral agent is NOT
        closed via task-wide process cleanup (maintainer problem #1)."""
        _run(
            "preview.restart",
            {"url": "http://localhost:5173", "cwd": "", "session_id": "ui1"},
            fake_session,
        )

        # A task-wide AIAgent.close() would kill every process for this task_id,
        # tearing down the very background server the restart just launched.
        override_calls["agent"].close.assert_not_called()

    def test_preview_no_profile_skips_override(self, override_calls):
        """With no profile_home the preview path never touches the override."""
        session = {"agent": MagicMock(), "session_key": "sess_k", "profile_home": None}
        _run(
            "preview.restart",
            {"url": "http://localhost:5173", "cwd": "", "session_id": "ui1"},
            session,
        )

        override_calls["set"].assert_not_called()
        override_calls["reset"].assert_not_called()
        override_calls["agent"].close.assert_not_called()


class TestSideAgentProfileRuntimeScope:
    """The ephemeral turn binds the FULL profile runtime scope (HOME + secret +
    terminal), not HOME alone.

    Setup: a tmp target profile P (``.env`` holds ``SIDECAR_PROBE_KEY=profile-val``,
    ``config.yaml`` pins ``terminal.backend: docker``) while the ambient process
    environment holds the launch profile's values (``SIDECAR_PROBE_KEY=launch-val``,
    ``TERMINAL_ENV=local``). The real ``prompt.background`` handler runs inline and
    the mocked agent captures real ``get_hermes_home()`` / ``get_secret()`` /
    ``terminal_env()`` values inside the turn. No scope machinery is mocked away:
    against the old HOME-only implementation the secret assertions observe the
    launch value with no scope installed, and the terminal assertions observe the
    ambient launch backend.
    """

    @pytest.fixture
    def profile_home(self, tmp_path):
        home = tmp_path / "profiles" / "work"
        home.mkdir(parents=True)
        (home / ".env").write_text("SIDECAR_PROBE_KEY=profile-val\n")
        (home / "config.yaml").write_text("terminal:\n  backend: docker\n")
        return home

    @pytest.fixture
    def launch_env(self, monkeypatch):
        monkeypatch.setenv("SIDECAR_PROBE_KEY", "launch-val")
        monkeypatch.setenv("TERMINAL_ENV", "local")

    @pytest.fixture
    def scoped_run(self, profile_home):
        """Run ``prompt.background`` inline; capture the real scope state observed
        inside the ephemeral turn. Returns (session, captured)."""
        from agent.secret_scope import current_secret_scope, get_secret
        from hermes_constants import get_hermes_home
        from tools.terminal_scope import get_terminal_scope, terminal_env

        captured = {}
        agent_instance = MagicMock()

        def _capture(**kwargs):
            captured["home"] = str(get_hermes_home())
            captured["scope"] = current_secret_scope()
            captured["secret"] = get_secret("SIDECAR_PROBE_KEY")
            captured["term_scope"] = get_terminal_scope()
            captured["terminal_env"] = terminal_env("TERMINAL_ENV", "local")
            return {"final_response": "done"}

        agent_instance.run_conversation.side_effect = _capture
        session = {
            "agent": MagicMock(), "session_key": "sess_k",
            "profile_home": str(profile_home),
        }
        with patch("tui_gateway.server.threading.Thread", _InlineThread), \
            patch("tui_gateway.server._background_agent_kwargs", return_value={}), \
            patch("tui_gateway.server._set_session_context", return_value=None), \
            patch("tui_gateway.server._clear_session_context"), \
            patch("tui_gateway.server._session_cwd", return_value="/tmp"), \
            patch("tui_gateway.server._emit"), \
            patch("run_agent.AIAgent", return_value=agent_instance):
            with patch("tui_gateway.server._sess", return_value=(session, None)):
                handler = srv._methods["prompt.background"]
                resp = handler(
                    "rid1", {"text": "hi", "session_id": "ui1"})
        assert resp.get("result", {}).get("task_id", "").startswith("bg_")
        return session, captured

    def test_side_agent_binds_home(self, launch_env, scoped_run):
        """HOME resolves to the target profile inside the ephemeral turn."""
        _session, captured = scoped_run
        assert captured["home"] == _session["profile_home"]

    def test_side_agent_binds_secret_scope(self, launch_env, scoped_run):
        """Credentials resolve from the target profile, not ambient launch env."""
        from agent.secret_scope import get_secret

        _session, captured = scoped_run
        assert captured["scope"] is not None
        assert captured["scope"].get("SIDECAR_PROBE_KEY") == "profile-val"
        assert captured["secret"] == "profile-val"
        # ... and the scope is released afterwards (no leak into this context).
        assert get_secret("SIDECAR_PROBE_KEY") == "launch-val"

    def test_side_agent_binds_terminal_scope(self, launch_env, scoped_run):
        """Terminal authority resolves from the target profile's config."""
        from tools.terminal_scope import get_terminal_scope, terminal_env

        _session, captured = scoped_run
        assert captured["term_scope"] is not None
        assert captured["terminal_env"] == "docker"
        # ... and the scope is released afterwards (ambient restored).
        assert get_terminal_scope() is None
        assert terminal_env("TERMINAL_ENV", "local") == "local"

    def test_side_agent_profiless_unchanged(self, launch_env):
        """Without profile_home the turn runs fully ambient, as before."""
        from agent.secret_scope import current_secret_scope, get_secret
        from hermes_constants import get_hermes_home
        from tools.terminal_scope import get_terminal_scope

        ambient_home = str(get_hermes_home())
        captured = {}
        agent_instance = MagicMock()

        def _capture(**kwargs):
            captured["home"] = str(get_hermes_home())
            captured["scope"] = current_secret_scope()
            captured["secret"] = get_secret("SIDECAR_PROBE_KEY")
            captured["term_scope"] = get_terminal_scope()
            return {"final_response": "done"}

        agent_instance.run_conversation.side_effect = _capture
        session = {"agent": MagicMock(), "session_key": "sess_k", "profile_home": None}
        with patch("tui_gateway.server.threading.Thread", _InlineThread), \
            patch("tui_gateway.server._background_agent_kwargs", return_value={}), \
            patch("tui_gateway.server._set_session_context", return_value=None), \
            patch("tui_gateway.server._clear_session_context"), \
            patch("tui_gateway.server._session_cwd", return_value="/tmp"), \
            patch("tui_gateway.server._emit"), \
            patch("run_agent.AIAgent", return_value=agent_instance):
            with patch("tui_gateway.server._sess", return_value=(session, None)):
                srv._methods["prompt.background"](
                    "rid1", {"text": "hi", "session_id": "ui1"})
        assert captured["home"] == ambient_home
        assert captured["scope"] is None
        assert captured["secret"] == "launch-val"
        assert captured["term_scope"] is None
