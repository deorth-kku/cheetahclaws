"""Tests for ChatSession.steer() — the web entry point for mid-run injection.

Covers:
  * idle steer  -> delegates to submit_prompt()
  * busy steer  -> persists a user turn, queues it on AgentState, broadcasts
                   a steer_queued event, and fixes the live-assistant ordering
  * slash steer -> delegates to _handle_slash()

Run with:  pytest tests/test_web_steer.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import cheetahclaws.web.db as db  # noqa: E402
from cheetahclaws.web.api import ChatSession  # noqa: E402


@pytest.fixture
def session(tmp_path):
    os.environ["CHEETAHCLAWS_WEB_DB"] = str(tmp_path / "t.db")
    db._engine = None
    db._SessionLocal = None
    db.init_db()
    u = db.repo.create_user("alice", "x", is_admin=True)
    sid = "sess1"
    db.repo.upsert_session(sid, u["id"], title="t")
    return db, sid, u["id"]


def _base_config():
    return {
        "model": "custom/qwen2.5-72b",
        "permission_mode": "accept-all",
        "no_tools": False,
        "_session_id": "test-steer",
    }


def _make_chat(db, sid, uid):
    return ChatSession(_base_config(), uid, session_id=sid)


def test_steer_when_idle_delegates_to_submit_prompt(session, monkeypatch):
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)

    called = {}

    def fake_submit(prompt):
        called["prompt"] = prompt
        return True

    monkeypatch.setattr(chat, "submit_prompt", fake_submit)
    assert chat.steer("hello") is True
    assert called["prompt"] == "hello"


def test_steer_when_busy_persists_queues_and_broadcasts(session, monkeypatch):
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)

    # Simulate the agent being mid-run.
    chat._busy.set()
    events = []
    monkeypatch.setattr(chat, "_broadcast", lambda ev: events.append(ev))

    assert chat.steer("mid-run instruction") is True

    # Queued on the shared AgentState for run() to drain before the next call.
    assert chat._agent_state.drain_pending_steers() == ["mid-run instruction"]

    # Persisted as a user turn in the DB (UI view).
    ui = db.repo.get_messages_for_ui(sid)
    assert any(
        m["role"] == "user" and m["content"] == "mid-run instruction"
        for m in ui
    )

    # Broadcast a steer_queued event so clients render a bubble.
    types = [e.type for e in events]
    assert "steer_queued" in types
    q = next(e for e in events if e.type == "steer_queued")
    assert q.data.get("text") == "mid-run instruction"


def test_steer_carries_image_into_queue_and_event(session, monkeypatch):
    """A steer with an attached image queues the image on AgentState so run()
    injects it into the next API call, and echoes text+image+type to clients."""
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)
    chat._busy.set()
    events = []
    monkeypatch.setattr(chat, "_broadcast", lambda ev: events.append(ev))

    assert chat.steer("look at this", "BASE64IMG", image_type="image/png") is True

    # Queued full entry (text + image) for run() to drain.
    entry = chat._agent_state.drain_pending_steers_full()[0]
    assert entry["text"] == "look at this"
    assert entry["image"] == "BASE64IMG"

    q = next(e for e in events if e.type == "steer_queued")
    assert q.data["text"] == "look at this"
    assert q.data["image"] == "BASE64IMG"
    assert q.data["imageType"] == "image/png"


def test_run_injects_steer_image_into_user_message(monkeypatch):
    """agent.run() attaches a steered image to the injected user turn so the
    next API call carries it — the web image-steer path."""
    from cheetahclaws import agent
    from cheetahclaws.agent import AgentState

    seen_messages = []

    def fake_stream(**kwargs):
        seen_messages.append(kwargs["messages"])
        yield agent._turn(text="ok")

    monkeypatch.setattr(agent, "stream", fake_stream)

    state = AgentState()
    state.steer("look at this", "IMGDATA", image_type="image/jpeg")
    list(agent.run("", state, _base_config(), "system"))

    # The steered user message (empty text, image attached) reached the API.
    assert seen_messages, "no API call was made"
    user_msgs = [m for m in seen_messages[0] if m.get("role") == "user"]
    injected = [m for m in user_msgs if m.get("images")]
    assert injected, "no steered user message carried the image"
    assert injected[-1]["images"] == ["IMGDATA"]
    assert injected[-1]["content"] == ""


def test_steer_when_busy_keeps_partial_assistant_row(session, monkeypatch):
    """A steer while an assistant row is live must KEEP that partial row so
    it renders BEFORE the steer user message on reload, then stop tracking it
    so the next streamed event opens a fresh continuation row at a larger id.

    Reload order therefore stays …assistant(partial) → user(steer)
    (the partial answer must NOT vanish, and the steer must NOT land next to
    the earlier user input). _ensure_live() resets the output accumulators
    when it opens the fresh continuation row.
    """
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)

    chat._busy.set()
    monkeypatch.setattr(chat, "_broadcast", lambda ev: None)

    # Simulate a live assistant row (partial output) already created at some id.
    mid = db.repo.append_message(sid, "assistant", "partial")
    chat._live_mid = mid
    chat._live_msg = {"role": "assistant", "content": "partial",
                      "blocks": [], "id": mid}
    chat.messages.append(chat._live_msg)

    chat.steer("steer after partial")

    # The live pointer is dropped (next event opens a fresh row), but the
    # partial row itself is KEPT in the DB and in-memory cache.
    assert chat._live_mid is None
    assert chat._live_msg is None
    ui = db.repo.get_messages_for_ui(sid)
    assert [m["role"] for m in ui] == ["assistant", "user"], ui
    assert ui[0]["content"] == "partial"                       # partial kept
    assert ui[1]["content"] == "steer after partial"           # steer after it
    # The partial assistant content is truly still there (not dropped).
    assert any(m["content"] == "partial" for m in ui)
    # The steer was still queued for the agent's next API call.
    assert chat._agent_state.drain_pending_steers() == ["steer after partial"]


def test_steer_slash_delegates_to_handle_slash(session, monkeypatch):
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)

    called = {}

    def fake_slash(line):
        called["line"] = line
        return True

    monkeypatch.setattr(chat, "_handle_slash", fake_slash)
    assert chat.steer("/clear") is True
    assert called["line"] == "/clear"


def test_run_turn_loop_auto_starts_new_turn_from_residual_steer(session, monkeypatch):
    """After a turn completes, a steer that survived (queued after the final
    API call) is drained and used as the next prompt — the residual auto-new
    turn. Steers consumed mid-run by run() do NOT trigger a new turn."""
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)

    prompts = []

    def fake_run_agent(prompt):
        prompts.append(prompt)
        # Simulate a steer that was queued AFTER the turn's final API call,
        # so run() never drained it — it survives as a residual. Only the
        # first turn leaves one, so the loop terminates after the 2nd turn.
        if len(prompts) == 1:
            chat._agent_state.steer("residual steer")

    monkeypatch.setattr(chat, "_run_agent", fake_run_agent)

    # No more residuals after the second turn → loop terminates.
    chat._run_turn_loop("first prompt")

    assert prompts == ["first prompt", "residual steer"]


def test_run_turn_loop_stops_when_no_residual(session, monkeypatch):
    """With no residual steer, the loop runs exactly one turn."""
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)

    prompts = []

    def fake_run_agent(prompt):
        prompts.append(prompt)

    monkeypatch.setattr(chat, "_run_agent", fake_run_agent)
    chat._run_turn_loop("only prompt")

    assert prompts == ["only prompt"]


def test_run_turn_loop_aborts_on_error(session, monkeypatch):
    """An exception in _run_agent aborts the residual loop (prompt set to None)."""
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)

    calls = {"n": 0}

    def fake_run_agent(prompt):
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(chat, "_run_agent", fake_run_agent)
    # Should not hang or loop forever despite a residual steer present.
    chat._agent_state.steer("leftover")
    chat._run_turn_loop("first")

    assert calls["n"] == 1  # error aborted the loop, residual never retried


def test_has_pending_image(session, monkeypatch):
    """_has_pending_image reflects the RuntimeContext.pending_image flag."""
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)
    from cheetahclaws import runtime
    ctx = runtime.get_session_ctx(chat.session_id)
    assert chat._has_pending_image() is False
    ctx.pending_image = "iVBORw0KGgo"
    assert chat._has_pending_image() is True
    ctx.pending_image = None
    assert chat._has_pending_image() is False


def test_run_turn_loop_runs_image_only_turn(session, monkeypatch):
    """An image-only send (empty prompt text + a pending image) must still run
    exactly one turn.

    Regression: the old ``while prompt:`` guard skipped empty prompts, so an
    image-only turn ran no agent, emitted no events, and the UI spinner never
    cleared (the OpenAI API never received a request). The pending image is
    consumed by ``run()`` (simulated here by clearing the flag), so the loop
    runs once and then exits.
    """
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)
    from cheetahclaws import runtime
    runtime.get_session_ctx(chat.session_id).pending_image = "iVBORw0KGgo"

    prompts = []

    def fake_run_agent(prompt):
        prompts.append(prompt)
        # Real run() consumes the pending image (attaches it to the user
        # message), so the loop must not spin again after this turn.
        runtime.get_session_ctx(chat.session_id).pending_image = None

    monkeypatch.setattr(chat, "_run_agent", fake_run_agent)
    chat._run_turn_loop("")

    assert prompts == [""]


def test_run_turn_loop_no_prompt_no_image_runs_nothing(session, monkeypatch):
    """With neither a prompt nor a pending image, the loop runs zero turns."""
    db, sid, uid = session
    chat = _make_chat(db, sid, uid)

    calls = {"n": 0}

    def fake_run_agent(prompt):
        calls["n"] += 1

    monkeypatch.setattr(chat, "_run_agent", fake_run_agent)
    chat._run_turn_loop("")

    assert calls["n"] == 0
