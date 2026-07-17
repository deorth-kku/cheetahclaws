"""Tests for compaction persistence: one DB store, two read projections.

Verifies that:
  * get_messages_for_ui returns the full real history (compact rows excluded)
  * get_messages_for_agent returns compact block + recent real rows, in order
  * the neutral boundary mapping (_compute_after_id) locates the right row
  * stacked compactions keep exactly one compact marker
  * a reconnect rebuilds the compacted AgentState (no context re-inflation)
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
from cheetahclaws.agent import AgentState  # noqa: E402


@pytest.fixture
def session(tmp_path):
    os.environ["CHEETAHCLAWS_WEB_DB"] = str(tmp_path / "t.db")
    # Force a fresh engine bound to the temp DB.
    db._engine = None
    db._SessionLocal = None
    db.init_db()
    u = db.repo.create_user("alice", "x", is_admin=True)
    sid = "sess1"
    db.repo.upsert_session(sid, u["id"], title="t")
    return db, sid


def _real_rows(db, sid):
    """Append a couple of realistic real turns and return their neutral form."""
    db.repo.append_message(sid, "user", "hello")
    db.repo.append_message(
        sid, "assistant", "I'll check.",
        blocks=[{"type": "text", "text": "I'll check."},
                {"type": "tool", "name": "Bash", "tool_id": "t1",
                 "inputs": {}, "status": "done", "result": "out"}])
    db.repo.append_message(sid, "user", "again?")
    db.repo.append_message(
        sid, "assistant", "done",
        blocks=[{"type": "text", "text": "done"}])
    rows = db.repo.get_messages_for_ui(sid)
    neutral = ChatSession._messages_to_neutral(rows)
    return rows, neutral


def test_ui_view_excludes_compact(session):
    db, sid = session
    _real_rows(db, sid)
    state = AgentState()
    state.messages = ChatSession._messages_to_neutral(
        db.repo.get_messages_for_ui(sid))
    summary = "[Previous conversation summary]\nSUMMARY"
    ack = "Understood. I have the context from the previous conversation. Let's continue."
    after_id = ChatSession._compute_after_id(
        db.repo.get_messages_for_ui(sid), state.messages, split=1)
    assert after_id is not None
    db.repo.upsert_compaction(sid, summary, ack, after_id)

    ui = db.repo.get_messages_for_ui(sid)
    assert all(not m.get("is_compact") for m in ui)
    assert [m["role"] for m in ui] == ["user", "assistant", "user", "assistant"]


def test_agent_view_order_and_content(session):
    db, sid = session
    rows, neutral = _real_rows(db, sid)
    summary = "[Previous conversation summary]\nSUMMARY"
    ack = "Understood. I have the context from the previous conversation. Let's continue."
    after_id = ChatSession._compute_after_id(rows, neutral, split=1)
    db.repo.upsert_compaction(sid, summary, ack, after_id)

    agent_rows = db.repo.get_messages_for_agent(sid)
    assert agent_rows[0]["is_compact"] and agent_rows[0]["role"] == "user"
    assert agent_rows[1]["is_compact"] and agent_rows[1]["role"] == "assistant"
    assert [m["role"] for m in agent_rows if not m["is_compact"]] == \
        ["assistant", "user", "assistant"]

    reconstructed = ChatSession._messages_to_neutral(agent_rows)
    assert reconstructed[0]["content"].endswith("SUMMARY")
    assert reconstructed[1]["role"] == "assistant"
    assert reconstructed[2:] == neutral[1:]


def test_compute_after_id_finds_boundary(session):
    db, sid = session
    rows, neutral = _real_rows(db, sid)
    after_id = ChatSession._compute_after_id(rows, neutral, split=3)
    assert after_id == rows[1]["id"]
    db.repo.upsert_compaction(sid, "s", "a", after_id)
    recent = [m for m in db.repo.get_messages_for_agent(sid)
              if not m["is_compact"]]
    assert [m["role"] for m in recent] == ["user", "assistant"]


def test_stacked_compaction_keeps_one_marker(session):
    db, sid = session
    rows, neutral = _real_rows(db, sid)
    a1 = ChatSession._compute_after_id(rows, neutral, split=3)
    db.repo.upsert_compaction(sid, "summary1", "ack1", a1)
    compacted = ChatSession._messages_to_neutral(
        db.repo.get_messages_for_agent(sid))
    a2 = ChatSession._compute_after_id(rows, compacted, split=1)
    db.repo.upsert_compaction(sid, "summary2", "ack2", a2)

    import sqlalchemy
    with db.session_scope() as s:
        from cheetahclaws.web.models import Message
        n_compact = s.scalar(
            sqlalchemy.select(sqlalchemy.func.count(Message.id))
            .where(Message.session_id == sid,
                   Message.is_compact == True))  # noqa: E712
    assert n_compact == 2
    agent_rows = db.repo.get_messages_for_agent(sid)
    assert agent_rows[0]["content"].endswith("summary2")


def test_reconnect_rebuilds_compacted_state(session):
    db, sid = session
    rows, neutral = _real_rows(db, sid)
    after_id = ChatSession._compute_after_id(rows, neutral, split=1)
    db.repo.upsert_compaction(sid, "SUMMARY", "ACK", after_id)

    rebuilt = ChatSession._messages_to_neutral(
        db.repo.get_messages_for_agent(sid))
    assert any(m["content"].endswith("SUMMARY") for m in rebuilt)
    assert "hello" not in [m.get("content", "") for m in rebuilt
                           if m["role"] == "user"]


def test_no_compaction_when_split_zero(session):
    db, sid = session
    rows, neutral = _real_rows(db, sid)
    after_id = ChatSession._compute_after_id(rows, neutral, split=0)
    assert after_id is None
