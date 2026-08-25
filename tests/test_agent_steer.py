"""Tests for the steer feature.

Steer lets the user inject a user turn *mid-run* so it lands in the NEXT API
request. The queued messages live on ``AgentState.pending_user_turns`` and are
drained at the top of each ``run()`` loop iteration (top-level only). The web
layer exposes ``ChatSession.steer()`` (see tests/test_web_steer.py for the web
side).

Run with:  pytest tests/test_agent_steer.py -v
"""
from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws import agent
from cheetahclaws.agent import AgentState, run
from cheetahclaws.providers import AssistantTurn, TextChunk


# ── Unit: AgentState steer queue ─────────────────────────────────────────


def test_state_steer_queue_basic():
    state = AgentState()
    assert state.has_pending_steers() is False
    state.steer("first")
    state.steer("second")
    assert state.has_pending_steers() is True

    drained = state.drain_pending_steers()
    assert drained == ["first", "second"]
    # drain clears the queue
    assert state.drain_pending_steers() == []
    assert state.has_pending_steers() is False


def test_state_steer_queue_thread_safety():
    """Many threads steering concurrently; drain must return every message
    exactly once with no race corruption."""
    state = AgentState()
    n_threads, per = 16, 200

    def worker(tid):
        for i in range(per):
            state.steer(f"t{tid}-{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    drained = state.drain_pending_steers()
    assert len(drained) == n_threads * per
    assert sorted(drained) == sorted(
        f"t{t}-{i}" for t in range(n_threads) for i in range(per)
    )
    assert state.drain_pending_steers() == []


# ── Helpers ──────────────────────────────────────────────────────────────


def _turn(text: str = "", tool_calls: list | None = None) -> AssistantTurn:
    t = AssistantTurn.__new__(AssistantTurn)
    t.text = text
    t.tool_calls = tool_calls or []
    t.in_tokens = 1
    t.out_tokens = 1
    t.cache_read_tokens = 0
    t.cache_write_tokens = 0
    return t


def _config(**extra) -> dict:
    return {
        "model": "custom/qwen2.5-72b",
        "permission_mode": "accept-all",
        "no_tools": False,
        "_session_id": "test-steer",
        **extra,
    }


# ── run() loop integration ───────────────────────────────────────────────


def test_top_level_injects_preseeded_steer_before_first_api_call(monkeypatch):
    """A steer queued before run() is injected as a user turn before the
    first API request (top-level loop drain)."""
    seen_messages = []

    def fake_stream(**kwargs):
        seen_messages.append([m["role"] for m in kwargs["messages"]])
        yield _turn(text="all done")

    monkeypatch.setattr(agent, "stream", fake_stream)

    state = AgentState()
    state.steer("please also check the tests")
    list(run("hello", state, _config(), "system"))

    roles = [m["role"] for m in state.messages]
    assert roles == ["user", "user", "assistant"], roles
    assert state.messages[1]["content"] == "please also check the tests"
    # drained after injection
    assert state.drain_pending_steers() == []
    # the steer was actually in the first API request's messages
    assert seen_messages[0].count("user") == 2


def test_subagent_depth_does_not_drain(monkeypatch):
    """A steer queued before run(depth=1) must NOT be injected — sub-agents
    get their own run() and must not consume the parent's steer queue."""
    monkeypatch.setattr(agent, "stream",
                        lambda **kw: iter([_turn(text="sub done")]))

    state = AgentState()
    state.steer("injected into parent only")
    list(run("hello", state, _config(), "system", depth=1))

    roles = [m["role"] for m in state.messages]
    assert roles == ["user", "assistant"], roles
    # untouched — still queued for the parent to consume
    assert state.drain_pending_steers() == ["injected into parent only"]


def test_steer_injected_between_tool_rounds(monkeypatch):
    """A steer arriving during the first tool round-trip is injected before
    the SECOND API call, right after the tool result."""
    tool_call = _turn(tool_calls=[{
        "id": "t1", "name": "WebFetch",
        "input": {"url": "https://example.test"},
    }])
    text_turn = _turn(text="final answer")
    state = AgentState()
    idx = {"i": -1}

    def fake_stream(**kwargs):
        idx["i"] += 1
        if idx["i"] == 0:
            # steer arrives mid-first-round-trip (e.g. user typed it while the
            # model was executing tools)
            state.steer("after the tool, please summarize")
        yield tool_call if idx["i"] == 0 else text_turn

    monkeypatch.setattr(agent, "stream", fake_stream)
    list(run("hello", state, _config(tool_profile="standard"), "system"))

    roles = [m["role"] for m in state.messages]
    assert roles == ["user", "assistant", "tool", "user", "assistant"], roles
    assert state.messages[3]["content"] == "after the tool, please summarize"


def test_steer_reaches_next_api_call_messages(monkeypatch):
    """The steer must be present in state.messages at the time of the NEXT
    API request (i.e. actually sent to the provider)."""
    tool_call = _turn(tool_calls=[{
        "id": "t1", "name": "WebFetch",
        "input": {"url": "https://example.test"},
    }])
    text_turn = _turn(text="final answer")
    seen_messages = []
    state = AgentState()
    idx = {"i": -1}

    def fake_stream(**kwargs):
        idx["i"] += 1
        seen_messages.append([m["role"] for m in kwargs["messages"]])
        if idx["i"] == 0:
            state.steer("inject me into the next call")
        yield tool_call if idx["i"] == 0 else text_turn

    monkeypatch.setattr(agent, "stream", fake_stream)
    list(run("hello", state, _config(tool_profile="standard"), "system"))

    # First API call: only the original user turn.
    assert seen_messages[0] == ["user"]
    # Second API call includes the injected steer as a user message.
    assert seen_messages[1].count("user") == 2
    user_contents = [m["content"] for m in state.messages if m.get("role") == "user"]
    assert "inject me into the next call" in user_contents


def test_multiple_steers_drained_in_order(monkeypatch):
    """Queued steers are injected in FIFO order before the next API call."""
    monkeypatch.setattr(agent, "stream",
                        lambda **kw: iter([_turn(text="done")]))

    state = AgentState()
    state.steer("one")
    state.steer("two")
    state.steer("three")
    list(run("hello", state, _config(), "system"))

    steer_msgs = [m["content"] for m in state.messages
                  if m["role"] == "user" and m["content"] != "hello"]
    assert steer_msgs == ["one", "two", "three"]
