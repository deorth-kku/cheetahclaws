"""Regression: OpenAI-compat streaming must surface reasoning as ThinkingChunk.

Backends disagree on the field name:
  - DeepSeek / older vLLM: delta.reasoning_content
  - Newer vLLM + OpenAI guidance: delta.reasoning
  - Some proxies nest {"content": "..."} under either key

Previously stream_openai_compat only read `reasoning_content`, so clients
connected to backends that emit `reasoning` never saw thinking traces —
CLI --verbose stayed silent and the web UI only ever showed "Processing".
"""
from __future__ import annotations

import openai as _openai

from cheetahclaws.providers import (
    AssistantTurn,
    TextChunk,
    ThinkingChunk,
    _openai_reasoning_delta,
    stream_openai_compat,
)


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None,
                 reasoning_content=None, reasoning=None, model_extra=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.reasoning = reasoning
        if model_extra is not None:
            self.model_extra = model_extra


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, delta, usage=None):
        self.choices = [_FakeChoice(delta)]
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        return iter(self.chunks)


class _FakeChatCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks):
        self.completions = _FakeChatCompletions(chunks)


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, default_headers=None):
        self.chat = _FakeChat([])

    def _set_chunks(self, chunks):
        self.chat = _FakeChat(chunks)


def _run_stream(monkeypatch, chunks, config=None, model="custom/deepseek-r1"):
    fake = _FakeOpenAI()
    fake._set_chunks(chunks)
    monkeypatch.setattr(_openai, "OpenAI", lambda **k: fake)
    events = list(stream_openai_compat(
        api_key="dummy",
        base_url="http://localhost:8000/v1",
        model=model,
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        tool_schemas=[],
        config=config or {"model": model},
    ))
    return events, fake


# ── unit: field extraction ───────────────────────────────────────────────

def test_reasoning_delta_prefers_reasoning_content():
    d = _FakeDelta(reasoning_content="rc", reasoning="r")
    assert _openai_reasoning_delta(d) == "rc"


def test_reasoning_delta_accepts_reasoning_field():
    d = _FakeDelta(reasoning="think step by step")
    assert _openai_reasoning_delta(d) == "think step by step"


def test_reasoning_delta_reads_model_extra():
    d = _FakeDelta(model_extra={"reasoning": "from extra"})
    assert _openai_reasoning_delta(d) == "from extra"


def test_reasoning_delta_nested_dict():
    d = _FakeDelta(reasoning={"content": "nested"})
    assert _openai_reasoning_delta(d) == "nested"


def test_reasoning_delta_empty_is_none():
    assert _openai_reasoning_delta(_FakeDelta(reasoning="")) is None
    assert _openai_reasoning_delta(_FakeDelta()) is None
    assert _openai_reasoning_delta(None) is None


# ── e2e: stream_openai_compat ────────────────────────────────────────────

def test_stream_yields_thinking_from_reasoning_content(monkeypatch):
    events, _ = _run_stream(monkeypatch, [
        _FakeChunk(_FakeDelta(reasoning_content="step 1 ")),
        _FakeChunk(_FakeDelta(reasoning_content="step 2")),
        _FakeChunk(_FakeDelta(content="answer")),
    ])
    thinking = [e for e in events if isinstance(e, ThinkingChunk)]
    texts = [e for e in events if isinstance(e, TextChunk)]
    turns = [e for e in events if isinstance(e, AssistantTurn)]

    assert "".join(t.text for t in thinking) == "step 1 step 2"
    assert "".join(t.text for t in texts) == "answer"
    assert turns[0].reasoning_content == "step 1 step 2"
    assert turns[0].text == "answer"


def test_stream_yields_thinking_from_reasoning_field(monkeypatch):
    """Newer vLLM / OpenAI guidance uses delta.reasoning, not reasoning_content."""
    events, _ = _run_stream(monkeypatch, [
        _FakeChunk(_FakeDelta(reasoning="internal CoT ")),
        _FakeChunk(_FakeDelta(reasoning="more CoT")),
        _FakeChunk(_FakeDelta(content="final")),
    ])
    thinking = [e for e in events if isinstance(e, ThinkingChunk)]
    turns = [e for e in events if isinstance(e, AssistantTurn)]

    assert "".join(t.text for t in thinking) == "internal CoT more CoT"
    assert turns[0].reasoning_content == "internal CoT more CoT"
    assert turns[0].text == "final"


def test_stream_thinking_only_no_content(monkeypatch):
    events, _ = _run_stream(monkeypatch, [
        _FakeChunk(_FakeDelta(reasoning="only thinking")),
    ])
    thinking = [e for e in events if isinstance(e, ThinkingChunk)]
    texts = [e for e in events if isinstance(e, TextChunk)]
    turns = [e for e in events if isinstance(e, AssistantTurn)]

    assert len(thinking) == 1
    assert texts == []
    assert turns[0].reasoning_content == "only thinking"
    assert turns[0].text == ""


def test_thinking_true_enables_compat_request_flags(monkeypatch):
    """Explicit /thinking ON should ask OpenAI-compat backends to emit CoT."""
    events, fake = _run_stream(
        monkeypatch,
        [_FakeChunk(_FakeDelta(content="ok"))],
        config={"model": "custom/qwen3", "thinking": True},
        model="custom/qwen3",
    )
    assert events  # stream completed
    kwargs = fake.chat.completions.last_kwargs
    body = kwargs.get("extra_body") or {}
    assert body.get("thinking") == {"type": "enabled"}
    assert body.get("enable_thinking") is True
    assert body.get("chat_template_kwargs", {}).get("enable_thinking") is True


def test_thinking_false_disables_without_enable_flags(monkeypatch):
    events, fake = _run_stream(
        monkeypatch,
        [_FakeChunk(_FakeDelta(content="ok"))],
        config={"model": "custom/qwen3", "thinking": False},
        model="custom/qwen3",
    )
    assert events
    body = (fake.chat.completions.last_kwargs.get("extra_body") or {})
    assert body.get("thinking") == {"type": "disabled"}
    assert "enable_thinking" not in body
