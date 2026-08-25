"""Tests for the Read-tool overflow redirect — defense-in-depth that
catches the case where the model ignores the template's "use
SummarizeLargeFile" instruction and calls Read/ReadPDF on a too-big file
anyway. The Read response itself routes the model to SummarizeLargeFile,
so the raw content never overflows the next API call."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cheetahclaws.tools.files import (
    _is_cjk_heavy,
    _maybe_redirect_to_summarize,
)


# ── CJK-heavy detection ──────────────────────────────────────────────────


def test_is_cjk_heavy_pure_english_false():
    assert _is_cjk_heavy("hello world this is plain english") is False


def test_is_cjk_heavy_pure_chinese_true():
    assert _is_cjk_heavy("中文内容测试一下分词的情况" * 10) is True


def test_is_cjk_heavy_pure_japanese_true():
    assert _is_cjk_heavy("こんにちは世界これは日本語のテキストです" * 10) is True


def test_is_cjk_heavy_mixed_minority_cjk_false():
    """A document with <20% CJK characters should NOT be flagged."""
    text = "Mostly English text with a few 中文 characters here and there." * 20
    assert _is_cjk_heavy(text) is False


def test_is_cjk_heavy_empty():
    assert _is_cjk_heavy("") is False


# ── Redirect threshold logic ─────────────────────────────────────────────


def test_no_redirect_for_small_files():
    """A small file → returns None (caller returns original content)."""
    assert _maybe_redirect_to_summarize(
        "small file", "/tmp/x.txt", {"model": "custom/qwen2.5-72b"},
    ) is None


def test_no_redirect_for_empty_text():
    assert _maybe_redirect_to_summarize(
        "", "/tmp/x.txt", {"model": "claude-opus-4-7"},
    ) is None


def test_redirect_fires_on_users_actual_failure_case():
    """Reproduce the user's exact scenario: ~25K-token PDF on
    custom/qwen2.5-72b. The custom provider's declared ctx is 128K but
    the actual model is 32K — the safe_ctx cap at 30K must catch this."""
    # ~25K tokens of English-ish text (roughly 70K chars at 2.8 chars/token)
    big_text = "Sample paragraph with citations [Smith 2024]. " * 1500
    redirect = _maybe_redirect_to_summarize(
        big_text, "/home/user/autodan.pdf", {"model": "custom/qwen2.5-72b"},
    )
    assert redirect is not None
    assert "ReadTooLarge" in redirect
    assert "SummarizeLargeFile" in redirect
    assert "/home/user/autodan.pdf" in redirect
    # Redirect message must be MUCH smaller than the input it replaces —
    # that's the whole point of the redirect.
    assert len(redirect) < len(big_text) / 10


def test_redirect_fires_on_cjk_at_lower_char_count():
    """CJK content tokenizes 1:1 with chars, so a 24K-char CJK file is
    24K tokens — over the 18,737-token ceiling for qwen2.5-72b's
    registry 32K context. The same chars in English would NOT trigger
    (~8.5K tokens at chars/2.8)."""
    cjk_text = "中文内容测试" * 4000   # 24K chars CJK
    redirect_cjk = _maybe_redirect_to_summarize(
        cjk_text, "/tmp/cn.txt", {"model": "custom/qwen2.5-72b"},
    )
    assert redirect_cjk is not None, "CJK content of this size must trigger redirect"

    # Same character count in English should NOT trigger
    eng_text = "abcdef" * 4000   # also 24K chars but English
    redirect_eng = _maybe_redirect_to_summarize(
        eng_text, "/tmp/en.txt", {"model": "custom/qwen2.5-72b"},
    )
    assert redirect_eng is None, (
        "Equivalent char-count in English should NOT redirect (chars/2.8 = ~8.5K tokens, fits)"
    )


def test_redirect_fires_on_very_large_file_custom_provider():
    """No artificial ceiling: the redirect trusts the declared context
    (custom/ defaults to 256K) and only fires when the file genuinely
    exceeds 70% of (declared - 6K) — here 175,000 tokens. A ~178K-token
    file must trigger; the redirect protects the tail of long sessions
    even on large-context models."""
    text = "x" * 500000   # ~178K tokens English, over the 175K ceiling
    redirect = _maybe_redirect_to_summarize(
        text, "/tmp/big.txt", {"model": "custom/some-model"},
    )
    assert redirect is not None, (
        "custom-provider redirect must fire for a file over the safe ceiling"
    )


def test_no_redirect_on_genuine_large_context_model_with_modest_file():
    """A ~30K-token file on claude-opus-4-7 (200K context) should NOT
    redirect — there's plenty of room. No artificial ceiling: the safe
    ceiling is 0.7*(200K-6K) = 135,800 tokens, so 30K fits with margin.
    (An earlier 30K cap redirected here too aggressively and contributed
    to the max_tokens under-cap that truncated replies mid-sentence.)"""
    text = "x" * 84000   # ~30K tokens
    redirect = _maybe_redirect_to_summarize(
        text, "/tmp/x.txt", {"model": "claude-opus-4-7"},
    )
    assert redirect is None


def test_redirect_message_includes_preview():
    """The redirect must include a preview chunk so the model has *some*
    context to decide on a focus parameter for SummarizeLargeFile."""
    text = ("UNIQUE_PREVIEW_CONTENT_MARKER " * 200) + ("X" * 100000)
    redirect = _maybe_redirect_to_summarize(
        text, "/tmp/x.txt", {"model": "custom/qwen2.5-72b"},
    )
    assert redirect is not None
    assert "PREVIEW" in redirect
    # The preview comes from the start of the file
    assert "UNIQUE_PREVIEW_CONTENT_MARKER" in redirect


# ── Integration: Read tool wrapper actually applies the redirect ────────


def test_read_tool_redirects_huge_text_file(tmp_path):
    """Write a fake 'huge' text file, call Read via the tool dispatcher,
    verify the result is the redirect message (not the raw content).
    CJK content: tokenizes 1:1 with chars, so the 50K-char capped read
    (~50K tokens) clears qwen2.5-72b's 18,737-token ceiling. (Pure
    English caps at ~17.8K tokens — just under — by design.)"""
    big = tmp_path / "big.txt"
    big.write_text("中文内容测试" * 12000, encoding="utf-8")  # 72K chars / 216KB

    # Call via the tool registry (simulates what agent.py does)
    from cheetahclaws.tools import execute_tool
    out = execute_tool(
        "Read",
        {"file_path": str(big)},
        permission_mode="accept-all",
        config={"model": "custom/qwen2.5-72b"},
    )
    # Defensive redirect must have fired
    assert "ReadTooLarge" in out
    assert "SummarizeLargeFile" in out
    assert str(big) in out


def test_read_tool_passes_through_small_file(tmp_path):
    """Small files — Read returns the actual content, not a redirect."""
    small = tmp_path / "small.txt"
    small.write_text("just a few lines\nof normal text\n", encoding="utf-8")

    from cheetahclaws.tools import execute_tool
    out = execute_tool(
        "Read",
        {"file_path": str(small)},
        permission_mode="accept-all",
        config={"model": "custom/qwen2.5-72b"},
    )
    assert "ReadTooLarge" not in out
    assert "just a few lines" in out


def test_standard_profile_redirects_large_read_to_an_available_follow_up(tmp_path):
    # CJK content: 1:1 char/token, so the 50K-char capped read (~50K
    # tokens) clears qwen2.5-72b's 18,737-token ceiling and triggers
    # the redirect (English content caps just under it).
    big = tmp_path / "big-cjk.txt"
    big.write_text("中文内容测试" * 8_000, encoding="utf-8")  # 48K chars / 144KB

    from cheetahclaws.tools import execute_tool
    out = execute_tool(
        "Read", {"file_path": str(big)}, permission_mode="accept-all",
        config={
            "model": "custom/qwen2.5-72b", "tool_profile": "standard",
            "_active_tool_names": frozenset({"Read"}),
        },
    )

    assert "ReadTooLarge" in out
    assert "SummarizeLargeFile" not in out
    assert "narrower `offset` and `limit`" in out
