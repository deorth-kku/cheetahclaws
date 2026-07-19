"""Unit tests for the disabled_tools -> system prompt prose filtering.

Mirrors tool_registry.get_tool_schemas: a disabled tool must be stripped
from the *text* of the system prompt (the ``# Available Tools`` bullet list
and the ``# Multi-Agent Guidelines`` prose section), not just the schemas.
Small models follow the prose list even when the schema is withheld, so
leaving it in made disabled Agent/SendMessage "still available".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cheetahclaws.prompts import pick_base_prompt
from cheetahclaws.context import _apply_disabled_tools
from cheetahclaws.tool_registry import _normalize_disabled


def _default() -> str:
    return pick_base_prompt("kimi/moonshot-v1-128k", "")


def test_no_disabled_is_noop():
    p = _default()
    # build_system_prompt only calls the filter when disabled is truthy, and
    # the filter is a no-op for an empty set, so the output is unchanged.
    assert _apply_disabled_tools(p, set()) == p
    # nothing removed when disabled_tools empty
    assert "**Agent**" in _apply_disabled_tools(p, _normalize_disabled([]))


def test_all_subagent_tools_stripped():
    disabled = _normalize_disabled(
        ["Agent", "SendMessage", "CheckAgentResult", "ListAgentTasks", "ListAgentTypes"]
    )
    out = _apply_disabled_tools(_default(), disabled)
    for name in ("Agent", "SendMessage", "CheckAgentResult", "ListAgentTasks", "ListAgentTypes"):
        assert f"**{name}**" not in out
    # the now-empty ## Multi-Agent header is dropped too
    assert "## Multi-Agent" not in out
    # the prose section that only makes sense with sub-agents is removed
    assert "Multi-Agent Guidelines" not in out


def test_partial_disable_keeps_others():
    disabled = _normalize_disabled(["SendMessage", "ListAgentTasks"])
    out = _apply_disabled_tools(_default(), disabled)
    # disabled tokens gone
    assert "**SendMessage**" not in out
    assert "**ListAgentTasks**" not in out
    # siblings kept on the same line
    assert "**Agent**" in out
    assert "**CheckAgentResult**" in out
    assert "**ListAgentTypes**" in out
    # Guidelines section preserved because core Agent is still enabled
    assert "Multi-Agent Guidelines" in out


def test_prose_bold_not_mistaken_for_tool():
    # "**Workflow:**" is a bold lead-in in the Task Management section, not a
    # tool name, and must survive even if the word "workflow" were disabled.
    disabled = _normalize_disabled(["workflow"])
    out = _apply_disabled_tools(_default(), disabled)
    assert "**Workflow:**" in out
