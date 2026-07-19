"""Tests for /config parsing, especially list-type keys (comma-split)."""
from __future__ import annotations

from cheetahclaws.commands.config_cmd import cmd_config


def test_disabled_tools_comma_split():
    cfg = {"model": "x"}
    cmd_config("disabled_tools=Bash,WebFetch", None, cfg)
    assert cfg["disabled_tools"] == ["Bash", "WebFetch"]


def test_disabled_tools_comma_split_with_spaces():
    cfg = {"model": "x"}
    cmd_config("disabled_tools= bash , WebFetch ", None, cfg)
    assert cfg["disabled_tools"] == ["bash", "WebFetch"]


def test_disabled_tools_single_value():
    cfg = {"model": "x"}
    cmd_config("disabled_tools=Glob", None, cfg)
    assert cfg["disabled_tools"] == ["Glob"]


def test_disabled_tools_empty_value_clears():
    cfg = {"model": "x", "disabled_tools": ["Bash"]}
    cmd_config("disabled_tools=", None, cfg)
    assert cfg["disabled_tools"] == []


def test_disabled_tools_json_array_still_works():
    cfg = {"model": "x"}
    cmd_config('disabled_tools=["Bash","WebFetch"]', None, cfg)
    assert cfg["disabled_tools"] == ["Bash", "WebFetch"]


def test_non_list_key_not_split():
    cfg = {"model": "x"}
    cmd_config("notes=a,b,c", None, cfg)
    # 'notes' is not a list-type key → stays a plain string
    assert cfg["notes"] == "a,b,c"


def test_whitelist_comma_split():
    cfg = {"model": "x"}
    cmd_config("wechat_smart_reply_whitelist=alice,bob", None, cfg)
    assert cfg["wechat_smart_reply_whitelist"] == ["alice", "bob"]
