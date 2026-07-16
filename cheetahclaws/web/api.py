"""Structured chat API for CheetahClaws web UI.

Bridges the synchronous agent.run() generator to WebSocket event streaming,
following the same pattern as the Telegram/Slack/WeChat bridges:
wire RuntimeContext callbacks → run agent on background thread → push events.
"""
from __future__ import annotations

import copy
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure the package root is importable (web/ is a subpackage)
_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


# ── Event envelope ─────────────────────────────────────────────────────────

@dataclass
class ChatEvent:
    """JSON-serializable event sent to browser via WebSocket."""
    type: str       # text_chunk | thinking_chunk | tool_start | tool_end |
                    # permission_request | permission_response | turn_done |
                    # error | status
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "data": self.data, "ts": self.ts})


# ── Slash command handler (can't import from cheetahclaws.py — it has
#    top-level code that runs on import).  Build our own from commands/*.
# ──────────────────────────────────────────────────────────────────────────

_WEB_COMMANDS: dict | None = None


def _get_web_commands() -> dict:
    """Lazily build the slash command registry from commands/ submodules."""
    global _WEB_COMMANDS
    if _WEB_COMMANDS is not None:
        return _WEB_COMMANDS

    cmds: dict = {}
    # Import each group separately so partial failures don't block others
    _imports = [
        # (module, [(cmd_name, func_name), ...])
        ("commands.core", [
            ("help", "cmd_help"), ("clear", "cmd_clear"),
            ("context", "cmd_context"), ("cost", "cmd_cost"),
            ("compact", "cmd_compact"), ("status", "cmd_status"),
            ("export", "cmd_export"), ("copy", "cmd_copy"),
            ("doctor", "cmd_doctor"), ("init", "cmd_init"),
            ("proactive", "cmd_proactive"), ("image", "cmd_image"),
            ("img", "cmd_image"),
        ]),
        ("commands.session", [
            ("save", "cmd_save"), ("load", "cmd_load"),
            ("resume", "cmd_resume"), ("search", "cmd_search"),
            ("history", "cmd_history"), ("cloudsave", "cmd_cloudsave"),
            ("exit", "cmd_exit"), ("quit", "cmd_exit"),
        ]),
        ("commands.config_cmd", [
            ("model", "cmd_model"), ("config", "cmd_config"),
            ("verbose", "cmd_verbose"), ("thinking", "cmd_thinking"),
            ("permissions", "cmd_permissions"), ("cwd", "cmd_cwd"),
        ]),
        ("commands.advanced", [
            ("brainstorm", "cmd_brainstorm"), ("worker", "cmd_worker"),
            ("ssj", "cmd_ssj"), ("skills", "cmd_skills"),
            ("memory", "cmd_memory"), ("agents", "cmd_agents"),
            ("mcp", "cmd_mcp"), ("plugin", "cmd_plugin"),
            ("tasks", "cmd_tasks"), ("task", "cmd_tasks"),
        ]),
        ("commands.checkpoint_plan", [
            ("plan", "cmd_plan"), ("checkpoint", "cmd_checkpoint"),
        ]),
        ("commands.agent_cmd", [
            ("agent", "cmd_agent"),
        ]),
        ("commands.monitor_cmd", [
            ("subscribe", "cmd_subscribe"),
            ("subscriptions", "cmd_subscriptions"),
            ("subs", "cmd_subscriptions"),
            ("unsubscribe", "cmd_unsubscribe"),
            ("monitor", "cmd_monitor"),
        ]),
        # External bridges — telegram / slack / wechat / voice.
        # Each lives in its own module so missing deps (e.g. sounddevice for
        # voice) just skip that one command instead of blocking the rest.
        ("bridges.telegram", [("telegram", "cmd_telegram")]),
        ("bridges.slack",    [("slack",    "cmd_slack")]),
        ("bridges.wechat",   [("wechat",   "cmd_wechat"),
                              ("weixin",   "cmd_wechat")]),
        ("modular.voice.cmd", [("voice",   "cmd_voice")]),
    ]
    import importlib
    for mod_name, pairs in _imports:
        try:
            mod = importlib.import_module(mod_name)
            for cmd_name, func_name in pairs:
                fn = getattr(mod, func_name, None)
                if fn:
                    cmds[cmd_name] = fn
        except ImportError:
            pass
    _WEB_COMMANDS = cmds
    return cmds


def _web_handle_slash(line: str, state, config):
    """Handle /command. Returns True if handled, or sentinel tuple."""
    if not line.startswith("/"):
        return False
    parts = line[1:].split(None, 1)
    if not parts:
        return False
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    commands = _get_web_commands()
    handler = commands.get(cmd)
    if handler:
        result = handler(args, state, config)
        # Sentinel tuples need special handling by the caller
        _SENTINELS = ("__voice__", "__image__", "__brainstorm__", "__worker__",
                      "__ssj_cmd__", "__ssj_query__", "__ssj_debate__",
                      "__ssj_passthrough__", "__ssj_promote_worker__", "__plan__")
        if isinstance(result, tuple) and result[0] in _SENTINELS:
            return result
        return True
    print(f"Unknown command: /{cmd}  (type /help for commands)")
    return True


# ── Chat Session ───────────────────────────────────────────────────────────

_IDLE_TIMEOUT = 1800  # 30 min before session is considered stale

_SAFE_CONFIG_KEYS = frozenset({
    "model", "permission_mode", "max_tokens", "verbose", "thinking",
    "thinking_budget", "max_tool_output", "max_agent_depth",
    "shell_policy", "log_level",
})

_WRITABLE_CONFIG_KEYS = frozenset({
    "model", "permission_mode", "verbose", "thinking",
    "thinking_budget", "max_tokens",
    # API keys — written to session config only, not persisted to disk
    "anthropic_api_key", "openai_api_key", "gemini_api_key",
    "kimi_api_key", "qwen_api_key", "zhipu_api_key",
    "deepseek_api_key", "minimax_api_key", "custom_api_key",
    "custom_base_url", "ollama_base_url",
})

# Keys for which the server config file is the source of truth and the DB
# holds only *explicit per-session overrides* (deltas).  API-key style keys
# are intentionally excluded — they may legitimately be written to a session
# with the same value as the file (e.g. when the user re-enters a key), and
# we never want to silently ignore such an update.
_CONFIG_OVERRIDE_KEYS = frozenset({
    "model", "permission_mode", "verbose", "thinking",
    "thinking_budget", "max_tokens", "max_tool_output",
    "max_agent_depth", "shell_policy", "log_level",
})


def _is_real_override(key: str, value, live_cfg: dict, defaults: dict) -> bool:
    """Return True only if ``value`` is a genuine per-session override.

    A stored value is NOT a real override (so the session should follow the
    server config file) when it equals either the live config file value or
    the original DEFAULTS.  This discards creation-time snapshots: legacy
    sessions were snapshotted with default-looking values, not explicit user
    choices.  It also handles the tri-state ``thinking`` key, whose default
    changed from ``False`` (older versions) to ``None`` — a stored ``False``
    is treated as equivalent to the file's ``None``/falsy value, so stale
    "off" defaults don't pin a session to a now-changed file.
    """
    if value == live_cfg.get(key) or value == defaults.get(key):
        return False
    if value is False and not live_cfg.get(key):
        return False  # legacy "off" default vs current None/false → follow file
    return True

# Keys that contain secrets — never expose in GET responses
_SECRET_KEYS = frozenset({
    "anthropic_api_key", "openai_api_key", "gemini_api_key",
    "kimi_api_key", "qwen_api_key", "zhipu_api_key",
    "deepseek_api_key", "minimax_api_key", "custom_api_key",
})

_API_KEY_CONFIG_MAP = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "gemini_api_key",
    "kimi": "kimi_api_key",
    "qwen": "qwen_api_key",
    "zhipu": "zhipu_api_key",
    "deepseek": "deepseek_api_key",
    "minimax": "minimax_api_key",
    "custom": "custom_api_key",
}


class ChatSession:
    """One agent conversation, bridged to WebSocket clients.

    Persistence: session metadata and message history are mirrored to SQLite
    via web.db.repo. The in-memory `messages` list is a write-through cache
    for fast replay; event queue/buffer stay in-memory only.
    """

    def __init__(self, base_config: dict, user_id: int, *,
                 session_id: Optional[str] = None,
                 title: Optional[str] = None):
        from cheetahclaws.web import db as _db
        _db.init_db()

        # Hydrate-from-DB path vs new-session path
        existing = (_db.repo.get_session(session_id, user_id)
                    if session_id else None)

        self.session_id: str = (existing["id"] if existing
                                else (session_id or uuid.uuid4().hex[:12]))
        self.user_id: int = user_id
        self.title: str = (existing["title"] if existing
                           else (title or "New chat"))
        self.created_at: float = (existing["created_at"] if existing
                                  else time.time())
        self.last_active: float = time.time()

        # Deep-copy config so permission_mode changes don't leak.  The server
        # config FILE (base_config) is the source of truth — DB rows only hold
        # *explicit per-session overrides* (deltas), so changes to the config
        # file are reflected on every reload.
        #
        # A stored value is honored as a real per-session override only when it
        # differs from BOTH the live config file AND the original DEFAULTS in
        # config.py.  Anything matching either is a creation-time default (the
        # session was snapshotted at creation, not explicitly configured), so
        # the session follows the file instead of being pinned to a stale
        # copy.  This also transparently migrates legacy full-snapshot rows:
        # the rewrite below drops the now-ignored keys, leaving a clean delta.
        # Keys not in _CONFIG_OVERRIDE_KEYS (e.g. API-key style keys) are
        # ignored on load so a stale snapshot can never clobber the live value.
        from cheetahclaws.config import DEFAULTS
        base = copy.deepcopy(base_config)
        self._loaded_overrides = {}
        if existing and existing.get("config"):
            stored = existing["config"]
            # Legacy rows persisted a FULL snapshot at creation.  Such a
            # snapshot always contains at least one key whose value matches
            # either the current config file or the original DEFAULTS (e.g.
            # max_tokens → 16000), whereas a genuine per-session override set
            # only contains the few keys the user actually changed — and those
            # differ from the file.  Detect that signature and discard the
            # whole legacy snapshot, so the session follows the server config
            # file instead of being pinned to a stale creation-time copy.
            is_legacy = any(
                k in _CONFIG_OVERRIDE_KEYS
                and (v == base_config.get(k) or v == DEFAULTS.get(k))
                for k, v in stored.items())
            if is_legacy:
                try:
                    _db.repo.upsert_session(self.session_id, user_id, config={})
                except Exception:  # noqa: BLE001
                    pass
            else:
                for k, v in stored.items():
                    if k in _CONFIG_OVERRIDE_KEYS and _is_real_override(
                            k, v, base_config, DEFAULTS):
                        base[k] = v
                        self._loaded_overrides[k] = v
        self.config: dict = base
        self.config["_session_id"] = self.session_id

        # Event fan-out: multiple WS clients can subscribe
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()

        # Buffer recent events so late-joining subscribers don't miss them.
        # Capped at 500 events; covers the gap between agent start and WS connect.
        self._event_buffer: list[ChatEvent] = []
        self._EVENT_BUFFER_MAX = 500

        # Agent state (in-process, NOT a PTY subprocess)
        self._agent_state = None  # type: ignore[assignment]
        self._agent_thread: Optional[threading.Thread] = None
        self._busy = threading.Event()
        # Stop signal — set by request_stop() (Stop button / WS "stop" msg).
        self._cancelled = threading.Event()

        # Message history for UI replay on reconnect (hydrated from DB)
        self.messages: list[dict] = (_db.repo.get_messages(self.session_id)
                                     if existing else [])
        self._msg_lock = threading.Lock()

        # Persist (create-or-update) metadata.  Only the explicit per-session
        # override delta is stored (not the full config), so the server config
        # file remains the source of truth and later file changes propagate.
        _db.repo.upsert_session(
            self.session_id, user_id,
            title=self.title,
            config=dict(getattr(self, "_loaded_overrides", {})),
        )

        self._init_runtime()

    def _init_runtime(self):
        """Initialize RuntimeContext and AgentState."""
        from cheetahclaws.agent import AgentState
        from cheetahclaws import runtime

        self._agent_state = AgentState()
        # On reconnect to an existing session, the DB holds the conversation
        # (self.messages, hydrated in __init__) but the freshly-created
        # AgentState is empty.  The agent loop (run()) and /history both read
        # AgentState.messages, so without rehydrating it the agent "forgets"
        # everything after a server restart even though the UI shows the
        # history.  Rebuild the neutral message list from the persisted rows.
        if self.messages:
            try:
                self._agent_state.messages = self._messages_to_neutral(
                    self.messages)
            except Exception as exc:  # noqa: BLE001
                from cheetahclaws.web.logging_setup import get_logger
                get_logger("api").exception(
                    "agent history rehydrate failed",
                    extra={"session_id": self.session_id, "err": str(exc)})
        ctx = runtime.get_session_ctx(self.session_id)
        ctx.agent_state = self._agent_state
        ctx.run_query = lambda msg: self.submit_prompt(msg)
        # Let ask_input_interactive() (AskUserQuestion tool, etc.) push an
        # "ask_request" event to browser WS clients directly.
        ctx.web_broadcast = lambda event: self._broadcast(
            ChatEvent(event["type"], event.get("data", {})))

    @staticmethod
    def _messages_to_neutral(db_messages: list[dict]) -> list[dict]:
        """Convert persisted ChatSession messages (per-turn, block-based) into
        the fine-grained neutral format the agent loop expects.

        Persisted shape (one row per *turn*):
            {"role": "user",    "content": ...}
            {"role": "assistant", "content": ..., "blocks": [
                {type:"text", text}, {type:"tool", name, inputs, tool_id,
                 status, result}, {type:"ask", ...}]}
        Neutral shape (interleaved, one row per message) required by the LLM
        backends:
            {"role":"user", "content":...}
            {"role":"assistant","content":...,"tool_calls":[
                {"id","name","input","type":"function"}]}
            {"role":"tool","tool_call_id":...,"name":...,"content":...}
        """
        from cheetahclaws.compaction import sanitize_history

        neutral: list[dict] = []
        for m in db_messages:
            role = m.get("role")
            if role == "user":
                nm: dict = {"role": "user",
                            "content": m.get("content", "")}
                if m.get("images"):
                    nm["images"] = m["images"]
                neutral.append(nm)
                continue

            if role != "assistant":
                # Shouldn't happen at this layer, but keep it safe.
                neutral.append({"role": role,
                                "content": m.get("content", "")})
                continue

            text_parts: list[str] = []
            tool_calls: list[dict] = []
            tool_results: list[dict] = []

            blocks = m.get("blocks")
            if isinstance(blocks, list) and blocks:
                for b in blocks:
                    btype = b.get("type")
                    if btype == "text":
                        if b.get("text"):
                            text_parts.append(b["text"])
                    elif btype == "tool":
                        tid = (b.get("tool_id") or b.get("id")
                               or f"tool_{len(tool_calls)}")
                        tool_calls.append({
                            "id":   tid,
                            "name": b.get("name", ""),
                            "input": b.get("inputs") or {},
                            "type": "function",
                        })
                        # Only pair a tool_result for completed calls; dropped
                        # or unanswered calls are cleaned by sanitize_history.
                        if b.get("status") in ("done", "denied") \
                                and "result" in b:
                            tool_results.append({
                                "role":         "tool",
                                "tool_call_id": tid,
                                "name":         b.get("name", ""),
                                "content":      b.get("result") or "",
                            })
                    # "ask" blocks have no neutral representation needed to
                    # continue the conversation (the question was already
                    # answered); skip to keep tool_call pairing valid.
                if not text_parts and not tool_calls:
                    content = m.get("content", "")
                    if content:
                        text_parts.append(content)
            else:
                # Legacy rows: flat tool_calls list, no blocks.
                tcs = m.get("tool_calls")
                if isinstance(tcs, list) and tcs:
                    for tc in tcs:
                        tid = (tc.get("id") or tc.get("tool_id")
                               or f"tool_{len(tool_calls)}")
                        tool_calls.append({
                            "id":    tid,
                            "name":  tc.get("name", ""),
                            "input": tc.get("inputs") or tc.get("input")
                                     or {},
                            "type":  "function",
                        })
                        if tc.get("status") in ("done", "denied") \
                                and "result" in tc:
                            tool_results.append({
                                "role":         "tool",
                                "tool_call_id": tid,
                                "name":         tc.get("name", ""),
                                "content":      tc.get("result") or "",
                            })
                content = m.get("content", "")
                if content:
                    text_parts.append(content)

            msg: dict = {"role": "assistant",
                         "content": "".join(text_parts)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            neutral.append(msg)
            neutral.extend(tool_results)

        # Drop any orphan tool_results / unanswered tool_calls so the
        # reconstructed history is valid for the next API call.
        try:
            neutral = sanitize_history(neutral)
        except Exception:
            pass
        return neutral

    # ── Subscriber management ──────────────────────────────────────────

    def subscribe(self) -> queue.Queue:
        """Add a subscriber and replay any buffered events."""
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._sub_lock:
            # Replay buffered events so late-joiners don't miss anything
            for evt in self._event_buffer:
                try:
                    q.put_nowait(evt)
                except queue.Full:
                    break
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _clear_event_buffer(self):
        """Drop buffered events so a freshly-connecting WS client (e.g. after
        a page refresh) doesn't replay the previous turn's tool_start /
        tool_end / ask_request events — they'd duplicate the already-persisted
        history rendered from the DB."""
        with self._sub_lock:
            self._event_buffer.clear()

    def _broadcast(self, event: ChatEvent):
        with self._sub_lock:
            # Buffer for late-joining subscribers
            self._event_buffer.append(event)
            if len(self._event_buffer) > self._EVENT_BUFFER_MAX:
                self._event_buffer = self._event_buffer[-self._EVENT_BUFFER_MAX:]
            # Push to live subscribers
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    # ── Prompt submission ──────────────────────────────────────────────

    def handle_slash_sync(self, line: str) -> list[dict]:
        """Handle a slash command synchronously. Returns list of event dicts
        to send back in the HTTP response.

        Synchronous events are returned via HTTP only — re-broadcasting them
        to WS subscribers would duplicate every reply in the chat UI, since
        the same client also iterates the HTTP `events` payload. Background
        threads spawned by the handler still broadcast normally, because
        `_broadcast` is restored before they emit anything.
        """
        events: list[dict] = []
        orig_broadcast = self._broadcast

        def capture_broadcast(event: ChatEvent):
            events.append({"type": event.type, "data": event.data})

        self._broadcast = capture_broadcast  # type: ignore
        try:
            self._handle_slash(line)
        finally:
            self._broadcast = orig_broadcast  # type: ignore
        return events

    def handle_slash_stream(self, line: str, event_callback):
        """Handle a slash command, calling event_callback(dict) for each event.
        Blocks until the command (including long-running ones) completes.
        Used by the SSE streaming endpoint.

        Events are delivered via the SSE callback only — re-broadcasting them
        to WS subscribers would duplicate every reply in the chat UI, since
        the same client also calls _handleEvent on the SSE stream.
        """
        done_event = threading.Event()
        orig_broadcast = self._broadcast

        def stream_broadcast(event: ChatEvent):
            event_callback({"type": event.type, "data": event.data})
            if event.type == "status" and event.data.get("state") == "idle":
                done_event.set()

        self._broadcast = stream_broadcast  # type: ignore
        try:
            self._handle_slash(line)
            # For long-running commands, wait until the bg thread finishes
            if self._busy.is_set():
                done_event.wait(timeout=600)  # 10 min max
        finally:
            self._broadcast = orig_broadcast  # type: ignore

    def submit_prompt(self, prompt: str) -> bool:
        """Submit a prompt or slash command. Returns False if agent is busy."""
        # Handle slash commands locally (don't send to LLM)
        if prompt.startswith("/"):
            return self._handle_slash(prompt)

        if self._busy.is_set():
            self._broadcast(ChatEvent("error", {"message": "Agent is busy"}))
            return False

        self.last_active = time.monotonic()
        # Clear event buffer for fresh turn — don't replay stale events
        with self._sub_lock:
            self._event_buffer.clear()
        self._append_msg({"role": "user", "content": prompt})
        self._broadcast(ChatEvent("status", {"state": "running"}))

        def _run():
            self._busy.set()
            self._cancelled.clear()
            try:
                self._run_agent(prompt)
            except Exception as exc:
                self._broadcast(ChatEvent("error", {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }))
            finally:
                self._busy.clear()
                self._broadcast(ChatEvent("status", {"state": "idle"}))

        self._agent_thread = threading.Thread(target=_run, daemon=True)
        self._agent_thread.start()
        return True

    def _handle_slash(self, line: str) -> bool:
        """Handle /commands locally, capture stdout, broadcast as system message.

        Some commands return sentinel tuples that require follow-up agent runs
        (e.g. __brainstorm__, __worker__, __plan__, __ssj_cmd__).  These are
        executed on a background thread exactly like a regular prompt.
        """
        import io
        import re as _re
        self.last_active = time.monotonic()

        self._append_msg({"role": "user", "content": line})

        # Parse command and args
        cmd_parts = line[1:].split(None, 1)
        cmd_name = cmd_parts[0].lower() if cmd_parts else ""
        cmd_args = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

        # /ssj with no args → show interactive menu
        if cmd_name == "ssj" and not cmd_args:
            self._broadcast(ChatEvent("interactive_menu", {
                "command": line,
                "menu": "ssj",
                "items": [
                    {"key":"1",  "icon":"bulb",    "label":"Brainstorm",    "cmd":"/brainstorm"},
                    {"key":"2",  "icon":"clipboard","label":"Show TODO",     "cmd":"/ssj todo"},
                    {"key":"3",  "icon":"worker",   "label":"Worker",        "cmd":"/worker"},
                    {"key":"4",  "icon":"brain",    "label":"Debate File",   "cmd":"/ssj debate"},
                    {"key":"5",  "icon":"sparkle",  "label":"Propose Improvement","cmd":"/ssj propose"},
                    {"key":"6",  "icon":"search",   "label":"Review File",   "cmd":"/ssj review"},
                    {"key":"7",  "icon":"book",     "label":"Generate README","cmd":"/ssj readme"},
                    {"key":"8",  "icon":"chat",     "label":"AI Commit Msg", "cmd":"/ssj commit"},
                    {"key":"9",  "icon":"test",     "label":"Scan Git Diff", "cmd":"/ssj scan"},
                    {"key":"10", "icon":"note",     "label":"Promote to Tasks","cmd":"/ssj promote"},
                    {"key":"13", "icon":"monitor",  "label":"Monitor",       "cmd":"/monitor"},
                    {"key":"15", "icon":"robot",    "label":"Autonomous Agent","cmd":"/agent"},
                ],
            }))
            return True

        # /ssj <subcommand> → map to direct actions (skip the interactive menu)
        _SSJ_DIRECT = {
            "debate":  ("__ssj_query__", "Act as a panel of 3 expert engineers. Each gives 2-3 critical insights on the codebase. Be specific and constructive."),
            "propose": ("__ssj_query__", "Analyze the codebase and propose 3 high-impact improvements with code examples. Focus on correctness, performance, or maintainability."),
            "review":  ("__ssj_query__", "Give a quick code review: identify bugs, code smells, or missing edge cases. Be concise."),
            "readme":  ("__ssj_query__", "Generate a comprehensive README.md for this project. Include: project description, features, installation, usage examples, and contributing guidelines."),
            "commit":  ("__ssj_query__", "Review the git diff (git diff HEAD) and suggest a concise, descriptive commit message following conventional commits format. Also list files changed."),
            "scan":    ("__ssj_query__", "Run git diff HEAD and analyze the changes. Summarize what was changed, why it might have been changed, and flag any potential issues or regressions."),
            "todo":    None,  # handled below
        }
        if cmd_name == "ssj" and cmd_args.split()[0].lower() in _SSJ_DIRECT:
            sub = cmd_args.split()[0].lower()
            extra_args = cmd_args[len(sub):].strip()
            if sub == "todo":
                pass  # fall through to normal handler
            else:
                action = _SSJ_DIRECT[sub]
                prompt = action[1]
                if extra_args:
                    prompt += f" Focus on: {extra_args}"
                # Run as agent query
                self._broadcast(ChatEvent("status", {"state": "running"}))
                self._broadcast(ChatEvent("command_result", {
                    "command": line,
                    "output": f"Running SSJ {sub}...",
                }))

                def _run_ssj():
                    self._busy.set()
                    from cheetahclaws import runtime
                    ctx = runtime.get_session_ctx(self.session_id)
                    ctx.in_web_turn = True
                    try:
                        self._run_agent(prompt)
                    except Exception as exc:
                        self._broadcast(ChatEvent("error",
                                                  {"message": str(exc)}))
                    finally:
                        ctx.in_web_turn = False
                        self._busy.clear()
                        self._broadcast(ChatEvent("status",
                                                  {"state": "idle"}))

                self._agent_thread = threading.Thread(target=_run_ssj,
                                                      daemon=True)
                self._agent_thread.start()
                return True

        # /brainstorm with no topic → ask for topic via input_request event
        if cmd_name == "brainstorm" and not cmd_args:
            self._broadcast(ChatEvent("input_request", {
                "command": "/brainstorm",
                "prompt": "Brainstorm topic (Enter for general):",
                "placeholder": "e.g. improve test coverage, refactor auth...",
                "default_cmd": "/brainstorm general project improvement",
            }))
            return True

        # Long-running commands — run on background thread with live events.
        # These call providers.stream() internally and take minutes.
        # We redirect stdout so their print() output streams to the browser.
        _LONG_RUNNING = {"brainstorm", "worker", "agent", "plan"}
        if cmd_name in _LONG_RUNNING:
            self._broadcast(ChatEvent("status", {"state": "running"}))
            session_ref = self  # capture for closure

            # Thread-local stdout wrapper: intercepts print() calls from
            # the command handler and broadcasts them as text_chunk events.
            # Uses threading.current_thread() check to avoid affecting other threads.
            _target_thread_id = [None]  # set inside the thread

            class _ThreadLocalStdout:
                """Only intercepts writes from the target thread."""
                def __init__(self, broadcast_fn, real):
                    self._broadcast = broadcast_fn
                    self._real = real
                def write(self, s):
                    if not s:
                        return
                    if threading.current_thread().ident == _target_thread_id[0]:
                        import re as _re2
                        clean = _re2.sub(r'\x1b\[[0-9;]*m', '', s)
                        if clean.strip():
                            self._broadcast(ChatEvent("text_chunk",
                                                      {"text": clean}))
                    else:
                        self._real.write(s)
                def flush(self):
                    self._real.flush()
                # Forward attributes to real stdout for compatibility
                def fileno(self):
                    return self._real.fileno()
                @property
                def encoding(self):
                    return getattr(self._real, 'encoding', 'utf-8')

            def _run_long():
                _target_thread_id[0] = threading.current_thread().ident
                self._busy.set()
                self._cancelled.clear()
                from cheetahclaws import runtime
                ctx = runtime.get_session_ctx(self.session_id)
                ctx.in_web_turn = True
                wrapper = _ThreadLocalStdout(session_ref._broadcast,
                                            sys.stdout)
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = wrapper
                sys.stderr = wrapper
                try:
                    result = _web_handle_slash(line, self._agent_state,
                                              self.config)
                    if isinstance(result, tuple):
                        self._process_sentinel(result)
                    elif result is True:
                        self._broadcast(ChatEvent("command_result", {
                            "command": line,
                            "output": "(done)",
                        }))
                except Exception as exc:
                    self._broadcast(ChatEvent("error",
                                              {"message": str(exc)}))
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err
                    ctx.in_web_turn = False
                    self._busy.clear()
                    self._broadcast(ChatEvent("status", {"state": "idle"}))

            self._agent_thread = threading.Thread(target=_run_long,
                                                  daemon=True)
            self._agent_thread.start()
            return True

        # Quick commands — capture stdout synchronously
        capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = capture
            sys.stderr = capture
            result = _web_handle_slash(line, self._agent_state, self.config)
        except Exception as exc:
            self._broadcast(ChatEvent("error", {"message": str(exc)}))
            return True
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        output = capture.getvalue().strip()
        output = _re.sub(r'\x1b\[[0-9;]*m', '', output)

        if output:
            self._append_msg({"role": "assistant", "content": output})
            self._broadcast(ChatEvent("command_result", {
                "command": line, "output": output,
            }))

        # Handle sentinel tuples from quick commands (unlikely but safe)
        if isinstance(result, tuple):
            self._broadcast(ChatEvent("status", {"state": "running"}))

            def _run_sentinel():
                self._busy.set()
                try:
                    self._process_sentinel(result)
                except Exception as exc:
                    self._broadcast(ChatEvent("error",
                                              {"message": str(exc)}))
                finally:
                    self._busy.clear()
                    self._broadcast(ChatEvent("status", {"state": "idle"}))

            self._agent_thread = threading.Thread(target=_run_sentinel,
                                                  daemon=True)
            self._agent_thread.start()
            return True

        if not output and result is True:
            self._broadcast(ChatEvent("command_result", {
                "command": line, "output": "(done)",
            }))

        return True

    def _process_sentinel(self, result: tuple):
        """Execute the multi-step workflow described by a sentinel tuple."""
        sentinel = result[0]

        if sentinel == "__brainstorm__":
            _, brain_prompt, brain_out_file = result
            self._broadcast(ChatEvent("command_result", {
                "command": "/brainstorm",
                "output": "Starting multi-persona brainstorm...",
            }))
            self._run_agent(brain_prompt)
            # Generate todo list from synthesis
            from pathlib import Path
            todo_path = str(Path(brain_out_file).parent / "todo_list.txt")
            self._run_agent(
                f"Based on the Master Plan you just synthesized, generate a "
                f"todo list file at {todo_path}. Format: one task per line, "
                f"each starting with '- [ ] '. Order by priority. Include ALL "
                f"actionable items from the plan. Use the Write tool to create "
                f"the file. Do NOT explain, just write the file now."
            )

        elif sentinel == "__worker__":
            _, worker_tasks = result
            total = len(worker_tasks)
            for i, (line_idx, task_text, prompt) in enumerate(worker_tasks):
                self._broadcast(ChatEvent("command_result", {
                    "command": f"/worker ({i+1}/{total})",
                    "output": task_text,
                }))
                self._run_agent(prompt)

        elif sentinel == "__plan__":
            _, plan_desc = result
            self._broadcast(ChatEvent("command_result", {
                "command": "/plan",
                "output": f"Entering plan mode: {plan_desc}",
            }))
            self._run_agent(
                f"Please analyze the codebase and create a detailed "
                f"implementation plan for: {plan_desc}"
            )

        elif sentinel == "__ssj_cmd__":
            # SSJ delegates to another slash command
            _, cmd_name, cmd_args = result
            inner_line = f"/{cmd_name} {cmd_args}".strip()
            self._broadcast(ChatEvent("command_result", {
                "command": "/ssj",
                "output": f"Executing: {inner_line}",
            }))
            # Re-enter slash handling for the delegated command
            self._handle_slash_inner(inner_line)

        elif sentinel in ("__ssj_query__", "__ssj_debate__",
                          "__ssj_passthrough__", "__ssj_promote_worker__"):
            # These carry a prompt to run through the agent
            prompt = result[1] if len(result) > 1 else ""
            if prompt:
                self._run_agent(prompt)

        elif sentinel == "__image__":
            self._broadcast(ChatEvent("command_result", {
                "command": "/image",
                "output": "Image/vision: paste an image URL or use the terminal for clipboard support.",
            }))

        elif sentinel == "__voice__":
            self._broadcast(ChatEvent("command_result", {
                "command": "/voice",
                "output": "Voice input requires the terminal (microphone access).",
            }))

        else:
            # Unknown sentinel — try to extract a prompt if it has one
            if len(result) > 1 and isinstance(result[1], str) and result[1]:
                self._run_agent(result[1])
            else:
                self._broadcast(ChatEvent("command_result", {
                    "command": str(result[0]),
                    "output": "This feature may require the terminal for full support.",
                }))

    def _handle_slash_inner(self, line: str):
        """Re-entrant slash handling for SSJ delegation."""
        import io
        import re as _re

        capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = capture
            sys.stderr = capture
            result = _web_handle_slash(line, self._agent_state, self.config)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        output = _re.sub(r'\x1b\[[0-9;]*m', '', capture.getvalue().strip())
        if output:
            self._append_msg({"role": "assistant", "content": output})
            self._broadcast(ChatEvent("command_result", {
                "command": line, "output": output,
            }))

        if isinstance(result, tuple):
            self._process_sentinel(result)

    def _run_agent(self, prompt: str):
        """Iterate agent.run() generator, broadcast events."""
        from cheetahclaws.agent import (run, TextChunk, ThinkingChunk, ToolStart,
                           ToolEnd, TurnDone, PermissionRequest)
        from cheetahclaws.context import build_system_prompt
        from cheetahclaws import runtime

        ctx = runtime.get_session_ctx(self.session_id)
        ctx.in_web_turn = True
        system_prompt = build_system_prompt(self.config)

        text_chunks: list[str] = []
        tool_calls: list[dict] = []
        # Ordered blocks keep interleaving intact across refreshes:
        #   {type:"text", text} | {type:"tool", ...} | {type:"ask", ...}
        blocks: list[dict] = []
        _cur_block = None  # current text block being accumulated

        # Do NOT wire RuntimeContext callbacks — we broadcast from the
        # generator loop below.  Wiring ctx.on_text_chunk etc. would cause
        # duplicate events because the REPL's run_query() also fires them
        # for every yielded event.  Single event source = generator loop only.
        ctx.on_text_chunk = None
        ctx.on_tool_start = None
        ctx.on_tool_end = None

        # Wrap web_broadcast so AskUserQuestion prompts also become ordered
        # blocks (at the right interleaved position) for persistence.
        _orig_web_broadcast = ctx.web_broadcast
        def _web_broadcast_blocker(event: dict):
            if event.get("type") == "ask_request":
                blocks.append({
                    "type": "ask",
                    "prompt": event.get("data", {}).get("prompt", ""),
                    "options": event.get("data", {}).get("options"),
                    "allow_freetext": event.get("data", {}).get("allow_freetext", True),
                })
            if _orig_web_broadcast is not None:
                _orig_web_broadcast(event)
        ctx.web_broadcast = _web_broadcast_blocker

        try:
            for event in run(prompt, self._agent_state, self.config,
                             system_prompt, cancel_check=self._cancelled.is_set):
                if isinstance(event, TextChunk):
                    text_chunks.append(event.text)
                    self._broadcast(ChatEvent("text_chunk",
                                              {"text": event.text}))
                    # Accumulate into the current text block so contiguous
                    # text stays merged but tool calls split it into separate
                    # blocks (preserving interleaving on reload).
                    if _cur_block is None or _cur_block["type"] != "text":
                        _cur_block = {"type": "text", "text": ""}
                        blocks.append(_cur_block)
                    _cur_block["text"] += event.text

                elif isinstance(event, ThinkingChunk):
                    self._broadcast(ChatEvent("thinking_chunk",
                                              {"text": event.text}))

                elif isinstance(event, ToolStart):
                    tc_block = {
                        "type": "tool",
                        "name": event.name,
                        "inputs": event.inputs,
                        "status": "running",
                        "tool_id": event.tool_id,
                        "result": "",
                    }
                    blocks.append(tc_block)
                    _cur_block = tc_block  # next text starts a new block
                    tool_calls.append(tc_block)
                    self._broadcast(ChatEvent("tool_start", {
                        "name": event.name,
                        "inputs": event.inputs,
                        "tool_id": event.tool_id,
                    }))

                elif isinstance(event, PermissionRequest):
                    self._broadcast(ChatEvent("permission_request", {
                        "description": event.description,
                    }))
                    # Block until browser responds
                    evt = threading.Event()
                    ctx.web_input_event = evt
                    try:
                        if evt.wait(timeout=300):
                            val = ctx.web_input_value.strip().lower()
                            event.granted = val in ("y", "yes", "true", "1")
                        else:
                            event.granted = False
                            self._broadcast(ChatEvent("error", {
                                "message": "Permission request timed out (5 min)",
                            }))
                    finally:
                        # Always clean up — prevents dangling event objects
                        ctx.web_input_event = None
                        ctx.web_input_value = ""
                    self._broadcast(ChatEvent("permission_response", {
                        "granted": event.granted,
                    }))

                elif isinstance(event, ToolEnd):
                    for tc in reversed(tool_calls):
                        if tc.get("type") != "tool" or tc["status"] != "running":
                            continue
                        # Prefer exact per-call id; fall back to name so old
                        # events without an id still match (e.g. deduped entries).
                        if event.tool_id:
                            if tc.get("tool_id") != event.tool_id:
                                continue
                        else:
                            if tc["name"] != event.name:
                                continue
                        tc["status"] = "done" if event.permitted else "denied"
                        tc["result"] = event.result[:2000] if event.result else ""
                        break
                    self._broadcast(ChatEvent("tool_end", {
                        "name": event.name,
                        "result": event.result[:2000] if event.result else "",
                        "permitted": event.permitted,
                        "tool_id": event.tool_id,
                    }))

                elif isinstance(event, TurnDone):
                    self._broadcast(ChatEvent("turn_done", {
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                    }))

            # Store assistant response in history
            final_text = "".join(text_chunks)
            msg: dict = {"role": "assistant", "content": final_text}
            # Persist ordered blocks (text/tool/ask) so interleaving survives
            # a refresh. Drop leading/trailing empty text blocks for cleanliness.
            _clean_blocks = [b for b in blocks if not (
                b.get("type") == "text" and not (b.get("text") or "").strip()
            )]
            if _clean_blocks:
                msg["blocks"] = _clean_blocks
            elif tool_calls:
                msg["tool_calls"] = tool_calls
            self._append_msg(msg, blocks=_clean_blocks or None)

        except Exception as exc:
            self._broadcast(ChatEvent("error", {"message": str(exc)}))
        finally:
            ctx.on_text_chunk = None
            ctx.on_tool_start = None
            ctx.on_tool_end = None
            ctx.web_broadcast = _orig_web_broadcast
            ctx.in_web_turn = False
            ctx.web_ask_event = None
            ctx.web_ask_value = ""
            # Clear the replay buffer so a page refresh / session switch
            # reconnecting via WS doesn't duplicate the just-finished turn's
            # tool and ask cards (history is already loaded from the DB).
            self._clear_event_buffer()

    # ── Permission approval ────────────────────────────────────────────

    def approve_permission(self, granted: bool):
        """Respond to a pending PermissionRequest."""
        from cheetahclaws import runtime
        ctx = runtime.get_session_ctx(self.session_id)
        evt = ctx.web_input_event
        if evt:
            ctx.web_input_value = "y" if granted else "n"
            evt.set()

    def respond_to_ask(self, value: str):
        """Respond to a pending AskUserQuestion / ask_input_interactive call.

        `value` is the user's answer — either the chosen option's label (or
        the literal text the agent expects) or free-form text.  Mirrors
        approve_permission() but for the AskUserQuestion tool path.
        """
        from cheetahclaws import runtime
        ctx = runtime.get_session_ctx(self.session_id)
        evt = ctx.web_ask_event
        if evt:
            ctx.web_ask_value = value or ""
            evt.set()

    def request_stop(self):
        """Raise the cancel flag so the running agent loop aborts.

        The `run()` generator checks cancel_check() at the top of each turn,
        so the agent stops after the current turn finishes (mirrors CLI
        Ctrl+C semantics: in-flight tool calls complete, the turn ends).
        """
        self._cancelled.set()

    # ── Introspection ──────────────────────────────────────────────────

    def _append_msg(self, msg: dict, blocks: Optional[list] = None):
        with self._msg_lock:
            self.messages.append(msg)
        # Persist to DB (best-effort; don't break streaming on DB failure)
        try:
            from cheetahclaws.web import db as _db
            _db.repo.append_message(
                self.session_id,
                msg.get("role", "system"),
                msg.get("content", "") or "",
                msg.get("tool_calls"),
                blocks=blocks,
            )
            # Keep in-memory title in sync with auto-titling in repo
            sess = _db.repo.get_session(self.session_id, self.user_id)
            if sess and sess["title"] != self.title:
                self.title = sess["title"]
        except Exception as exc:  # noqa: BLE001
            from cheetahclaws.web.logging_setup import get_logger
            get_logger("api").exception("message persist failed",
                                         extra={"session_id": self.session_id,
                                                "err": str(exc)})

    def get_messages(self) -> list[dict]:
        with self._msg_lock:
            return list(self.messages)

    def get_safe_config(self) -> dict:
        # Reveal the EFFECTIVE config so the web UI shows the real value
        # (e.g. the server config file's thinking/verbose/max_tokens) rather
        # than only the keys this session explicitly overrode.
        return _build_safe_config(
            self.config, getattr(self, "_loaded_overrides", {}))

    def update_config(self, updates: dict) -> dict:
        from cheetahclaws.config import load_config, DEFAULTS
        for k, v in updates.items():
            if k in _WRITABLE_CONFIG_KEYS:
                self.config[k] = v
        # Persist only the *override delta* vs the live server config, so the
        # DB never clobbers later config-file changes.  A key is dropped from
        # the stored override set when its new value matches the file OR the
        # original DEFAULTS (i.e. it is no longer an intentional per-session
        # override distinct from the default).  API-key style keys are always
        # persisted when written (handled below).
        live = load_config()
        self._loaded_overrides = getattr(self, "_loaded_overrides", {})
        for k, v in updates.items():
            if k not in _CONFIG_OVERRIDE_KEYS:
                # Non-override keys (e.g. API keys): store verbatim.
                self._loaded_overrides[k] = v
                continue
            if _is_real_override(k, v, live, DEFAULTS):
                self._loaded_overrides[k] = v
            else:
                self._loaded_overrides.pop(k, None)
        # Persist non-secret config keys to DB (secrets stay session-only)
        try:
            from cheetahclaws.web import db as _db
            _db.repo.upsert_session(
                self.session_id, self.user_id,
                title=self.title,
                config=dict(self._loaded_overrides),
            )
        except Exception as exc:  # noqa: BLE001
            from cheetahclaws.web.logging_setup import get_logger
            get_logger("api").exception("config persist failed",
                                         extra={"session_id": self.session_id,
                                                "err": str(exc)})
        return self.get_safe_config()

    def is_idle(self) -> bool:
        return not self._busy.is_set()

    def is_stale(self) -> bool:
        return (time.monotonic() - self.last_active) > _IDLE_TIMEOUT

    # ── Cleanup ────────────────────────────────────────────────────────

    def cleanup(self):
        from cheetahclaws import runtime
        runtime.release_session_ctx(self.session_id)

def _build_safe_config(config: dict, overrides: dict) -> dict:
    """Build a UI-safe config view from an effective ``config`` dict.

    Shared by live sessions and the no-session default path (so the settings
    panel shows the real server config file values on first load, before any
    chat session exists). ``overrides`` documents which keys were explicitly
    overridden for the session (empty for the no-session defaults).
    """
    result = {k: config.get(k) for k in _SAFE_CONFIG_KEYS
              if k in config}
    # Flag which keys were explicitly overridden for this session, so the
    # UI could show an "inherited from server config" indicator if wanted.
    result["overrides"] = dict(overrides)
    # Show which providers have API keys configured (without revealing them)
    result["api_keys_configured"] = {
        provider: bool(config.get(cfg_key) or
                      os.environ.get(cfg_key.upper(), ""))
        for provider, cfg_key in _API_KEY_CONFIG_MAP.items()
    }
    result["custom_base_url"] = config.get("custom_base_url", "")
    result["ollama_base_url"] = config.get("ollama_base_url",
                                           "http://localhost:11434")
    return result

# ── Session registry ───────────────────────────────────────────────────────

_chat_sessions: dict[str, ChatSession] = {}
_chat_lock = threading.Lock()


def create_chat_session(base_config: dict, user_id: int) -> ChatSession:
    session = ChatSession(base_config, user_id=user_id)
    with _chat_lock:
        _chat_sessions[session.session_id] = session
    return session


def get_chat_session(sid: str,
                    user_id: Optional[int] = None,
                    base_config: Optional[dict] = None) -> Optional[ChatSession]:
    """Return a live ChatSession, hydrating from DB if necessary.

    If the session isn't in the in-memory cache but exists in the DB (and is
    owned by `user_id`), it's lazily rehydrated so restarts don't lose state.
    `user_id` is required for DB hydration; pass None to skip hydration and
    only look in memory (used by internal callers that already validated).
    """
    with _chat_lock:
        sess = _chat_sessions.get(sid)
        if sess:
            # Enforce ownership even on cache hits — otherwise users could
            # read each other's sessions whenever the cache is warm.
            if user_id is not None and sess.user_id != user_id:
                return None
            return sess
    if user_id is None or base_config is None:
        return None
    # Try to hydrate from DB
    try:
        from cheetahclaws.web import db as _db
        row = _db.repo.get_session(sid, user_id)
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    session = ChatSession(base_config, user_id=user_id, session_id=sid)
    with _chat_lock:
        # Guard against a race where another thread hydrated concurrently.
        existing = _chat_sessions.get(sid)
        if existing:
            return existing
        _chat_sessions[sid] = session
    return session


def list_chat_sessions(user_id: int) -> list[dict]:
    """List this user's sessions (DB is the source of truth, not memory)."""
    try:
        from cheetahclaws.web import db as _db
        rows = _db.repo.list_sessions(user_id)
    except Exception as exc:  # noqa: BLE001
        from cheetahclaws.web.logging_setup import get_logger
        get_logger("api").exception("list_sessions failed",
                                     extra={"user_id": user_id,
                                            "err": str(exc)})
        rows = []
    busy_ids = set()
    with _chat_lock:
        for sid, s in _chat_sessions.items():
            if s._busy.is_set():
                busy_ids.add(sid)
    return [{**r, "busy": r["id"] in busy_ids} for r in rows]


def remove_chat_session(sid: str, user_id: int) -> bool:
    """Remove session from DB and in-memory cache. Returns True if removed."""
    try:
        from cheetahclaws.web import db as _db
        deleted = _db.repo.delete_session(sid, user_id)
    except Exception as exc:  # noqa: BLE001
        from cheetahclaws.web.logging_setup import get_logger
        get_logger("api").exception("delete_session failed",
                                     extra={"session_id": sid,
                                            "user_id": user_id,
                                            "err": str(exc)})
        deleted = False
    with _chat_lock:
        session = _chat_sessions.pop(sid, None)
    if session:
        session.cleanup()
    return deleted


def list_folders(user_id: int) -> list[dict]:
    from cheetahclaws.web import db as _db
    return _db.repo.list_folders(user_id)


def create_folder(user_id: int, name: str) -> Optional[dict]:
    from cheetahclaws.web import db as _db
    return _db.repo.create_folder(user_id, name)


def rename_folder(folder_id: int, user_id: int, name: str) -> bool:
    from cheetahclaws.web import db as _db
    return _db.repo.rename_folder(folder_id, user_id, name)


def remove_folder(folder_id: int, user_id: int) -> bool:
    from cheetahclaws.web import db as _db
    return _db.repo.delete_folder(folder_id, user_id)


def move_session_to_folder(sid: str, user_id: int,
                            folder_id: Optional[int]) -> bool:
    from cheetahclaws.web import db as _db
    return _db.repo.move_session_to_folder(sid, user_id, folder_id)


def batch_remove_chat_sessions(sids: list, user_id: int) -> dict:
    """Delete multiple sessions for a user. Cross-user IDs are silently
    skipped (delete_session enforces ownership). Returns counts."""
    deleted = 0
    failed: list[str] = []
    for sid in sids:
        try:
            if remove_chat_session(sid, user_id):
                deleted += 1
            else:
                failed.append(sid)
        except Exception:  # noqa: BLE001
            failed.append(sid)
    return {"deleted": deleted, "failed": failed, "requested": len(sids)}


def batch_export_chat_sessions_markdown(sids: list,
                                         user_id: int) -> Optional[str]:
    """Combine multiple sessions into a single markdown document. Returns
    None when no requested session belongs to the user."""
    parts: list[str] = []
    rendered = 0
    for sid in sids:
        md = export_chat_session_markdown(sid, user_id)
        if md is None:
            continue
        rendered += 1
        if parts:
            parts.append("\n\n---\n\n")
        parts.append(md)
    if rendered == 0:
        return None
    import datetime as _dt
    header = (
        f"# Chat Export — {rendered} session"
        f"{'s' if rendered != 1 else ''}\n\n"
        f"- Exported: {_dt.datetime.now():%Y-%m-%d %H:%M}\n"
        f"- User ID: {user_id}\n\n---\n\n"
    )
    return header + "".join(parts)


def rename_chat_session(sid: str, user_id: int, title: str) -> bool:
    try:
        from cheetahclaws.web import db as _db
        ok = _db.repo.rename_session(sid, user_id, title)
    except Exception:  # noqa: BLE001
        return False
    if ok:
        with _chat_lock:
            s = _chat_sessions.get(sid)
            if s:
                s.title = title.strip()[:200] or "Untitled"
    return ok


def export_chat_session_markdown(sid: str, user_id: int) -> Optional[str]:
    """Render a session's messages as Markdown. Returns None if not found."""
    try:
        from cheetahclaws.web import db as _db
        meta = _db.repo.get_session(sid, user_id)
        if not meta:
            return None
        msgs = _db.repo.get_messages(sid)
    except Exception:  # noqa: BLE001
        return None
    import datetime as _dt
    lines: list[str] = []
    lines.append(f"# {meta['title']}")
    lines.append("")
    lines.append(f"- Session ID: `{sid}`")
    lines.append(f"- Created: {_dt.datetime.fromtimestamp(meta['created_at']):%Y-%m-%d %H:%M}")
    lines.append(f"- Messages: {len(msgs)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    _import_json = __import__("json")
    for m in msgs:
        role = m.get("role", "?")
        when = _dt.datetime.fromtimestamp(m.get("created_at", 0)).strftime("%H:%M:%S")
        lines.append(f"## {role.title()} · {when}")
        lines.append("")
        blocks = m.get("blocks")
        if role == "assistant" and isinstance(blocks, list) and blocks:
            for b in blocks:
                if b.get("type") == "text":
                    if b.get("text"):
                        lines.append(b["text"])
                        lines.append("")
                elif b.get("type") == "tool":
                    lines.append(f"- **{b.get('name','?')}** "
                                 f"(status: {b.get('status','?')})")
                    if b.get("inputs"):
                        lines.append("  ```json")
                        lines.append("  " + _import_json.dumps(b["inputs"], indent=2)
                                     .replace("\n", "\n  "))
                        lines.append("  ```")
                    lines.append("")
                elif b.get("type") == "ask":
                    lines.append(f"> **Question:** {b.get('prompt','')}")
                    lines.append("")
            continue
        lines.append(m.get("content", "") or "_(no content)_")
        if m.get("tool_calls"):
            lines.append("")
            lines.append("<details><summary>Tool calls</summary>")
            lines.append("")
            for tc in m["tool_calls"]:
                lines.append(f"- **{tc.get('name','?')}** "
                             f"(status: {tc.get('status','?')})")
                if tc.get("inputs"):
                    lines.append("  ```json")
                    lines.append("  " + _import_json.dumps(tc["inputs"], indent=2)
                                 .replace("\n", "\n  "))
                    lines.append("  ```")
            lines.append("")
            lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def get_available_models() -> list[dict]:
    """Return all providers and their models for the UI model picker."""
    try:
        from cheetahclaws.providers import PROVIDERS
    except ImportError:
        return []
    result = []
    for name, info in PROVIDERS.items():
        result.append({
            "provider": name,
            "models": list(info.get("models", [])),
            "context_limit": info.get("context_limit", 128000),
            "needs_api_key": info.get("api_key_env") is not None,
            "has_api_key": bool(
                os.environ.get(info.get("api_key_env") or "", "") or
                info.get("api_key", "")
            ),
        })
    return result


def reap_stale_chat_sessions():
    """Periodically evict idle in-memory ChatSession objects to free memory.

    CRITICAL: this only removes the *in-memory* object. The DB row (and all
    message history) is the source of truth for the session list and must
    NEVER be deleted here — sessions rehydrate from the DB on the next request
    via ``get_chat_session()``. Calling ``remove_chat_session()`` (which deletes
    the DB row and cascades to messages) from the reaper was the bug that made
    sessions silently "disappear" from the list after 30 min of inactivity.
    To permanently delete a session, the user must hit the DELETE endpoint,
    which intentionally removes the DB row.

    Sessions with a live WebSocket subscriber are skipped so a quiet-but-open
    browser tab isn't evicted out from under the user (the agent can still push
    events to it).
    """
    stale: list[str] = []
    with _chat_lock:
        for sid, session in _chat_sessions.items():
            if not (session.is_idle() and session.is_stale()):
                continue
            with session._sub_lock:
                has_subscriber = bool(session._subscribers)
            if not has_subscriber:
                stale.append(sid)
    for sid in stale:
        with _chat_lock:
            session = _chat_sessions.pop(sid, None)
        if session is not None:
            try:
                session.cleanup()
            except Exception:  # noqa: BLE001
                pass
