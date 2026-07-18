"""Tool plugin registry for cheetahclaws.

Provides a central registry for tool definitions, lookup, schema export,
dispatch with output truncation, and result caching for read-only tools.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDef:
    """Definition of a single tool plugin.

    Attributes:
        name: unique tool identifier
        schema: JSON-schema dict sent to the API (name, description, input_schema)
        func: callable(params: dict, config: dict) -> str
        read_only: True if the tool never mutates state
        concurrent_safe: True if safe to run in parallel with other tools
    """
    name: str
    schema: Dict[str, Any]
    func: Callable[[Dict[str, Any], Dict[str, Any]], str]
    read_only: bool = False
    concurrent_safe: bool = False


# --------------- internal state ---------------

_registry: Dict[str, ToolDef] = {}

# --------------- result cache (read-only tools only) ---------------

_CACHE_MAX = 64  # max cached entries
_cache: Dict[str, str] = {}   # hash → result
_cache_order: list[str] = []  # LRU eviction order


def _cache_key(name: str, params: Dict[str, Any], session_id: str = "") -> str:
    """Create a stable hash from tool name + params + session.

    Including the session_id keeps cached results scoped to the originator —
    in a shared daemon, A's Read of ~/.env never gets handed to B's session.
    """
    raw = json.dumps(
        {"n": name, "p": params, "s": session_id},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def clear_tool_cache() -> None:
    """Clear the tool result cache. Called on file writes to invalidate."""
    _cache.clear()
    _cache_order.clear()


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance between two strings (iterative, O(n·m))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,       # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = cur
    return prev[len(b)]


def _suggest_param(missing: str, candidates: List[str]) -> Optional[str]:
    """Suggest the most similar candidate key for a missing required param.

    Returns the closest candidate within a length-scaled edit-distance
    threshold, or ``None`` if nothing is close enough (e.g. the caller sent
    a completely unrelated or empty key set). Threshold scales with the
    missing key's length so short keys stay strict (avoid `id`~`aid`) while
    longer keys tolerate a couple of typos/inflections.
    """
    if not candidates:
        return None
    # Allow at most ~40% of the key to differ, with a floor of 1 and a
    # ceiling of 4 so very long keys don't require an exact match.
    _max_dist = max(1, min(4, int(round(len(missing) * 0.4))))
    _best, _best_d = None, None
    for c in candidates:
        if c == missing:
            continue
        d = _levenshtein(missing, c)
        if d <= _max_dist and (_best_d is None or d < _best_d):
            _best, _best_d = c, d
    return _best


# --------------- public API ---------------

def register_tool(tool_def: ToolDef) -> None:
    """Register a tool, overwriting any existing tool with the same name."""
    _registry[tool_def.name] = tool_def


def get_tool(name: str) -> Optional[ToolDef]:
    """Look up a tool by name. Returns None if not found."""
    return _registry.get(name)


def get_all_tools() -> List[ToolDef]:
    """Return all registered tools (insertion order)."""
    return list(_registry.values())


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return the schemas of all registered tools (for API tool parameter)."""
    return [t.schema for t in _registry.values()]


def execute_tool(
    name: str,
    params: Dict[str, Any],
    config: Dict[str, Any],
    max_output: int = 32000,
) -> str:
    """Dispatch a tool call by name.

    Args:
        name: tool name
        params: tool input parameters dict
        config: runtime configuration dict
        max_output: maximum allowed output length in characters

    Returns:
        Tool result string, possibly truncated.
    """
    tool = get_tool(name)
    if tool is None:
        return f"Error: tool '{name}' not found."

    # Centralized required-parameter validation. Historically each ToolDef's
    # lambda hard-indexed params (e.g. `p["question"]`), so a model that fired
    # a tool_call with missing/renamed args raised a bare `KeyError: 'question'`
    # which bubbled up as ``Error executing {name}: 'question'`` — an
    # unhelpful message the model could not self-correct from (see docs/news.md
    # "KeyError: 'file_path'" entry). Validate against the schema's `required`
    # list here, *before* the lambda runs, so every tool (built-in, MCP, and
    # plugin-registered) gets a uniform, schema-aware error with the exact
    # missing key name and the keys actually received.
    _req = (
        tool.schema.get("input_schema", {}).get("required", [])
        if isinstance(tool.schema, dict) else []
    )
    if _req:
        _missing = [k for k in _req if k not in params]
        if _missing:
            # Offer a "did you mean?" hint by fuzzy-matching each missing
            # required key against the keys the caller actually sent. This
            # catches schema drift / typos generically (e.g. `questions`
            # instead of `question`, `filepath` instead of `file_path`,
            # `notebok_path` instead of `notebook_path`) without hardcoding
            # any per-tool special cases.
            _hint_parts = []
            for _m in _missing:
                _sug = _suggest_param(_m, list(params.keys()))
                if _sug:
                    _hint_parts.append(f"'{_m}' (did you mean '{_sug}'?)")
                else:
                    _hint_parts.append(f"'{_m}'")
            return (
                "Error: missing required parameter"
                f"{_hint_parts}. Received keys: {sorted(params.keys())}"
            )

    # Cache hit for read-only tools (same name + same params + same session).
    use_cache = tool.read_only
    if use_cache:
        sid = (config or {}).get("_session_id", "") or ""
        key = _cache_key(name, params, sid)
        if key in _cache:
            return _cache[key]
    else:
        # Write tools invalidate cache (file content may have changed)
        if name in ("Write", "Edit", "Bash", "NotebookEdit"):
            clear_tool_cache()

    try:
        result = tool.func(params, config)
    except Exception as e:
        return f"Error executing {name}: {e}"

    # Store in cache for read-only tools
    if use_cache:
        _cache[key] = result
        _cache_order.append(key)
        # Evict oldest if over limit
        while len(_cache_order) > _CACHE_MAX:
            old = _cache_order.pop(0)
            _cache.pop(old, None)

    conf_max_output=config.get("max_tool_output")
    if conf_max_output:
        max_output=conf_max_output
    # Model-aware truncation: the static 32K-char cap is fine for English
    # but blows up CJK content (1 token per char). Cap effective max by the
    # model's actual context window so a Bash / Read / WebFetch result
    # can never single-handedly overflow the next API call. ~30K-token
    # conservative ceiling (handles 32K-context models like qwen2.5-72b
    # behind a `custom/` provider that lies about context_limit).
    try:
        from cheetahclaws.compaction import get_context_limit
        model = config.get("model", "") if config else ""
        declared_ctx = get_context_limit(model) or 32768
        # Reserve 16K for system prompt + tool schemas + framing + headroom.
        # 0.5× for CJK-safety (1 char ≈ 1 token worst case).
        safe_ctx = declared_ctx
        effective_max = max(2000, int((safe_ctx - 16000) * 0.5))
        if effective_max < max_output:
            max_output = effective_max
    except Exception:
        # Compaction module unavailable in some test contexts — fall back
        # to the static 32K cap rather than crashing.
        pass

    if len(result) > max_output:
        first_half = max_output // 2
        last_quarter = max_output // 4
        truncated = len(result) - first_half - last_quarter
        # Surface a SummarizeLargeFile pointer when the truncated tool
        # call had a `file_path` arg — gives the model a path forward
        # instead of just losing 50%+ of the content.
        file_hint = ""
        fpath = (params or {}).get("file_path") if isinstance(params, dict) else None
        if fpath and isinstance(fpath, str):
            file_hint = (
                f"  Tip: this came from `{fpath}` — call "
                f"`SummarizeLargeFile(file_path='{fpath}')` to get a "
                f"complete chunked + map-reduce summary that fits."
            )
        result = (
            result[:first_half]
            + f"\n[... {truncated} chars truncated to keep total tool "
              f"output ≤ {max_output:,} chars (model context safety).\n"
              f"{file_hint}]\n"
            + result[-last_quarter:]
        )

    return result


def clear_registry() -> None:
    """Remove all registered tools. Intended for testing."""
    _registry.clear()
