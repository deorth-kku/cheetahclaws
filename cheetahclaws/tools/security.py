"""tools_security.py — Path-traversal guard and bash safety check."""
from __future__ import annotations

import os
from pathlib import Path

# Prefixes that are safe to run without a permission prompt
_SAFE_PREFIXES = (
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "date",
    "which", "type", "env", "printenv", "uname", "whoami", "id",
    "git log", "git status", "git diff", "git show", "git branch",
    "git remote", "git stash list", "git tag",
    "find ", "grep ", "rg ", "ag ", "fd ",
    "python ", "python3 ", "node ", "ruby ", "perl ",
    "pip show", "pip list", "npm list", "cargo metadata",
    "df ", "du ", "free ", "top -bn", "ps ",
    "curl -I", "curl --head",
)


_CHAIN_OPERATORS = (";", "&&", "||", "`", "$(", "\n")

import re as _re

_CD_PREFIX_RE = _re.compile(r'^cd\s+(.+?)\s+&&\s*(.*)$', _re.DOTALL)

# Commands safe to use as pipe tails — strictly read-only, no side effects.
# If the pipe tail is anything else (curl, rm, xargs, sh, bash, etc.), the
# whole command is rejected even if the prefix is safe.
_SAFE_PIPE_TAIL = (
    "head", "tail", "wc", "sort", "uniq",
    "tr", "cut", "sed", "awk", "rev", "split", "paste",
    "less", "more", "col", "fmt", "fold", "expand", "unexpand",
    "nl", "number", "pr", "column", "join", "comm", "diff", "cmp",
    "xxd", "od", "hexdump", "base64", "base32",
    "grep", "rg", "ag",
    "true", "false", "test",
)


def _is_safe_bash(cmd: str) -> bool:
    """Return True if cmd is read-only and never needs a permission prompt.

    Allows:
      - Simple safe-prefix commands (ls, git status, etc.)
      - Safe-prefix | safe-tail (grep ... | head, find ... | wc -l, etc.)
    Rejects:
      - Commands with dangerous chaining operators (;, &&, ||, `, $())
      - Pipes to unsafe tails (| rm, | curl, | xargs, | sh, etc.)
    """
    c = cmd.strip()

    # Strip "cd <dir> && " prefix when <dir> is the current directory or a
    # subdirectory of it.  This is a common agent pattern (e.g. "cd project
    # && ls") and the cd itself is harmless in that case.
    _m = _CD_PREFIX_RE.match(c)
    if _m:
        target_dir = _m.group(1).strip()
        remainder = _m.group(2).strip()
        try:
            cwd = Path(os.getcwd()).resolve()
            (cwd / target_dir).resolve().relative_to(cwd)
            c = remainder
        except Exception:
            pass  # not a subdirectory or resolution failed — keep original cmd

    # Reject truly dangerous chaining operators (no pipe — pipe is handled below)
    if any(op in c for op in _CHAIN_OPERATORS):
        return False

    # Check for pipe: if present, split on " | " and validate both sides
    if "|" in c:
        # Split on " | " (with spaces) to avoid matching >> or || etc.
        # Fallback: split on bare | if no spaced variant found
        if " | " in c:
            parts = c.split(" | ")
        else:
            # bare | without spaces — still allow if all parts are safe
            parts = c.split("|")

        # Every part must be a simple command (no further chaining within)
        for part in parts:
            part = part.strip()
            if not part:
                return False
            if any(op in part for op in _CHAIN_OPERATORS):
                return False

        # The first part must match a safe prefix
        if not any(c.startswith(p) for p in _SAFE_PREFIXES):
            return False

        # Every pipe tail must be an allowed safe command
        for part in parts[1:]:
            tail_cmd = part.strip().split()[0] if part.strip() else ""
            if tail_cmd not in _SAFE_PIPE_TAIL:
                return False

        return True

    return any(c.startswith(p) for p in _SAFE_PREFIXES)


# Path patterns that hold credentials or system secrets — never accessed by
# default, even when no allowed_root is configured. Set
# CHEETAHCLAWS_FS_NO_SANDBOX=1 to bypass (e.g. when intentionally auditing
# your own secrets).
_HOME = Path.home()
_SECRET_DIRS = (
    _HOME / ".aws",
    _HOME / ".gnupg",
    _HOME / ".kube",
    _HOME / ".docker",
    Path("/root"),
    Path("/etc/sudoers.d"),
)
_SECRET_FILES = (
    _HOME / ".netrc",
    _HOME / ".pgpass",
    Path("/etc/shadow"),
    Path("/etc/gshadow"),
    Path("/etc/sudoers"),
)
_SECRET_SSH_PREFIX = _HOME / ".ssh"
_SECRET_SSH_PUBLIC = {"config", "known_hosts", "known_hosts.old", "authorized_keys"}


def _is_secret_path(resolved: Path) -> bool:
    """Best-effort check: is this path a known credential / secret store?"""
    for d in _SECRET_DIRS:
        try:
            resolved.relative_to(d.resolve(strict=False))
            return True
        except ValueError:
            continue
    for f in _SECRET_FILES:
        try:
            if resolved == f.resolve(strict=False):
                return True
        except OSError:
            continue
    # ~/.ssh: deny everything except the documented public files (config,
    # known_hosts, authorized_keys). Private keys (id_*) are always denied.
    try:
        rel = resolved.relative_to(_SECRET_SSH_PREFIX.resolve(strict=False))
        return rel.name not in _SECRET_SSH_PUBLIC or rel.parent != Path(".")
    except ValueError:
        pass
    return False


def _check_path_allowed(file_path: str, config: dict) -> str | None:
    """Return an error string if file_path is disallowed, else None.

    Two layers of defense:
      1. If config["allowed_root"] / config["_worktree_cwd"] is set, the
         file_path must resolve inside that root.
      2. Independent of (1), a default credential denylist refuses paths
         like ~/.ssh/id_*, ~/.aws/credentials, /etc/shadow, etc.
         Set CHEETAHCLAWS_FS_NO_SANDBOX=1 to disable layer (2).
    """
    try:
        resolved = Path(file_path).resolve()
    except Exception as e:
        return f"Error: path validation failed: {e}"

    allowed_root = config.get("allowed_root") or config.get("_worktree_cwd")
    if allowed_root:
        try:
            root = Path(allowed_root).resolve()
            resolved.relative_to(root)
        except ValueError:
            return (
                f"Error: path '{file_path}' is outside the allowed root '{allowed_root}'. "
                "Set config['allowed_root'] to a broader directory if this is intentional."
            )
        except Exception as e:
            return f"Error: path validation failed: {e}"

    if os.environ.get("CHEETAHCLAWS_FS_NO_SANDBOX", "0") != "1":
        if _is_secret_path(resolved):
            return (
                f"Error: path '{file_path}' is on the credential denylist "
                f"(SSH keys, ~/.aws, ~/.gnupg, /etc/shadow, etc.). "
                f"Set CHEETAHCLAWS_FS_NO_SANDBOX=1 to override."
            )
    return None
