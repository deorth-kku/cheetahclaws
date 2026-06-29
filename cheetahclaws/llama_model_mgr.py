"""
llama_model_mgr.py — Automatic model loading for llama.cpp servers.

When the provider is "custom" pointing to a llama.cpp server (owned_by=llamacpp),
automatically:
  1. Check if the requested model exists via GET /v1/models
  2. If the model exists but is "unloaded", load it via POST /models/load
  3. If the model doesn't exist, warn the user

This module is called from:
  - cli.py:main() after bootstrap (startup hook)
  - commands/config_cmd.py:cmd_model() after model switch
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

USER_AGNET = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"}


def _get_base_url(config: dict) -> str:
    """Get the base URL for custom provider from config or env."""
    return (
        config.get("custom_base_url")
        or os.environ.get("CUSTOM_BASE_URL", "")
    )


def _get_api_key(config: dict) -> str:
    """Get the API key for custom provider from config or env."""
    return config.get("custom_api_key") or os.environ.get("CUSTOM_API_KEY", "")


def _normalize_base_url(base_url: str) -> str:
    """Normalize base URL to ensure correct path construction."""
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        return stripped
    return stripped + "/v1"


def check_model_exists(base_url: str, model_name: str) -> bool:
    """Check if a model exists on the llama.cpp server.
    
    Returns True if the model exists, False otherwise.
    """
    try:
        url = _normalize_base_url(base_url) + "/models"
        req = urllib.request.Request(
            url,
            headers={**USER_AGNET},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            return any(m.get("id") == model_name for m in models)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return False


def get_model_info(base_url: str, model_name: str) -> dict | None:
    """Get model info from the llama.cpp server.
    
    Returns the model dict if found, None otherwise.
    """
    try:
        url = _normalize_base_url(base_url) + "/models"
        req = urllib.request.Request(
            url,
            headers={**USER_AGNET},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            for m in models:
                if m.get("id") == model_name:
                    return m
            return None
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return None


def load_model(base_url: str, model_name: str) -> bool:
    """Load a model on the llama.cpp server.
    
    Calls POST /models/load with {"model": model_name}.
    Returns True if successful, False otherwise.
    """
    url = _normalize_base_url(base_url).rstrip("/v1") + "/models/load"
    payload = json.dumps({"model": model_name}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            **USER_AGNET,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("success", False)


def check_and_load_llama_model(config: dict) -> None:
    """Main entry point: check model status and load if needed.
    
    Called during startup and after model switches when provider is "custom".
    Logic:
      1. If provider is not "custom", do nothing
      2. If base_url is not configured, do nothing
      3. Get current model name
      4. Check if model exists on server
      5. If exists and owned_by=llamacpp and status.value=unloaded, load it
      6. If doesn't exist, warn user
    """
    from cheetahclaws.providers import detect_provider, bare_model
    
    # Only handle custom provider
    model = config.get("model", "")
    if not model:
        return
    
    provider = detect_provider(model)
    if provider != "custom":
        return
    
    # Get base URL
    base_url = _get_base_url(config)
    if not base_url:
        return
    
    # Get bare model name (strip "custom/" prefix)
    model_name = bare_model(model)
    
    # Check if model exists
    model_info = get_model_info(base_url, model_name)
    if model_info is None:
        # Model doesn't exist on server
        from cheetahclaws.ui.render import warn
        warn(f"Model '{model_name}' not found on custom api server at {base_url}")
        return
    
    # Check if we need to load it
    owned_by = model_info.get("owned_by", "")
    status_value = model_info.get("status", {}).get("value", "")
    
    if owned_by == "llamacpp" and status_value == "unloaded":
        # Model needs to be loaded
        from cheetahclaws.ui.render import info
        info(f"Model '{model_name}' is unloaded, loading...")
        success = load_model(base_url, model_name)
        if success:
            info(f"Model '{model_name}' loaded successfully")
        else:
            from cheetahclaws.ui.render import err
            err(f"Failed to load model '{model_name}'")
    elif status_value == "loaded":
        # Model is already loaded, all good
        pass
    else:
        # Model exists but in some other state
        from cheetahclaws.ui.render import info
        info(f"Model '{model_name}' status: {status_value}")


def check_model_on_switch(config: dict) -> None:
    """Wrapper for model switch hook.
    
    Same logic as check_and_load_llama_model but called after model change.
    """
    check_and_load_llama_model(config)
