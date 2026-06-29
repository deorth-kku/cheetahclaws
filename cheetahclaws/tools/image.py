"""Image reading tool — injects image into vision model context.

Unlike ReadImage (OCR-only), this tool sends the actual image to the vision
model so it can "see" screenshots, diagrams, photos, etc.
"""
from __future__ import annotations


def _read_image_tool(params: dict, config: dict) -> str:
    """Read an image file and inject it into the vision model context."""
    file_path = params.get("file_path", "")
    prompt = params.get("prompt", "What do you see in this image? Describe it in detail.")

    if not file_path:
        return "Error: file_path is required."

    from pathlib import Path

    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"

    # Supported formats
    supported = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    ext = p.suffix.lower()
    if ext not in supported:
        return f"Error: unsupported format {ext}. Supported: {', '.join(sorted(supported))}"

    try:
        import base64

        with open(p, "rb") as f:
            img_bytes = f.read()

        # Optional size check (default 20MB)
        max_size = params.get("max_size", 20 * 1024 * 1024)
        if len(img_bytes) > max_size:
            return f"Error: file too large ({len(img_bytes) / 1024 / 1024:.1f} MB). Max: {max_size / 1024 / 1024:.0f} MB."

        b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Inject into vision model context
        from cheetahclaws import runtime

        runtime.get_ctx(config).pending_image = b64

        size_kb = len(img_bytes) / 1024
        return (
            f"Image loaded: {p.name} ({size_kb:.0f} KB, format: {ext[1:]})\n"
            f"Prompt: {prompt}\n\n"
            "The image has been sent to the vision model. It will now analyze the image."
        )
    except Exception as e:
        return f"Error reading image: {type(e).__name__}: {e}"


# ── Register ─────────────────────────────────────────────────────────────

from cheetahclaws.tool_registry import ToolDef, register_tool

register_tool(
    ToolDef(
        name="ViewImage",
        schema={
            "name": "ViewImage",
            "description": (
                "Read an image file from disk and inject it into the vision model's context. "
                "Use this when the model needs to see an image (e.g., screenshots, diagrams, photos). "
                "The image is sent as base64 to the vision model for analysis."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the image file",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional instruction for the vision model (default: 'What do you see in this image?')",
                    },
                    "max_size": {
                        "type": "integer",
                        "description": "Maximum file size in bytes (default: 20MB)",
                    },
                },
                "required": ["file_path"],
            },
        },
        func=_read_image_tool,
        read_only=False,
        concurrent_safe=False,
    )
)
