# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""LiteLLM proxy callback that replaces image content with a text note.

Loaded by the proxy ``anthropic_proxy.serve`` starts (copied next to its
config). A text-only server rejects any image part with a 400, which ends a
Claude Code session: its Read tool returns image files as image blocks. The
note tells the model the image was dropped, so it can inspect the file another
way.
"""

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # imported outside the ANTHROPIC_PROXY venv
    CustomLogger = object

IMAGE_TYPES = frozenset({"image", "image_url", "input_image"})
IMAGE_NOTE = (
    "[image omitted: this model cannot view images; "
    "inspect the file with shell tools instead]"
)


def strip_images(value):
    """``value`` with every image part, at any depth, replaced by the note."""
    if isinstance(value, list):
        return [strip_images(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") in IMAGE_TYPES:
            return {"type": "text", "text": IMAGE_NOTE}
        return {key: strip_images(item) for key, item in value.items()}
    return value


class ImageFilter(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if "messages" in data:
            data["messages"] = strip_images(data["messages"])
        return data


proxy_handler_instance = ImageFilter()
