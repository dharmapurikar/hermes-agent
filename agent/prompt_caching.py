"""Prompt caching strategies for Anthropic and OpenAI-compatible APIs.

Anthropic (system_and_3 strategy):
  Reduces input token costs by ~75% on multi-turn conversations by caching
  the conversation prefix. Uses 4 cache_control breakpoints (Anthropic max):
    1. System prompt (stable across all turns)
    2-4. Last 3 non-system messages (rolling window)

OpenAI-compatible (prefix stability strategy):
  OpenAI-compatible APIs (including z.ai/GLM) use automatic prefix caching
  on the server side -- identical prefixes across requests are cached
  automatically. No special request markers are needed. The key is to keep
  the system prompt and early messages stable across turns.
  Cache hits are reported via prompt_tokens_details.cached_tokens in the
  response usage object.

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to 4 cache_control breakpoints: system prompt + last 3 non-system messages.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages


def apply_openai_cache_control(
    api_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prepare messages for OpenAI-compatible automatic prefix caching.

    OpenAI-compatible APIs cache prefixes automatically on the server side.
    This function ensures prefix stability by:
    - Not mutating message order or content of earlier messages
    - Returning a shallow copy (no deep copy needed since we don't modify)

    The server handles caching transparently -- cache hits are reported via
    prompt_tokens_details.cached_tokens in the response.

    Returns:
        Messages ready for the API (unchanged -- prefix caching is automatic).
    """
    # OpenAI-compatible prefix caching is automatic and server-side.
    # No request-side markers needed. The API caches based on matching
    # token prefixes across requests. We return messages as-is.
    return api_messages
