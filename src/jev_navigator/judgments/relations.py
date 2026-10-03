"""How a place's ``reached_by`` names the move that found it, and how a run pack may keep it.

Most relations name symbols. A key mention quotes a string literal from the code, so a pack that keeps
no code renders it from the place's location instead.
"""

from __future__ import annotations

_KEY_MENTION_PREFIX = "mentions `"


def key_mention(key: str) -> str:
    """The relation of a place reached because it mentions ``key``."""
    return f"{_KEY_MENTION_PREFIX}{key}`"


def without_quoted_code(reached_by: str, file: str, line: int) -> str:
    """``reached_by`` as a pack without code keeps it: a key mention becomes "mentions a key
    (file:line)", every other relation stays as it is."""
    if reached_by.startswith(_KEY_MENTION_PREFIX):
        return f"mentions a key ({file}:{line})"
    return reached_by
