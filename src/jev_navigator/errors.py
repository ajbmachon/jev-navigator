"""The errors JVN raises when it refuses what a caller asked, as distinct from a bug."""

from __future__ import annotations


class JvnRefusal(Exception):  # noqa: N818 - a base for the named errors, never raised by itself
    """JVN refuses what the caller asked, or cannot do it in the caller's setting. The command line
    reports it in one line; any other error is a bug and keeps its traceback. Each refusal also keeps
    its builtin base, so a caller that catches ValueError or RuntimeError still sees it."""


class UsageError(JvnRefusal, ValueError):
    """JVN refuses an input, a scope, a saved evidence pack or a setting."""
