"""The error JVN raises when it refuses what a caller asked, as distinct from a bug."""

from __future__ import annotations


class UsageError(ValueError):
    """JVN refuses what the caller asked: an input, a scope, a saved evidence pack or a setting.
    The command line reports it in one line; any other error is a bug and keeps its traceback."""
