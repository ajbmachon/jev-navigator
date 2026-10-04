"""What a pack's provider block says about how far its ``input_tokens`` can be trusted, and how its
report words that. Both packs (find and trace) write it, so it has one owner."""

from __future__ import annotations

from .judgments.answers import TokenTotal


def usage_receipt(previous: dict | None, input_total: TokenTotal, unanswered_requests: int) -> dict:
    """The responses that reported no usage, the requests that got no response, and whether the
    total is complete. A count an earlier receipt predates stays unknown."""
    without_usage = _plus_known(_carried_count(previous, "responses_without_usage"), input_total.not_reported)
    unanswered = _plus_known(_carried_count(previous, "unanswered_requests"), unanswered_requests)
    return {
        "responses_without_usage": without_usage,
        "unanswered_requests": unanswered,
        "input_tokens_complete": without_usage == 0 and unanswered == 0,
    }


def usage_report_lines(provider: dict) -> list[str]:
    return [
        f"- Responses without usage: {_count_text(provider['responses_without_usage'])}",
        f"- Requests without a response: {_count_text(provider['unanswered_requests'])}",
        f"- Input tokens: {_input_tokens_text(provider)}",
    ]


def _carried_count(previous: dict | None, name: str) -> int | None:
    """A count of an earlier receipt, ``None`` when it predates the field."""
    return 0 if previous is None else previous["provider"].get(name)


def _plus_known(carried: int | None, added: int) -> int | None:
    return None if carried is None else carried + added


def _input_tokens_text(provider: dict) -> str:
    if provider["input_tokens_complete"]:
        return str(provider["input_tokens"])
    return f"at least {provider['input_tokens']} (not complete)"


def _count_text(count: int | None) -> str:
    return "not known (earlier receipt)" if count is None else str(count)
