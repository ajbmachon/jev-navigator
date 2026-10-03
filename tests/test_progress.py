from __future__ import annotations

import pytest

from jev_navigator.judgments.journal import RawResponse
from jev_navigator.progress import TerminalProgress


def _response(body: bytes) -> RawResponse:
    return RawResponse(body, 200, "application/json")


def test_progress_says_not_reported_for_a_response_without_usage_and_never_adds_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress = TerminalProgress(None)

    progress.response("r1", _response(b'{"model": "m", "usage": {"input_tokens": 7, "output_tokens": 1}}'))
    progress.response("r2", _response(b'{"model": "m", "answers": {}}'))
    progress.close("done")

    log = capsys.readouterr().err
    assert "usage not reported in" in log
    assert "total 7 (1 not reported) in" in log
    assert "7 (1 not reported) input tokens" in log


def test_progress_says_not_reported_for_output_tokens_the_same_way(
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress = TerminalProgress(None)

    progress.response("r1", _response(b'{"model": "m", "usage": {"input_tokens": 7, "output_tokens": 3}}'))
    progress.response("r2", _response(b'{"model": "m", "usage": {"input_tokens": 5}}'))
    progress.close("done")

    log = capsys.readouterr().err
    assert "usage +5 in/not reported out" in log
    assert "3 (1 not reported) output tokens" in log
