"""Durable provider journal with concise live terminal progress."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from .judgments.answers import NOT_REPORTED_TEXT, TokenTotal, reported_input_tokens, reported_output_tokens
from .judgments.journal import JournalRequest, JsonlJournal, RawAttempt, RawResponse, error_message
from .run_files import failure_digested, place_location, step_shown

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class TerminalProgress:
    def __init__(self, journal_path: Path | None, *, verbose: bool = False) -> None:
        self.journal_path = journal_path
        self.verbose = verbose
        self.started = monotonic()
        self.phase_name = "starting"
        self.requests = 0
        self.responses = 0
        self.failures = 0
        self.input_total = TokenTotal()
        self.output_total = TokenTotal()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tty = sys.stderr.isatty()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._event(f"journal {self.journal_path}" if self.journal_path else "local analysis")
        if self._tty:
            self._thread = threading.Thread(target=self._spin, name="jvn-progress", daemon=True)
            self._thread.start()

    def phase(self, name: str) -> None:
        with self._lock:
            self.phase_name = name
        self._event(f"phase {name}")

    def scan(self, name: str, event: str, file_count: int) -> None:
        phase = f"scanning {name} ({file_count:,} code files)"
        with self._lock:
            self.phase_name = phase if event == "started" else f"{name} scan {event}"
        self._event(f"{phase} {event}; {self.elapsed():.1f}s elapsed")

    def request(self, request_id: str, request: JournalRequest) -> None:
        with self._lock:
            self.requests += 1
            number = self.requests
        state = request.state
        source = state.get("slice", {}) if isinstance(state, dict) else {}
        file = source.get("file", "") if isinstance(source, dict) else ""
        lines = source.get("lines", "") if isinstance(source, dict) else ""
        if isinstance(lines, (list, tuple)) and len(lines) == 2:
            lines = f"{lines[0]}-{lines[1]}"
        location = f" {file}:{lines}" if file else ""
        question_ids = ", ".join(request.questions)
        self._event(
            f"request {number} started{location}; purpose {question_ids}; {self.elapsed():.1f}s elapsed"
        )
        if self.verbose:
            self._event(
                "masked request "
                + json.dumps({"state": request.state, "questions": request.questions}, default=str)
            )

    def response(self, request_id: str, response: RawResponse) -> None:
        input_tokens, output_tokens = _usage(response)
        with self._lock:
            self.responses += 1
            self.input_total.add(input_tokens)
            self.output_total.add(output_tokens)
            number = self.responses
            totals = f"{_summary(self.input_total)} in/{_summary(self.output_total)} out"
        model = _model(response)
        self._event(
            f"response {number} received"
            f"{f' from {model}' if model else ''}; usage {_added(input_tokens)} in/"
            f"{_added(output_tokens)} out; total {totals}; {self.elapsed():.1f}s elapsed"
        )

    def failure(self, request_id: str, error: str) -> None:
        with self._lock:
            self.failures += 1
        self._event(f"request failed: {error}")

    def close(self, outcome: str) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._clear_spinner()
        self._event(
            f"{outcome}; {self.requests} requests, {self.responses} responses, "
            f"{_summary(self.input_total)} input tokens, {_summary(self.output_total)} output tokens, "
            f"{self.elapsed():.1f}s elapsed"
        )

    def elapsed(self) -> float:
        return monotonic() - self.started

    def _spin(self) -> None:
        position = 0
        while not self._stop.wait(0.1):
            with self._lock:
                line = (
                    f"{_SPINNER[position % len(_SPINNER)]} {self.phase_name} · {self.elapsed():.1f}s · "
                    f"{self.requests} requests/{self.responses} responses · "
                    f"{_summary(self.input_total)} in/{_summary(self.output_total)} out"
                )
            sys.stderr.write(f"\r{line[:160]:<160}")
            sys.stderr.flush()
            position += 1

    def _event(self, message: str) -> None:
        with self._lock:
            self._clear_spinner()
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            print(f"jvn [{stamp}] {message}", file=sys.stderr, flush=True)

    def _clear_spinner(self) -> None:
        if self._tty:
            sys.stderr.write("\r" + " " * 160 + "\r")


class ProgressJournal(JsonlJournal):
    """Without ``keep_request_text`` a history step shows each neighbour by ``place_label`` (the CLI
    sets the run's ``run_files.PlaceLabels`` once the index and any resumed frontier exist) and every
    relation as a run file keeps it."""

    def __init__(
        self,
        path: Path,
        progress: TerminalProgress,
        *,
        keep_request_text: bool = False,
        keep_error_text: bool = True,
    ) -> None:
        super().__init__(path, keep_request_text=keep_request_text, keep_error_text=keep_error_text)
        self.progress = progress
        self.place_label: Callable[[str], str] = place_location
        self.routes: dict[str, str] = {}
        self.statuses: dict[str, int] = {}

    def record_step(self, step: Mapping) -> None:
        shown = step if self.keep_request_text else self._shown(step)
        super().record_step(shown if self.keeps_error_text else failure_digested(shown))

    def _shown(self, step: Mapping) -> dict:
        shown = step_shown(step)
        judgments = shown["judgments"]
        if "could_contain" in judgments:
            judgments["could_contain"] = [
                {**offered, "signature": self.place_label(offered["place"])}
                for offered in judgments["could_contain"]
            ]
        return shown

    def record_request(self, request: JournalRequest) -> str:
        request_id = super().record_request(request)
        self.progress.request(request_id, request)
        return request_id

    def record_response(self, request_id: str, response: RawResponse) -> None:
        super().record_response(request_id, response)
        self.progress.response(request_id, response)

    def record_attempt(self, request_id: str, attempt: RawAttempt) -> None:
        """Also remembers the route a routed client sent the request on, and the HTTP status of its
        latest answer."""
        super().record_attempt(request_id, attempt)
        if attempt.route is not None:
            self.routes[request_id] = attempt.route
        self._remember_status(request_id, attempt.response)

    def record_failure(
        self, request_id: str, error: BaseException, response: RawResponse | None = None
    ) -> None:
        """The run file keeps the message as ``message_fields`` allows; stderr always shows it whole."""
        super().record_failure(request_id, error, response)
        self._remember_status(request_id, response)
        self.progress.failure(request_id, f"{type(error).__name__}: {error_message(error)}")

    def _remember_status(self, request_id: str, response: RawResponse | None) -> None:
        if response is not None and response.status is not None:
            self.statuses[request_id] = response.status

    def record_terminal(self, outcome: str) -> None:
        self._append({"kind": "terminal", "outcome": outcome})


def _usage(response: RawResponse) -> tuple[int | None, int | None]:
    try:
        raw = response.json()
    except Exception:  # noqa: BLE001 - progress never replaces the durable parser failure
        return None, None
    if not isinstance(raw, dict):
        return None, None
    return reported_input_tokens(raw), reported_output_tokens(raw)


def _added(tokens: int | None) -> str:
    return NOT_REPORTED_TEXT if tokens is None else f"+{tokens}"


def _summary(total: TokenTotal) -> str:
    if total.not_reported == 0:
        return str(total.reported)
    return f"{total.reported} ({total.not_reported} {NOT_REPORTED_TEXT})"


def _model(response: RawResponse) -> str:
    try:
        raw = response.json()
    except Exception:  # noqa: BLE001 - progress never replaces the durable parser failure
        return ""
    model = raw.get("model", "") if isinstance(raw, dict) else ""
    return str(model) if model else ""
