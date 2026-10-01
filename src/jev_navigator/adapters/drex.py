"""Drex by Nace.AI, the hosted System-One decision model at `https://drex.nace.ai`.

Drex speaks the `/v1/systemone` wire with one stricter rule: every criterion is a string (or null
for an undescribed choice option). The library's checks describe their outcomes as objects
(``{"what", "not_for", "examples"}``), which Drex refuses with HTTP 422, so this adapter sends
each such criterion as its compact JSON text. Keeping the fields labelled matters: rendered as one
prose sentence instead, the same criteria made Drex confirm unrelated code in a live search. The
request the judge hashes and stores is unchanged; the journal's sent body records the text Drex
actually received.

Drex also refuses a whole request, with HTTP 422 "is media", when any text holds a base64 data URL
(``data:image/jpeg;base64,``, case-insensitive, anywhere in a string, in the state or a question).
Source code builds such strings all the time, so this adapter puts a zero-width space after that
``data``. Drex then reads the code as text; the break is invisible to the model and changes no id
an answer refers to. Other ``data:`` text, such as ``{ data: rows }``, is sent unchanged.

The key is `DREX_API_KEY`, Nace.AI's own convention, and the endpoint is pinned. Drex is not
deterministic: identical requests can get slightly different probabilities, so one live run is not
a reproducible measurement.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from .system_one import SystemOneClient

# A base64 data URL, which Drex refuses as media: `data:` then a media type and `;base64,`.
BASE64_DATA_URL = re.compile(r"(data)(?=:[^\s,]*;base64,)", re.IGNORECASE)
ZERO_WIDTH_SPACE = "\u200b"


class DrexClient(SystemOneClient):
    name = "drex"
    endpoint = "https://drex.nace.ai"
    default_model = "drex-latest"
    api_key_env = "DREX_API_KEY"
    needs_key = True
    pinned = True

    def wire_questions(self, questions: Mapping) -> dict:
        return {question_id: _with_text_criteria(question) for question_id, question in questions.items()}

    def wire_body(self, state: Mapping, questions: Mapping) -> dict:
        body = super().wire_body(state, questions)
        return {**body, "state": _as_text(body["state"]), "questions": _as_text(body["questions"])}


def _as_text(value):
    """Every string with its base64 data URLs broken; keys, which answers refer to, unchanged."""
    if isinstance(value, str):
        return BASE64_DATA_URL.sub(rf"\1{ZERO_WIDTH_SPACE}", value)
    if isinstance(value, Mapping):
        return {key: _as_text(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_as_text(item) for item in value]
    return value


def _with_text_criteria(question: Mapping) -> dict:
    criteria = question.get("criteria")
    if isinstance(criteria, Mapping):
        criteria = {label: _criterion_text(value) for label, value in criteria.items()}
    elif isinstance(criteria, list):
        criteria = [_criterion_text(value) for value in criteria]
    else:
        return dict(question)
    return {**question, "criteria": criteria}


def _criterion_text(value: object) -> str | None:
    """Text and null pass unchanged; a structured criterion is sent as its compact JSON text."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
