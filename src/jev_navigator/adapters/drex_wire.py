"""Drex's wire dialect: the request as Drex accepts it, built from the request the judge made.

Drex speaks the `/v1/systemone` wire with two stricter rules than Jev:

- Every criterion is a string, or null for an undescribed choice option. The library's checks
  describe their outcomes as objects (``{"what", "not_for", "examples"}``), which Drex refuses with
  HTTP 422 ("must be a string"), so each such criterion goes as its compact JSON text. The fields
  stay labelled on purpose: rendered as one prose sentence instead, the same criteria made Drex
  confirm unrelated code in a live search.
- Drex refuses a whole request, with HTTP 422 "is media", when any string holds a base64 data URL
  (``data:image/png;base64,``, case-insensitive, anywhere in the state or a question). Source code
  builds such strings all the time, so a zero-width space goes after that ``data``. Drex then reads
  the code as text; the break is invisible to the model and changes no key an answer refers to.
  Other ``data:`` text, such as ``{ data: rows }``, is sent unchanged.

Only the bytes Drex receives change. The judge hashes and stores the request it built, and the
journal's sent body records what Drex was actually sent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

BASE64_DATA_URL = re.compile(r"(data)(?=:[^\s,]*;base64,)", re.IGNORECASE)
"""A base64 data URL, which Drex refuses as media: ``data:``, a media type, then ``;base64,``."""

ZERO_WIDTH_SPACE = "\u200b"


def drex_wire(state: Mapping, questions: Mapping) -> tuple[dict, dict]:
    """The state and questions as Drex accepts them; the caller's mappings are left untouched."""
    wire_questions = {name: _with_text_criteria(question) for name, question in questions.items()}
    return _as_text(state), _as_text(wire_questions)


def _with_text_criteria(question: Mapping) -> dict:
    criteria = question.get("criteria")
    if isinstance(criteria, Mapping):
        return {**question, "criteria": {label: _criterion_text(value) for label, value in criteria.items()}}
    if isinstance(criteria, list | tuple):
        return {**question, "criteria": [_criterion_text(value) for value in criteria]}
    return dict(question)


def _criterion_text(value: object) -> str | None:
    """Text and null pass unchanged; a structured criterion becomes its compact JSON text."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _as_text(value):
    """Every string with its base64 data URLs broken; keys, which answers refer to, unchanged."""
    if isinstance(value, str):
        return BASE64_DATA_URL.sub(rf"\1{ZERO_WIDTH_SPACE}", value)
    if isinstance(value, Mapping):
        return {key: _as_text(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_as_text(item) for item in value]
    return value
