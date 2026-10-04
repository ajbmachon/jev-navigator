"""Keep secrets out of every Jev request: mask what is found, then scan the final request and refuse on a hit.

Masking works by content: every value the masker hides anywhere in a request is hidden everywhere
in it, so a token found in an assignment is also hidden where a relation text or another item quotes
it. The built-in masker and scanner are lightweight and on by default. A host with a stronger scanner
passes its own objects; turning either off must be explicit (``masker=None`` or ``scanner=None``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Protocol

from ..errors import JvnRefusal

MASK = "[MASKED]"
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----|\Z)", re.S
)
_KEY_MARKER_LINE = re.compile(r"^.*-----(?:BEGIN|END) [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*$", re.M)
_TOKEN_SHAPES = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
)
_SECRET_ASSIGNMENT = re.compile(
    r"""(?i)\b[\w.-]*(?:secret|token|password|passwd|pwd|api[_-]?key|access[_-]?key|private[_-]?key|credential)"""
    r"""[\w.-]*\s*[:=]\s*["']([^"'\s]{8,})["']"""
)
_QUOTED_ASSIGNMENT = re.compile(r"""[:=]\s*["']([A-Za-z0-9+/=_\-]{20,})["']""")
HIGH_ENTROPY_BITS_PER_CHAR = 4.0


class Masker(Protocol):
    """``mask`` hides secrets in one text; ``masked_values`` lists the values it hides there, so they
    can be hidden everywhere else in the request too."""

    def mask(self, text: str) -> str: ...

    def masked_values(self, text: str) -> list[str]: ...


class Scanner(Protocol):
    def findings(self, text: str) -> list[str]: ...


class SecretInRequestError(JvnRefusal, RuntimeError):
    """The final pre-send scan found a secret; the request was not sent."""


@dataclass(frozen=True)
class SecretMasker:
    """Private-key blocks with their BEGIN and END lines, common token shapes, secret-named
    assignments, and high-entropy quoted values in assignments."""

    def mask(self, text: str) -> str:
        text = _PRIVATE_KEY_BLOCK.sub(MASK, text)
        text = _KEY_MARKER_LINE.sub(MASK, text)
        for shape in _TOKEN_SHAPES:
            text = shape.sub(MASK, text)
        text = _SECRET_ASSIGNMENT.sub(_mask_group, text)
        return _QUOTED_ASSIGNMENT.sub(_mask_if_high_entropy, text)

    def masked_values(self, text: str) -> list[str]:
        found = [match.group(0) for match in _PRIVATE_KEY_BLOCK.finditer(text)]
        found += [match.group(0) for match in _KEY_MARKER_LINE.finditer(text)]
        found += [match.group(0) for shape in _TOKEN_SHAPES for match in shape.finditer(text)]
        found += [match.group(1) for match in _SECRET_ASSIGNMENT.finditer(text)]
        found += [
            match.group(1) for match in _QUOTED_ASSIGNMENT.finditer(text) if _is_high_entropy(match.group(1))
        ]
        return [value for value in found if value != MASK]


@dataclass(frozen=True)
class SecretScanner:
    """Reports what the built-in masker would have masked; used as the final check before sending."""

    def findings(self, text: str) -> list[str]:
        found = [match.group(0)[:12] for match in _KEY_MARKER_LINE.finditer(text)]
        found += [match.group(0)[:8] for shape in _TOKEN_SHAPES for match in shape.finditer(text)]
        found += [match.group(1)[:4] for match in _SECRET_ASSIGNMENT.finditer(text) if match.group(1) != MASK]
        found += [
            match.group(1)[:4]
            for match in _QUOTED_ASSIGNMENT.finditer(text)
            if _is_high_entropy(match.group(1))
        ]
        return found


def mask_request(state: Mapping, questions: Mapping, masker: Masker) -> tuple[Mapping, Mapping, frozenset]:
    """The masked state and questions, and every value that was hidden in either."""
    values = masked_values([state, questions], masker)
    return mask_everywhere(state, masker, values), mask_everywhere(questions, masker, values), values


def mask_by_content(value: object, masker: Masker) -> object:
    """Masks nested JSON-like data so that a value hidden in one string is hidden in all of them."""
    return mask_everywhere(value, masker, masked_values(value, masker))


def masked_values(value: object, masker: Masker) -> frozenset[str]:
    """Every value the masker hides anywhere inside nested JSON-like data, keys included."""
    return frozenset(found for text in dict.fromkeys(_strings(value)) for found in masker.masked_values(text))


def mask_everywhere(value: object, masker: Masker, values: frozenset[str]) -> object:
    """Masks every string by the masker's rules, then hides each of ``values`` wherever it still
    appears. Keys are left as they are; ``refuse_if_secret`` refuses a request with one in a key."""
    longest_first = sorted(values - {MASK}, key=len, reverse=True)

    @cache
    def hide(text: str) -> str:
        text = masker.mask(text)
        for secret in longest_first:
            text = text.replace(secret, MASK)
        return text

    return _each_string(value, hide)


def safe_options(options: Mapping[str, str], masker: Masker | None) -> dict[str, str]:
    """Options whose key would change under masking are dropped: a secret is never offered as a choice."""
    if masker is None:
        return dict(options)
    return {key: masker.mask(text) for key, text in options.items() if masker.mask(key) == key}


def refuse_if_secret(
    state: Mapping, questions: Mapping, scanner: Scanner | None, masked: frozenset[str] = frozenset()
) -> None:
    """Refuses when a value masked elsewhere is still in the request (it can only sit in a key), or
    when the scanner finds a secret."""
    texts = _strings(state) + _strings(questions)
    if any(value in text for text in texts for value in masked):
        raise SecretInRequestError("a masked value is still in the request, in a key; nothing was sent")
    if scanner is None:
        return
    for text in texts:
        if scanner.findings(text):
            raise SecretInRequestError("the final scan found a secret in the request; nothing was sent")


def _each_string(value: object, change: Callable[[str], str]) -> object:
    if isinstance(value, str):
        return change(value)
    if isinstance(value, Mapping):
        return {key: _each_string(item, change) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_each_string(item, change) for item in value]
    return value


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for key, item in value.items() for text in [str(key), *_strings(item)]]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _strings(item)]
    return []


def _mask_group(match: re.Match) -> str:
    return match.group(0).replace(match.group(1), MASK)


def _mask_if_high_entropy(match: re.Match) -> str:
    return _mask_group(match) if _is_high_entropy(match.group(1)) else match.group(0)


def _is_high_entropy(value: str) -> bool:
    counts = Counter(value)
    bits = -sum(count / len(value) * math.log2(count / len(value)) for count in counts.values())
    return bits >= HIGH_ENTROPY_BITS_PER_CHAR
