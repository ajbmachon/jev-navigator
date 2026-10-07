"""What a text mentions: the paths, the modules a command runs, and the code names it spells out. This
module is the one owner of whether a token in a text is a path, and of how a text names code."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from operator import itemgetter

from .index.languages import LANGUAGE_BY_SUFFIX, SCHEMA_SUFFIX

PATH_TOKEN = re.compile(r"[\w./-]*[\w-]\.[A-Za-z0-9]{1,8}")
"""A file name with a suffix of one to eight letters or digits, perhaps under folders: ``ci.yml``,
``jobs/sweep.py``. A word without a suffix, such as ``Makefile``, is not a path token."""
_LEADING_RELATIVE = re.compile(r"^(?:\.{1,2}/|/)+")
_TEXT_FILE_SUFFIXES = (
    "md", "mdx", "markdown", "yml", "yaml", "json", "toml", "ini", "cfg", "conf", "env", "lock",
    "txt", "sql", "html", "css", "xml", "csv", "svg", "sh",
)  # fmt: skip
_FILE_SUFFIXES = sorted(
    {*(suffix.removeprefix(".") for suffix in (*LANGUAGE_BY_SUFFIX, SCHEMA_SUFFIX)), *_TEXT_FILE_SUFFIXES}
)
_FILE_PATH = re.compile(rf"/|\.(?:{'|'.join(_FILE_SUFFIXES)})$")
_BACKTICKED = re.compile(r"`([^`]{2,80})`")
_CALL = re.compile(r"\b([A-Za-z_$][\w$.]*)\(")
_CAMEL_CASE = re.compile(r"\b([a-z]+[A-Z][\w$]*|[A-Z][a-z0-9]+[A-Z][\w$]*)\b")
_SCREAMING_CASE = re.compile(r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b")
_SNAKE_CASE = re.compile(r"\b([a-z]+(?:_[a-z0-9]+)+)\b")
BARE_NAME_RULES = (_CALL, _CAMEL_CASE, _SCREAMING_CASE, _SNAKE_CASE)
"""The shapes of a word a text names code by without backticks; snake_case is its own rule."""
_DOTTED_MODULE = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
_INTERPRETER_OPTION = r"-(?:[WX]\s*[^\s-]\S*|[A-Za-z]\w*)"
"""An option before ``-m``: ``-u``, or ``-W`` and ``-X`` with their value (``-W ignore``, ``-X dev``)."""
_PYTHON_MODULE_RUN = re.compile(rf"\bpython[\d.]*(?:\s+{_INTERPRETER_OPTION})*\s+-m\s*({_DOTTED_MODULE})")
_IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")
_MEMBER_SEPARATOR = re.compile(r"[.#:]")
_SYMBOL_SEPARATOR = re.compile(r"[/,|]")


def paths_in(text: str) -> list[str]:
    """The path tokens of ``text`` in order of first mention, each once and without a leading
    ``./``, ``../`` or ``/``: ``../config/app.toml`` gives ``config/app.toml``."""
    tokens = (_LEADING_RELATIVE.sub("", token) for token in PATH_TOKEN.findall(text))
    return list(dict.fromkeys(token for token in tokens if token))


def python_modules_in(text: str) -> list[str]:
    """The modules ``text`` runs with ``python -m``, each once, in order of first mention:
    ``uv run python3 -u -m app.jobs --all`` and ``python -W ignore -mapp.jobs`` run ``app.jobs``. A
    dotted word no command runs is none."""
    return list(dict.fromkeys(_PYTHON_MODULE_RUN.findall(text)))


def is_file_path(span: str) -> bool:
    """Whether ``span`` names a file rather than code: it holds a ``/`` or ends in a file suffix JVN
    knows (``pyproject.toml``, ``res.json``). A member call such as ``res.json()`` names code."""
    return _FILE_PATH.search(span) is not None


def code_names_in(text: str) -> list[str]:
    """The code names ``text`` spells out, each once, in order of first mention: backticked spans
    that are not file paths, and words shaped as a call, camelCase or PascalCase, SCREAMING_CASE or
    snake_case. A dotted or qualified name gives its last member: ``Store.flush()`` names ``flush``.
    The word rules read a path too, so a path whose stem has such a shape names it, as a file often
    defines its stem: ``jobs/run_all.py`` names ``run_all``, while ``pyproject.toml`` names nothing."""
    mentions = sorted([*_backticked_names(text), *_bare_names(text)], key=itemgetter(0))
    return list(dict.fromkeys(name for _, name in mentions))


def member_names(symbol: str) -> list[str]:
    """The members a symbol list names, each once: ``Store.flush() / cache#evict`` names ``flush``
    and ``evict``. A piece holding a space is a description and names none."""
    pieces = (piece.strip() for piece in _SYMBOL_SEPARATOR.split(symbol))
    names = (_member(piece) for piece in pieces if piece and " " not in piece)
    return list(dict.fromkeys(name for name in names if name))


def _backticked_names(text: str) -> Iterator[tuple[int, str]]:
    for match in _BACKTICKED.finditer(text):
        if not is_file_path(match[1]) and (name := _member(match[1])):
            yield match.start(), name


def _bare_names(text: str) -> Iterator[tuple[int, str]]:
    for rule in BARE_NAME_RULES:
        for match in rule.finditer(text):
            if name := _member(match[1]):
                yield match.start(1), name


def _member(span: str) -> str | None:
    """The last member of a dotted or qualified name, when it is an identifier."""
    name = _MEMBER_SEPARATOR.split(span.strip().removesuffix("()"))[-1].strip()
    return name if _IDENTIFIER.fullmatch(name) else None


@dataclass(frozen=True)
class TextNames:
    """Code names and whole paths extracted without a model. Paths are never reduced to suffixes."""

    code: tuple[str, ...]
    paths: tuple[str, ...]


def names_from_text(text: str) -> TextNames:
    """Extract spelled identifiers and paths using the shared mention rules."""
    return TextNames(tuple(code_names_in(text)), tuple(paths_in(text)))


def literal_names_in(text: str) -> tuple[str, ...]:
    """Exact quoted keys, identifiers and paths, without generating domain vocabulary."""
    literals = re.findall(r"[\"']([A-Za-z_$][\w$./:#-]{2,120})[\"']", text)
    return tuple(dict.fromkeys(literals))


def spelling_variants(name: str) -> tuple[str, ...]:
    """Bounded case, separator and identifier-component spellings of an existing name."""
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    parts = re.split(r"[_./:#-]+", separated)
    snake = "_".join(part.lower() for part in parts)
    camel = parts[0].lower() + "".join(part.title() for part in parts[1:])
    variants = (name, snake, camel, snake.upper(), *(part.lower() for part in parts if len(part) >= 3))
    return tuple(dict.fromkeys(variant for variant in variants if variant))
