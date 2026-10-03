"""Persistent syntax facts keyed only by source bytes and parser/rule identity."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from functools import cache
from pathlib import Path

from ..cache_root import cache_root
from . import imports, languages, scope_scan, spans
from .languages import FLOW_LANGUAGE, FLOW_SGCONFIG, parse_language
from .scope_scan import CallMatch, FileFacts, FileStructure, ReferenceMatch, fact_rules
from .spans import Span
from .tools import ast_grep_version

_MODULES_THAT_READ_MATCHES = (scope_scan, languages, imports, spans)


class FactCache:
    """Facts keyed by the file's bytes, its language, the ast-grep version, the rule text a scan of
    that language sends, and the source of the code that turns matches into facts. Any change to
    one of them is a cache miss, so no version string needs bumping by hand."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or user_fact_cache()
        self.parser = ast_grep_version()

    def load(self, file: str, content: bytes) -> FileFacts | None:
        path = self._path(file, content)
        try:
            raw = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            return None
        try:
            return _decode(file, raw)
        except (KeyError, TypeError, ValueError):
            return None

    def save(self, file: str, content: bytes, facts: FileFacts) -> None:
        path = self._path(file, content)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary.write(json.dumps(_encode(facts), sort_keys=True, separators=(",", ":")))
            temporary_path = Path(temporary.name)
        try:
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _path(self, file: str, content: bytes) -> Path:
        language = parse_language(file, content) or "text"
        identity = hashlib.sha256()
        for part in (content, language.encode(), self.parser.encode(), _rules_identity(language).encode()):
            identity.update(part)
            identity.update(b"\0")
        return self.root / language / f"{identity.hexdigest()}.json"


def user_fact_cache() -> Path:
    """The fact cache every index on this machine shares."""
    return cache_root() / "facts"


@cache
def _rules_identity(language: str) -> str:
    """Computed once per language and process: the rules and the code that reads matches do not
    change while it runs. A test that patches a rule clears it with ``_rules_identity.cache_clear``."""
    rules = fact_rules([language]) if language in languages.FUNCTION_KINDS else ""
    config = FLOW_SGCONFIG if language == FLOW_LANGUAGE else ""
    return hashlib.sha256(f"{rules}\0{config}\0{_match_reader_source()}".encode()).hexdigest()


@cache
def _match_reader_source() -> str:
    """The source of the modules that build the rules and read the matches, and of this one."""
    digest = hashlib.sha256()
    for module_file in (*(module.__file__ for module in _MODULES_THAT_READ_MATCHES), __file__):
        digest.update(Path(module_file).read_bytes())
    return digest.hexdigest()


def _span(file: str, raw: dict) -> Span:
    return Span(file, raw["start"], raw["end"], raw.get("name", ""))


def _encode(facts: FileFacts) -> dict:
    return {
        "structure": {
            "functions": [asdict(span) for span in facts.structure.functions],
            "symbols": [asdict(span) for span in facts.structure.symbols],
            "declarations": [asdict(span) for span in facts.structure.declarations],
        },
        "calls": [asdict(call) for call in facts.calls],
        "references": [asdict(reference) for reference in facts.references],
        "incomplete": facts.incomplete,
        "export_names": list(facts.export_names),
        "unparsed_lines": [list(stretch) for stretch in facts.unparsed_lines],
    }


def _decode(file: str, raw: dict) -> FileFacts:
    structure = raw["structure"]
    return FileFacts(
        FileStructure(
            tuple(_span(file, span) for span in structure["functions"]),
            tuple(_span(file, span) for span in structure["symbols"]),
            tuple(_span(file, span) for span in structure["declarations"]),
        ),
        tuple(CallMatch(file, call["line"], call["name"], call.get("receiver")) for call in raw["calls"]),
        tuple(
            ReferenceMatch(
                file, reference["line"], reference["role"], reference["name"], reference["receiver"]
            )
            for reference in raw["references"]
        ),
        bool(raw["incomplete"]),
        tuple(raw.get("export_names", ())),
        tuple((int(start), int(end)) for start, end in raw["unparsed_lines"]),
    )
