"""Persistent syntax facts keyed only by source bytes and parser/rule identity."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict
from functools import cache
from pathlib import Path

from ..cache_root import cache_root
from ..confirmation import day_of, today
from . import imports, languages, scope_scan, spans, tools
from .languages import FLOW_LANGUAGE, parse_language, sgconfig_of
from .scope_scan import READ_AGAIN_AS_FLOW, CallMatch, FileFacts, FileStructure, ReferenceMatch, fact_rules
from .spans import Span
from .tools import ast_grep_version

_MODULES_THAT_READ_MATCHES = (scope_scan, languages, imports, spans, tools)


class FactCache:
    """Facts keyed by the file's bytes, its language, the ast-grep version, the rule text a scan of
    that language sends, and the source of the code that runs the parser and turns matches into
    facts. Any change to one of them is a cache miss, so no version string needs bumping by hand.

    An entry sits at ``<root>/<language>/<identity>/<sha256 of the bytes>.json``. The identity hashes
    the ast-grep version and the language's rules identity, so the entries one JVN version reads
    share one identity folder, and a folder no version reads any more is dead as a whole. A load
    stamps two modification times: its identity folder's, the first time this cache reads from it
    (the folder's last use), and the entry's, once a day (the entry's last confirmation against a
    real file)."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or user_fact_cache()
        self.parser = ast_grep_version()
        self._used: set[Path] = set()

    def load(self, file: str, content: bytes) -> FileFacts | None:
        path = self._path(file, content)
        self._mark_used(path.parent)
        try:
            with path.open("rb") as entry:
                raw = json.loads(entry.read())
                modified = os.fstat(entry.fileno()).st_mtime
            facts = _decode(file, raw)
        except (OSError, KeyError, TypeError, ValueError):
            return None
        _confirm(path, modified)
        return facts

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

    def current_folders(self) -> list[Path]:
        """The identity folder on disk that this JVN reads, for each language folder."""
        return sorted(
            folder
            for language in self._language_folders()
            if (folder := language / self._identity(language.name)).is_dir()
        )

    def retired(self) -> list[Path]:
        """Everything under the root this JVN never reads: other versions' identity folders, and
        files left by a layout without identity folders. Each one's modification time is its last use."""
        current = set(self.current_folders())
        loose = [path for path in _children(self.root) if not path.is_dir()]
        held = [path for language in self._language_folders() for path in _children(language)]
        return [*loose, *(path for path in held if path not in current)]

    def _language_folders(self) -> list[Path]:
        return [path for path in _children(self.root) if path.is_dir()]

    def _path(self, file: str, content: bytes) -> Path:
        language = parse_language(file, content) or "text"
        key = hashlib.sha256(content).hexdigest()
        return self.root / language / self._identity(language) / f"{key}.json"

    def _identity(self, language: str) -> str:
        return hashlib.sha256(f"{self.parser}\0{_rules_identity(language)}".encode()).hexdigest()

    def _mark_used(self, folder: Path) -> None:
        if folder in self._used:
            return
        self._used.add(folder)
        with suppress(OSError):
            os.utime(folder)


def _confirm(entry: Path, modified: float) -> None:
    """A stamp is evidence for housekeeping, never a condition of serving: a cache that refuses it
    (gone meanwhile, read-only, another user's) still serves the facts."""
    if day_of(modified) < today():
        with suppress(OSError):
            os.utime(entry)


def _children(folder: Path) -> list[Path]:
    try:
        return list(folder.iterdir())
    except FileNotFoundError:
        return []


def user_fact_cache() -> Path:
    """The fact cache every index on this machine shares."""
    return cache_root() / "facts"


def facts_identity() -> str:
    """One identity for the facts of every language: the parser version and each language's rules
    and match-reading code. It changes whenever any language's cached facts would."""
    languages_parsed = sorted({*languages.FUNCTION_KINDS, FLOW_LANGUAGE})
    parts = [ast_grep_version(), *(_rules_identity(language) for language in languages_parsed)]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


@cache
def _rules_identity(language: str) -> str:
    """Computed once per language and process: the rules and the code that reads matches do not
    change while it runs. JavaScript facts may come from the flow rules too, so those count for it.
    A test that patches a rule clears it with ``_rules_identity.cache_clear``."""
    rules = fact_rules([language]) if language in languages.FUNCTION_KINDS else ""
    config = sgconfig_of(language) or ""
    if language == READ_AGAIN_AS_FLOW:
        rules, config = f"{rules}\0{fact_rules([FLOW_LANGUAGE])}", sgconfig_of(FLOW_LANGUAGE)
    return hashlib.sha256(f"{rules}\0{config}\0{_match_reader_source()}".encode()).hexdigest()


@cache
def _match_reader_source() -> str:
    """The source of the modules that build the rules, run the parser and read its matches, and of
    this one."""
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
            "decorated": [list(decorated) for decorated in facts.structure.decorated],
            "stubs": [list(stub) for stub in facts.structure.stubs],
        },
        "calls": [asdict(call) for call in facts.calls],
        "references": [asdict(reference) for reference in facts.references],
        "incomplete": facts.incomplete,
        "export_names": list(facts.export_names),
        "unparsed_lines": [list(stretch) for stretch in facts.unparsed_lines],
        "language": facts.language,
    }


def _decode(file: str, raw: dict) -> FileFacts:
    structure = raw["structure"]
    return FileFacts(
        FileStructure(
            tuple(_span(file, span) for span in structure["functions"]),
            tuple(_span(file, span) for span in structure["symbols"]),
            tuple(_span(file, span) for span in structure["declarations"]),
            tuple((int(start), int(end), int(line)) for start, end, line in structure["decorated"]),
            tuple((int(start), int(end)) for start, end in structure["stubs"]),
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
        language=raw["language"],
    )
