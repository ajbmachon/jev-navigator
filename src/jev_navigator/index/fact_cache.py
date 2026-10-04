"""Persistent syntax facts keyed only by source bytes and parser/rule identity."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from .languages import language_of
from .scope_scan import CallMatch, FileFacts, FileStructure, LocalName, ModuleAlias, ReferenceMatch
from .spans import Span
from .tools import ast_grep_version

FACT_RULE_VERSION = "combined-facts-v28-python-module-aliases"


class FactCache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.home() / ".cache/jev-navigator/facts"
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
        language = language_of(file) or "text"
        identity = hashlib.sha256()
        identity.update(content)
        identity.update(b"\0")
        identity.update(language.encode())
        identity.update(b"\0")
        identity.update(self.parser.encode())
        identity.update(b"\0")
        identity.update(FACT_RULE_VERSION.encode())
        return self.root / language / f"{identity.hexdigest()}.json"


def _span(file: str, raw: dict) -> Span:
    return Span(file, raw["start"], raw["end"], raw.get("name", ""))


def _encode(facts: FileFacts) -> dict:
    return {
        "structure": {
            "functions": [asdict(span) for span in facts.structure.functions],
            "symbols": [asdict(span) for span in facts.structure.symbols],
            "declarations": [asdict(span) for span in facts.structure.declarations],
            "module_symbols": [asdict(span) for span in facts.structure.module_symbols],
            "commonjs_exports": [asdict(span) for span in facts.structure.commonjs_exports],
            "type_declarations": [asdict(span) for span in facts.structure.type_declarations],
            "value_declarations": [asdict(span) for span in facts.structure.value_declarations],
            "local_names": [list(local) for local in facts.structure.local_names],
        },
        "calls": [asdict(call) for call in facts.calls],
        "references": [asdict(reference) for reference in facts.references],
        "incomplete": facts.incomplete,
        "export_names": list(facts.export_names),
        "unparsed_lines": [list(stretch) for stretch in facts.unparsed_lines],
        "module_aliases": [list(alias) for alias in facts.module_aliases],
        "exported_values": list(facts.exported_values),
    }


def _decode(file: str, raw: dict) -> FileFacts:
    structure = raw["structure"]
    return FileFacts(
        FileStructure(
            tuple(_span(file, span) for span in structure["functions"]),
            tuple(_span(file, span) for span in structure["symbols"]),
            tuple(_span(file, span) for span in structure["declarations"]),
            tuple(_span(file, span) for span in structure["module_symbols"]),
            tuple(_span(file, span) for span in structure["commonjs_exports"]),
            tuple(_span(file, span) for span in structure["type_declarations"]),
            tuple(_span(file, span) for span in structure["value_declarations"]),
            tuple(LocalName(int(first), int(last), name) for first, last, name in structure["local_names"]),
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
        tuple(ModuleAlias(name, specifier) for name, specifier in raw["module_aliases"]),
        tuple(raw["exported_values"]),
    )
