"""Persist mixed dataclass and typed-plan receipts through their JSON contract."""

import json
from dataclasses import replace
from pathlib import Path

import msgspec


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(msgspec.to_builtins(value), indent=2, ensure_ascii=False) + "\n")


def merge_candidates(candidates, result):
    for candidate in result.candidates:
        prior = candidates.get(candidate.unit.id)
        candidates[candidate.unit.id] = (
            candidate
            if prior is None
            else replace(
                prior,
                approaches=tuple(dict.fromkeys((*prior.approaches, *candidate.approaches))),
                reaches=tuple(dict.fromkeys((*prior.reaches, *candidate.reaches))),
            )
        )
