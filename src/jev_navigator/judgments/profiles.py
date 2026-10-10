"""The admitted local-match ranking question, its existence form over a set, and the role wording owner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files

from .questions import Check, Criterion, content_hash

ROLE_QUESTIONS = json.loads(files(__package__).joinpath("role_questions.json").read_text())


def bind_question(value, item: str, point: str):
    """Bind state references without changing the admitted wording or criteria."""
    if isinstance(value, str):
        return value.replace("`unit", f"`{item}").replace("`point`", f"`{point}`")
    if isinstance(value, dict):
        return {key: bind_question(part, item, point) for key, part in value.items()}
    if isinstance(value, list):
        return [bind_question(part, item, point) for part in value]
    return value


@dataclass(frozen=True)
class QuestionProfile:
    """Question templates and their versioned meaning, without ranking composition."""

    name: str
    templates: Mapping

    def questions(self, target: str) -> tuple[Check, ...]:
        checks = []
        for name, template in self.templates.items():
            question = bind_question(template, "{item}", f"targets.{target}")
            criteria = question["criteria"]
            checks.append(
                Check(
                    f"{name}_{target}",
                    question["instructions"],
                    yes=Criterion(**criteria["true"]),
                    no=Criterion(**criteria["false"]),
                )
            )
        return tuple(checks)

    def asked(self, targets: Mapping[str, str]) -> dict:
        """Checks by name, with their template name and target for answer collection."""
        return {
            check.name: (check, check.name.removesuffix(f"_{target}"), target)
            for target in targets
            for check in self.questions(target)
        }

    def identity(self) -> str:
        return f"{self.name}@{content_hash(self.templates)}"


J1 = QuestionProfile("J1-3", json.loads(files(__package__).joinpath("local_match_question.json").read_text()))
EXISTS = QuestionProfile(
    "exists-1", json.loads(files(__package__).joinpath("existence_question.json").read_text())
)
"""Whether a set of supplied entries holds the code that settles a part of a point: J1-3's local-match
property asked once over the set (``fetched``) instead of once per unit, so a ranking says where to look
and this says whether the shortlist answers the point."""
