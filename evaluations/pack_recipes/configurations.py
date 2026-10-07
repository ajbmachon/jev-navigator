"""Evidence-pack recipes. Domain composition lives outside the JVN library."""

from dataclasses import dataclass
from typing import ClassVar

from jev_navigator.index.units import LineAnchor
from jev_navigator.recipe_blocks import OWNER_DEFINITIONS, ChainAttachment, guard_chain, value_source
from jev_navigator.recipe_configuration import SearchRecipe
from jev_navigator.sources import (
    ANCHORS,
    CALLEES,
    CALLERS,
    FILES,
    NAMED_FILES,
    TEXT_NAMED_FILES,
    Reach,
    function_span,
)


@dataclass(frozen=True)
class GuardSource:
    depth: int = 2
    attachments: tuple[ChainAttachment, ...] = ()
    name: ClassVar[str] = "guard_chain"
    label: ClassVar[str] = "downward entry chain"

    def reach(self, index, seeds):
        for unit in seeds.units:
            entry = function_span(index, unit)
            if entry is None:
                continue
            chain = guard_chain(index, entry, depth=self.depth, attachments=self.attachments)
            for span in chain.units:
                yield Reach(LineAnchor(span.file, span.start), self.name, entry.key, 1)
            for link in chain.links:
                yield Reach(link.at, self.name, entry.key, 1)


@dataclass(frozen=True)
class ValueSource:
    name: ClassVar[str] = "value_source"
    label: ClassVar[str] = "setting sources"

    def reach(self, index, seeds):
        # Only input names that look like explicit setting keys. No label terms.
        for name in seeds.names:
            if name.isupper() or "." in name or "_" in name:
                yield from value_source(index, name)


LOCAL = SearchRecipe("pack-local", 1, (ANCHORS, CALLEES, FILES))
NAMED = SearchRecipe(
    "pack-named", 1, (ANCHORS, OWNER_DEFINITIONS, NAMED_FILES, TEXT_NAMED_FILES), (CALLEES, CALLERS)
)
CONVENTION = SearchRecipe(
    "pack-convention", 1, (ANCHORS, GuardSource(), ValueSource(), NAMED_FILES, TEXT_NAMED_FILES, FILES)
)
RECIPES = (LOCAL, NAMED, CONVENTION)

# Category priors describe which blocks help; they do not assert that every
# member of a category is local or needs an absence proof.
GUARD_CATEGORIES = {"authorization", "identity", "input", "injection", "paths", "trust", "consent"}
VALUE_CATEGORIES = {"configuration", "secrets", "values", "keys", "storage-version"}
PRESENCE_CATEGORIES = {
    "authorization",
    "identity",
    "input",
    "consent",
    "retention",
    "ai-governance",
    "audit",
    "licenses",
    "test-reach",
    "agent-map",
    "repo-state",
}


def category_recipes(category: str) -> tuple[str, ...]:
    """Priors are capabilities, not a category-only decision about a finding's locality."""
    if category in GUARD_CATEGORIES | VALUE_CATEGORIES | PRESENCE_CATEGORIES:
        return (LOCAL.name, NAMED.name, CONVENTION.name)
    return (LOCAL.name, NAMED.name)


def primary_recipe(statement: str, names: tuple[str, ...]) -> SearchRecipe:
    """Frozen lexical dispatch, independent of deciding labels and stored role scores."""
    text = statement.casefold()
    absence = any(
        term in text
        for term in (
            "missing",
            "never checks",
            "no validation",
            "no guard",
            "without checking",
            "does not enforce",
            "nothing checks",
            "not validated",
            "no check",
            "defaults to",
            "environment",
            "runtime version",
        )
    )
    if absence:
        return CONVENTION
    if names:
        return NAMED
    return LOCAL
