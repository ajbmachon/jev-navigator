"""Every exception JVN defines is a refusal the CLI reports in one line, or a listed internal one."""

import importlib
import inspect
import pkgutil

import jev_navigator
from jev_navigator.errors import JvnRefusal

# Exception classes that never reach a caller as a refusal, each with the reason.
INTERNAL_ERRORS = {
    "jev_navigator.judgments.journal.AttemptJournalCallbackError": "the judge raises the original instead",
    "jev_navigator.llm_step.ReplyParseError": "the LLM step catches it and asks again",
    "jev_navigator.history.UnknownSectionError": "code named a section it never declared, which is a bug",
}


def _package_exception_classes() -> dict[str, type[BaseException]]:
    classes = {}
    for module_info in pkgutil.walk_packages(jev_navigator.__path__, "jev_navigator."):
        module = importlib.import_module(module_info.name)
        for name, value in vars(module).items():
            if (
                inspect.isclass(value)
                and issubclass(value, BaseException)
                and value.__module__ == module.__name__
            ):
                classes[f"{module.__name__}.{name}"] = value
    return classes


def test_every_exception_class_is_a_named_refusal_or_listed_as_internal() -> None:
    classes = _package_exception_classes()

    unclassified = [
        name
        for name, cls in classes.items()
        if not issubclass(cls, JvnRefusal) and name not in INTERNAL_ERRORS
    ]

    assert unclassified == []


def test_the_internal_list_names_only_existing_classes_that_are_not_refusals() -> None:
    classes = _package_exception_classes()

    stale = [name for name in INTERNAL_ERRORS if name not in classes or issubclass(classes[name], JvnRefusal)]

    assert stale == []
