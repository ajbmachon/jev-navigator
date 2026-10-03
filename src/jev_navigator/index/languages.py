"""Which ast-grep language parses a file, which syntax nodes are functions, and how to read their names.

A ``.js`` file whose leading comments carry the ``@flow`` pragma parses as ``flow``: flow uses
ast-grep's available ``tsx`` grammar, selected
per scan through a ``languageGlobs`` sgconfig. What that grammar cannot recover still surfaces as
ERROR nodes, so incomplete coverage stays visible."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
}

FUNCTION_KINDS = {
    "python": ("function_definition",),
    "typescript": ("function_declaration", "method_definition", "arrow_function", "function_expression"),
    "tsx": ("function_declaration", "method_definition", "arrow_function", "function_expression"),
    "javascript": ("function_declaration", "method_definition", "arrow_function", "function_expression"),
}

CLASS_KINDS = {
    "python": ("class_definition",),
    "typescript": ("class_declaration",),
    "tsx": ("class_declaration",),
    "javascript": ("class_declaration",),
}

_SCRIPT_DECLARATIONS = """  any:
    - kind: type_alias_declaration
    - kind: interface_declaration
    - kind: enum_declaration
    - kind: lexical_declaration
      inside:
        any:
          - kind: program
          - kind: export_statement"""

DECLARATION_RULES = {
    "python": """  kind: assignment
  inside:
    kind: expression_statement
    inside:
      kind: module""",
    "typescript": _SCRIPT_DECLARATIONS,
    "tsx": _SCRIPT_DECLARATIONS,
    "javascript": """  kind: lexical_declaration
  inside:
    any:
      - kind: program
      - kind: export_statement""",
}

# The installed ast-grep supports tsx but not Flow. Route marked files through tsx;
# unsupported Flow constructs remain visible through ERROR nodes.
FLOW_LANGUAGE = "flow"
FUNCTION_KINDS[FLOW_LANGUAGE] = FUNCTION_KINDS["tsx"]
CLASS_KINDS[FLOW_LANGUAGE] = CLASS_KINDS["tsx"]
DECLARATION_RULES[FLOW_LANGUAGE] = _SCRIPT_DECLARATIONS

# ast-grep reads `languageGlobs` only from a config file: a scan of flow files passes this sgconfig,
# which parses every JavaScript suffix with the tsx grammar. Plain-JS files are scanned in their own
# invocation without it, so their grammar is unchanged.
FLOW_SGCONFIG = 'languageGlobs:\n  tsx:\n    - "*.js"\n    - "*.jsx"\n    - "*.mjs"\n    - "*.cjs"\n'


def grammar_of(language: str) -> str:
    """The ast-grep language whose grammar parses ``language`` (flow rides on the tsx grammar)."""
    return "tsx" if language == FLOW_LANGUAGE else language


_DECLARED_NAME = re.compile(
    r"^\s*(?:export\s+)?(?:declare\s+)?(?:(?:type|interface|enum|const|let|var)\s+)?(\w+)"
)

# A declaration names itself at the start of its own first line.
_HEAD_NAME_PATTERNS = (
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+(\w+)"),
    re.compile(r"^\s*(?:async\s+)?def\s+(\w+)"),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?function\b\s*\*?\s*(\w+)"),
    re.compile(
        r"^\s*(?:(?:public|private|protected|static|override|abstract|async|get|set)\s+)*"
        r"\*?\s*#?(\w+)\s*[<(]"
    ),
)

# An unnamed function takes the name that the code just before it binds it to, opening parentheses
# included (`x = ((a) => a)`): a declared name whatever its type holds, an assigned name or member
# whose type holds no `,` or `;` (so a binding never starts inside an earlier parameter), or a key.
_BINDING_NAME_PATTERNS = (
    re.compile(r"\b(?:const|let|var)\s+(\w+)(?:\s*:.*)?\s*=[\s(]*$"),
    re.compile(r"(\w+)\s*(?::(?:[^=,;]|=>)+)?=[\s(]*$"),
    re.compile(r"(\w+)\??\s*:[\s(]*$"),
)

# The line before names a function that starts its line only when that line starts as a binding
# (`export const render =`), never when it is the end of a type that spans lines.
_BINDING_LINE = re.compile(r"^\s*(?:export\s+)?(?:(?:const|let|var)\s+)?[\w$.]+\??\s*[:=]")


def language_of(path: str) -> str | None:
    return LANGUAGE_BY_SUFFIX.get(PurePosixPath(path).suffix)


def language_for(path: str, lines: Sequence[str] | None = None) -> str | None:
    """The language ``path`` parses as: like ``language_of``, but a JavaScript file whose leading
    comments carry the ``@flow`` pragma parses as ``flow`` (with the tsx grammar)."""
    language = language_of(path)
    if language == "javascript" and lines is not None and has_flow_pragma(lines):
        return FLOW_LANGUAGE
    return language


_FLOW_PRAGMA = re.compile(r"@flow\b")


def has_flow_pragma(lines: Sequence[str]) -> bool:
    """True when a leading comment of the source carries the ``@flow`` pragma. Only comments before
    the first line of code count — never ``@flow`` in a string or the body — and the scan stops at
    that first code line, however far down it sits: leading comments may be arbitrarily long. A
    byte-order mark or shebang may precede the comments."""
    in_block = False
    for index, line in enumerate(lines):
        text = line[1:] if index == 0 and line.startswith("\ufeff") else line
        position = 0
        while True:
            rest = text[position:]
            stripped = rest.lstrip()
            if not stripped:
                break
            position = len(text) - len(stripped)
            if in_block:
                end = stripped.find("*/")
                if _FLOW_PRAGMA.search(stripped if end < 0 else stripped[:end]):
                    return True
                if end < 0:
                    break
                in_block = False
                position += end + 2
            elif stripped.startswith("//"):
                if _FLOW_PRAGMA.search(stripped[2:]):
                    return True
                break
            elif stripped.startswith("/*"):
                in_block = True
                position += 2
            elif index == 0 and stripped.startswith("#!"):
                break
            else:
                return False
    return False


def declared_name(first_line: str) -> str:
    """The name a module-level assignment, constant, type, interface or enum declares."""
    match = _DECLARED_NAME.search(first_line)
    return match.group(1) if match else "<anonymous>"


def function_name(node_text: str, text_before_node: str = "", line_before: str = "") -> str:
    """The name a function or class node declares at the start of its own first line, or else the
    name the code just before the node binds it to (``x = ``, ``x: ``, ``const x = ``): on the node's
    own line, or on the line before when the node starts its line. The node's body never names it:
    a declaration inside the body names only itself, and a call the node is passed to names nothing."""
    head = node_text.split("\n", 1)[0]
    found = _first_name(_HEAD_NAME_PATTERNS, head) or _bound_name(text_before_node, line_before)
    return found or "<anonymous>"


def _bound_name(text_before_node: str, line_before: str) -> str:
    if text_before_node.strip():
        return _first_name(_BINDING_NAME_PATTERNS, text_before_node)
    if _BINDING_LINE.match(line_before):
        return _first_name(_BINDING_NAME_PATTERNS, line_before)
    return ""


def _first_name(patterns: Sequence[re.Pattern[str]], text: str) -> str:
    for pattern in patterns:
        match = pattern.search(text)
        if match and match.group(1) not in _NOT_NAMES:
            return match.group(1)
    return ""


_NOT_NAMES = frozenset({"if", "for", "while", "switch", "catch", "return", "function", "async"})


@dataclass(frozen=True)
class ReferenceRole:
    """An identifier of ``kind`` has the role ``name`` when a node listed in ``inside`` holds it and
    no node listed in ``not_inside`` does, and its text does not match ``not_regex``. Each listed node
    is an ast-grep relational rule about the direct parent, unless it sets its own ``stopBy``."""

    name: str
    inside: tuple[str, ...] = ()
    kind: str = "identifier"
    not_inside: tuple[str, ...] = ()
    not_regex: str = ""


_PYTHON_SUPERCLASSES = (
    "kind: argument_list\ninside:\n  stopBy: neighbor\n  kind: class_definition\n  field: superclasses"
)

_PYTHON_ROLES = (
    ReferenceRole(
        "argument",
        ("kind: argument_list", "kind: keyword_argument\nfield: value"),
        not_inside=(_PYTHON_SUPERCLASSES,),
    ),
    ReferenceRole(
        "argument",
        ("kind: argument_list", "kind: keyword_argument\nfield: value"),
        kind="attribute",
    ),
    ReferenceRole("decorator", ("kind: decorator",)),
    ReferenceRole("collection", ("kind: pair\nfield: value", "kind: list", "kind: tuple", "kind: set")),
    ReferenceRole("assignment", ("kind: assignment\nfield: right",)),
    ReferenceRole("return", ("kind: return_statement",)),
    ReferenceRole("receiver", ("kind: attribute\nfield: object",), not_regex="^(self|cls)$"),
    ReferenceRole("type", ("kind: type\nstopBy: end",)),
    ReferenceRole("base", (_PYTHON_SUPERCLASSES,)),
    ReferenceRole(
        "condition",
        (
            "kind: comparison_operator",
            "kind: boolean_operator",
            "kind: not_operator",
            "kind: assert_statement",
            "field: condition\nany:\n  - kind: if_statement\n  - kind: elif_clause\n"
            "  - kind: while_statement",
        ),
    ),
)

_COMPARISON_OR_LOGIC = r"^(===|!==|==|!=|<|>|<=|>=|&&|\|\||\?\?|instanceof|in)$"

_SCRIPT_ROLES = (
    ReferenceRole("argument", ("kind: arguments",)),
    ReferenceRole("argument", ("kind: arguments",), kind="member_expression"),
    ReferenceRole("decorator", ("kind: decorator",)),
    ReferenceRole("collection", ("kind: pair\nfield: value", "kind: array")),
    ReferenceRole("collection", kind="shorthand_property_identifier"),
    ReferenceRole(
        "assignment", ("kind: variable_declarator\nfield: value", "kind: assignment_expression\nfield: right")
    ),
    ReferenceRole("export", ("kind: export_specifier", "kind: export_statement")),
    ReferenceRole("return", ("kind: return_statement",)),
    ReferenceRole("receiver", ("kind: member_expression\nfield: object",)),
    ReferenceRole("base", ("kind: class_heritage",)),
    ReferenceRole(
        "condition",
        (
            f"kind: binary_expression\nhas:\n  field: operator\n  regex: {_COMPARISON_OR_LOGIC}",
            "kind: unary_expression\nhas:\n  field: operator\n  regex: ^!$",
            "kind: ternary_expression\nfield: condition",
            "kind: parenthesized_expression\ninside:\n  stopBy: neighbor\n  field: condition\n  any:\n"
            "    - kind: if_statement\n    - kind: while_statement\n    - kind: do_statement",
        ),
    ),
)

_TYPED_SCRIPT_ROLES = (
    *_SCRIPT_ROLES,
    ReferenceRole("base", ("kind: extends_clause",)),
    ReferenceRole(
        "type",
        kind="type_identifier",
        not_inside=(
            "field: name\nany:\n  - kind: interface_declaration\n  - kind: type_alias_declaration\n"
            "  - kind: class_declaration\n  - kind: abstract_class_declaration\n  - kind: type_parameter",
        ),
    ),
)

REFERENCE_ROLES = {
    "python": _PYTHON_ROLES,
    "typescript": _TYPED_SCRIPT_ROLES,
    "tsx": _TYPED_SCRIPT_ROLES,
    "javascript": _SCRIPT_ROLES,
}
REFERENCE_ROLES[FLOW_LANGUAGE] = _TYPED_SCRIPT_ROLES


def export_rules(languages: Iterable[str]) -> str:
    """ast-grep rules for the script export surface: statement nodes, and the ``{ ... }`` clause
    specifiers that carry aliased names. Python has no such kinds, so it contributes no rules."""
    documents = []
    for language in languages:
        if language == "python":
            continue
        grammar = grammar_of(language)
        documents.append(f"id: export_surface\nlanguage: {grammar}\nrule:\n  kind: export_statement")
        documents.append(f"id: export_specifier\nlanguage: {grammar}\nrule:\n  kind: export_specifier")
    return "\n---\n".join(documents)


def reference_rules(languages: Iterable[str]) -> str:
    """ast-grep rules, one per listed language and role, matching every identifier that has that role.
    Calls and imports have no role, so they never match."""
    return "\n---\n".join(
        _role_rule(language, role) for language in languages for role in REFERENCE_ROLES[language]
    )


def _role_rule(language: str, role: ReferenceRole) -> str:
    lines = [f"id: {role.name}", f"language: {grammar_of(language)}", "rule:", f"  kind: {role.kind}"]
    if role.inside:
        lines += ["  any:", *_inside_entries(role.inside, "    ")]
    exclusions = [f"      - regex: {role.not_regex}"] if role.not_regex else []
    exclusions += _inside_entries(role.not_inside, "      ")
    if exclusions:
        lines += ["  not:", "    any:", *exclusions]
    return "\n".join(lines)


def _inside_entries(parents: tuple[str, ...], indent: str) -> list[str]:
    lines = []
    for parent in parents:
        lines.append(f"{indent}- inside:")
        if not any(line.startswith("stopBy:") for line in parent.split("\n")):
            lines.append(f"{indent}    stopBy: neighbor")
        lines += [f"{indent}    {line}" for line in parent.split("\n")]
    return lines
