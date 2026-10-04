"""Which ast-grep language parses a file, which syntax nodes are functions, and how to read their names.

A ``.js`` file whose leading comments carry the ``@flow`` pragma parses as ``flow``: flow uses
ast-grep's available ``tsx`` grammar, selected
per scan through a ``languageGlobs`` sgconfig. What that grammar cannot recover still surfaces as
ERROR nodes, so incomplete coverage stays visible."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

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

_SCRIPT_FUNCTIONS = (
    "function_declaration",
    "generator_function_declaration",
    "method_definition",
    "arrow_function",
    "function_expression",
    "generator_function",
)

FUNCTION_KINDS = {
    "python": ("function_definition",),
    "typescript": _SCRIPT_FUNCTIONS,
    "tsx": _SCRIPT_FUNCTIONS,
    "javascript": _SCRIPT_FUNCTIONS,
}

CLASS_KINDS = {
    "python": ("class_definition",),
    "typescript": ("class_declaration", "abstract_class_declaration", "class"),
    "tsx": ("class_declaration", "abstract_class_declaration", "class"),
    "javascript": ("class_declaration", "class"),
}

# A function or class expression is called by the name that holds it: `const save = function
# inner() {}` is called as `save`. So an expression is named by the declarator, class field, object
# key or assignment it is the value of, looking through the grammar's parentheses and type casts,
# and only then by its own name. A callback passed as an argument, `it("works", () => ...)`, is held
# by no name and stays anonymous. Each grammar lists only the node kinds it has: one unknown kind
# makes ast-grep reject the whole scan. The tables are keyed by grammar, see ``grammar_of``.
EXPRESSION_KINDS = frozenset({"arrow_function", "function_expression", "generator_function", "class"})
_SCRIPT_HOLDERS = (
    ("variable_declarator", "name"),
    ("public_field_definition", "name"),
    ("pair", "key"),
    ("assignment_expression", "left"),
)
NAME_HOLDERS = {
    "python": (),
    "typescript": _SCRIPT_HOLDERS,
    "tsx": _SCRIPT_HOLDERS,
    "javascript": (
        ("variable_declarator", "name"),
        ("field_definition", "property"),
        ("pair", "key"),
        ("assignment_expression", "left"),
    ),
}
_TSX_WRAPPERS = ("parenthesized_expression", "as_expression", "satisfies_expression", "non_null_expression")
NAME_WRAPPERS = {
    "python": (),
    "typescript": (*_TSX_WRAPPERS, "type_assertion"),
    "tsx": _TSX_WRAPPERS,
    "javascript": ("parenthesized_expression",),
}

# ast-grep prints every node a rule's relations match, so asking whether a declaration sits in the
# program printed the whole file once per declaration. Under a double negation the condition holds
# the same and only the declaration is printed.
_MODULE_VARIABLES = (
    "{kind: lexical_declaration, not: {not: {inside: {any: [{kind: program}, {kind: export_statement}]}}}}"
)
_SCRIPT_DECLARATIONS = f"""  any:
    - kind: type_alias_declaration
    - kind: interface_declaration
    - kind: enum_declaration
    - {_MODULE_VARIABLES}"""

DECLARATION_RULES = {
    "python": """  kind: assignment
  not: {not: {inside: {kind: expression_statement, inside: {kind: module}}}}""",
    "typescript": _SCRIPT_DECLARATIONS,
    "tsx": _SCRIPT_DECLARATIONS,
    "javascript": f"  any: [{_MODULE_VARIABLES}]",
}

# A Python function's node starts at `def`: its decorators sit before it, beside it inside
# `decorated_definition`. A TypeScript method's decorators sit before it in the class body. So in these
# grammars a function's first decorator is the earliest decorator before it with only decorators and
# comments between. JavaScript's grammar holds a method's decorators inside `method_definition`, whose
# node already starts at the first one, and a class's decorators stay with the class head.
DECORATED_KINDS = {
    "python": ("function_definition",),
    "typescript": ("method_definition",),
    "tsx": ("method_definition",),
    "javascript": (),
}

# A function whose body only declares a shape: `...`, `pass`, a docstring or `raise NotImplementedError`,
# alone or together, as in a Protocol. TypeScript declares a shape without a body (an interface's or an
# abstract method's signature, an overload), and such a signature is no function.
STUB_RULES = {
    "python": """  kind: function_definition
  has:
    field: body
    not:
      has:
        not:
          any:
            - kind: pass_statement
            - kind: comment
            - kind: expression_statement
              not: {has: {not: {any: [{kind: ellipsis}, {kind: string}]}}}
            - kind: raise_statement
              has:
                any:
                  - {kind: identifier, regex: ^NotImplementedError$}
                  - {kind: call, has: {field: function, kind: identifier, regex: ^NotImplementedError$}}""",
}

# The installed ast-grep supports tsx but not Flow. Route marked files through tsx;
# unsupported Flow constructs remain visible through ERROR nodes.
FLOW_LANGUAGE = "flow"
FUNCTION_KINDS[FLOW_LANGUAGE] = FUNCTION_KINDS["tsx"]
CLASS_KINDS[FLOW_LANGUAGE] = CLASS_KINDS["tsx"]
DECLARATION_RULES[FLOW_LANGUAGE] = _SCRIPT_DECLARATIONS
DECORATED_KINDS[FLOW_LANGUAGE] = DECORATED_KINDS["tsx"]

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
_SCRIPT_VALUE_DECLARATION = re.compile(r"^\s*(?:export\s+)?(?:const|let)\s+(?!enum\b)[A-Za-z_$]")
_SCRIPT_TYPE_DECLARATION = re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(?:type|interface)\s+[A-Za-z_$]")


def language_of(path: str) -> str | None:
    """The language of ``path``'s suffix, read from the string because the index asks for every
    file many times: a name's last dot after its first character starts the suffix."""
    name = path.rpartition("/")[2]
    dot = name.rfind(".")
    return LANGUAGE_BY_SUFFIX.get(name[dot:]) if dot > 0 else None


def parse_language(path: str, content: bytes) -> str | None:
    """The language a file holding ``content`` parses as: like ``language_of``, but a JavaScript file
    whose leading comments carry the ``@flow`` pragma parses as ``flow`` (with the tsx grammar)."""
    language = language_of(path)
    if language == "javascript" and has_flow_pragma(split_lines(content.decode(errors="replace"))):
        return FLOW_LANGUAGE
    return language


def split_lines(text: str) -> tuple[str, ...]:
    """Lines split at newlines only, as the parser counts them; ``str.splitlines`` also splits at form
    feeds and other separators, which would shift every line number after them."""
    lines = text.replace("\r", "").split("\n")
    return tuple(lines[:-1] if lines and lines[-1] == "" else lines)


_FLOW_PRAGMA = re.compile(r"@flow\b")


def has_flow_pragma(lines: Iterable[str]) -> bool:
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


def declares_type(first_line: str) -> bool:
    """Whether a type can name what a module-level declaration declares: a type alias, interface or
    enum, or a Python assignment (which may be a type alias), but not a script constant or variable."""
    return not _SCRIPT_VALUE_DECLARATION.match(first_line)


def declares_value(first_line: str) -> bool:
    """Whether a value can name what a module-level declaration declares: a script constant, variable
    or enum, or a Python assignment, but not a script type alias or interface."""
    return not _SCRIPT_TYPE_DECLARATION.match(first_line)


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


_PYTHON_ROLES = (
    ReferenceRole("argument", ("kind: argument_list", "kind: keyword_argument\nfield: value")),
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


_EXPORT = "{field: declaration, kind: export_statement}"
_AMBIENT_EXPORT = f"{{kind: ambient_declaration, inside: {_EXPORT}}}"
_VARIABLES = "[{kind: lexical_declaration}, {kind: variable_declaration}]"
_EXPORTED_NAME = f"""  inside:
    field: name
    any:
      - inside: {{field: declaration, kind: export_statement, not: {{has: {{regex: '^default$'}}}}}}
      - kind: variable_declarator
        inside: {{any: {_VARIABLES}, inside: {_EXPORT}}}"""
_TYPED_EXPORTED_NAME = f"""{_EXPORTED_NAME}
      - inside: {_AMBIENT_EXPORT}
      - kind: variable_declarator
        inside: {{any: {_VARIABLES}, inside: {_AMBIENT_EXPORT}}}"""
# The name node of each declaration an ``export`` statement makes, one match per name, so the
# declaration's body (a nested function, a template literal) never names the export. A default
# export has no name of its own.
EXPORTED_NAMES = {
    "typescript": f"  any: [{{kind: identifier}}, {{kind: type_identifier}}]\n{_TYPED_EXPORTED_NAME}",
    "tsx": f"  any: [{{kind: identifier}}, {{kind: type_identifier}}]\n{_TYPED_EXPORTED_NAME}",
    "javascript": f"  kind: identifier\n{_EXPORTED_NAME}",
}


def export_rules(languages: Iterable[str]) -> str:
    """ast-grep rules for the script export surface: the names exported declarations make, and the
    ``{ ... }`` clause specifiers that carry aliased names. Python has no such kinds, so it
    contributes no rules."""
    documents = []
    for language in languages:
        if language == "python":
            continue
        grammar = grammar_of(language)
        documents.append(f"id: export_surface\nlanguage: {grammar}\nrule:\n{EXPORTED_NAMES[grammar]}")
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
