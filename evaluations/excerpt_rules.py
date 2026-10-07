"""Development excerpt rules over JVN's existing ast-grep grammars and CodeIndex.

This is measurement code, not a new parser or a shipping excerpt API. Caller
queries are data. Labels are never used by line selection.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.languages import CLASS_KINDS, FUNCTION_KINDS, grammar_of, parse_language, sgconfig_of
from jev_navigator.index.spans import Span
from jev_navigator.index.tools import ast_grep_rules
from jev_navigator.mentions import names_from_text, spelling_variants

RULES = ("A", "B", "C", "D", "E", "F", "G15", "G25", "G40")
# Fixed before scoring. Same prose stop words as the earlier window study,
# supplied by census.py from its frozen ordering-method.json.
STOP_WORDS: set[str] = set()


@lru_cache(maxsize=32768)
def words(text: str) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return {
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", text)
        if len(w) > 2 and w.lower() not in STOP_WORDS
    }


@lru_cache(maxsize=512)
def query_terms(query: str):
    variants = {variant for name in names_from_text(query).code for variant in spelling_variants(name)}
    pattern = (
        re.compile(r"(?<![\w$])(?:" + "|".join(map(re.escape, sorted(variants))) + r")(?![\w$])")
        if variants
        else None
    )
    return words(query), pattern


@lru_cache(maxsize=32768)
def matches(text: str, query: str) -> bool:
    terms, pattern = query_terms(query)
    return bool(words(text) & terms or pattern is not None and pattern.search(text))


def ranges(numbers):
    result = []
    for n in sorted(set(numbers)):
        if result and result[-1][1] + 1 == n:
            result[-1][1] = n
        else:
            result.append([n, n])
    return result


def node_range(match):
    r = match["range"]
    end = r["end"]["line"] + (r["end"]["column"] > 0)
    return [r["start"]["line"] + 1, max(r["start"]["line"] + 1, end)]


def _rule(language, role, kind, field=None):
    result = f"id: {role}_{kind}\nlanguage: {grammar_of(language)}\nrule:\n  kind: {kind}"
    if field:
        result += f"\n  has:\n    field: {field}\n    pattern: $PART"
    return result


def syntax_rules(language):
    python = language == "python"
    grammar = grammar_of(language)
    rules = [
        _rule(language, "signature", k, "body") for k in (*FUNCTION_KINDS[grammar], *CLASS_KINDS[grammar])
    ]
    if python:
        rules += [
            _rule(language, "decorator", "decorator"),
            _rule(language, "string", "expression_statement", None),
        ]
    condition_fields = (
        {"if_statement": "consequence", "while_statement": "body", "for_statement": "body"}
        if python
        else {
            "if_statement": "consequence",
            "while_statement": "body",
            "for_statement": "body",
            "for_in_statement": "body",
            "switch_statement": "body",
            "catch_clause": "body",
            "else_clause": None,
            "try_statement": "body",
            "finally_clause": None,
            "do_statement": "body",
        }
    )
    if python:
        condition_fields.update(
            {
                "elif_clause": "consequence",
                "else_clause": "body",
                "except_clause": None,
                "with_statement": "body",
                "try_statement": "body",
                "finally_clause": None,
            }
        )
    rules += [_rule(language, "condition", k, field) for k, field in condition_fields.items()]
    exits = ["return_statement", "raise_statement"] if python else ["return_statement", "throw_statement"]
    rules += [_rule(language, "exit", k) for k in (*exits, "break_statement", "continue_statement")]
    assignments = (
        {"assignment": "left", "augmented_assignment": "left", "named_expression": "name"}
        if python
        else {
            "variable_declarator": "name",
            "assignment_expression": "left",
            "augmented_assignment_expression": "left",
        }
    )
    rules += [_rule(language, "assignment", k, field) for k, field in assignments.items()]
    rules += [
        _rule(language, "identifier", "identifier"),
        _rule(language, "call", "call" if python else "call_expression"),
    ]
    return "\n---\n".join(rules)


class Structure:
    """Extra structural facts using JVN's parser owner, with JVN binding resolution."""

    def __init__(self, root, files):
        self.root = Path(root)
        self.index = CodeIndex.from_git(self.root)
        self._facts = {
            file: {
                "signatures": [],
                "decorators": [],
                "conditions": [],
                "exits": [],
                "assignments": [],
                "identifiers": [],
                "calls": [],
                "strings": [],
                "functions": [],
                "refused": "",
            }
            for file in files
        }
        groups = defaultdict(list)
        for file in files:
            language = parse_language(file, (self.root / file).read_bytes())
            if language:
                # CodeIndex owns grammar recovery, including JavaScript read
                # again as Flow when that grammar parses more of the file.
                indexed = self.index._facts_in(file)
                language = indexed.language
                self._facts[file]["refused"] = indexed.refusal or ""
            self._facts[file]["language"] = language
            if language:
                groups[language].append(file)
        for language, paths in groups.items():
            refused = {}
            for m in ast_grep_rules(
                syntax_rules(language), paths, self.root, config=sgconfig_of(language), refused=refused
            ):
                f = self._facts[m["file"]]
                role = m["ruleId"].split("_", 1)[0]
                a, b = node_range(m)
                part = m.get("metaVariables", {}).get("single", {}).get("PART")
                record = {"range": [a, b], "text": m["text"], "kind": m["ruleId"].split("_", 1)[1]}
                if role in {"signature", "condition"}:
                    c = part["range"]["start"]["line"] + 1 if part else a + 1
                    record["header"] = [a, max(a, c - 1 if language == "python" else c)] if part else [a, a]
                    record["body_start"] = c
                if role == "assignment":
                    record["target_text"] = part["text"]
                    record["target_range"] = node_range(part)
                key = {
                    "signature": "signatures",
                    "condition": "conditions",
                    "exit": "exits",
                    "assignment": "assignments",
                    "identifier": "identifiers",
                    "call": "calls",
                    "string": "strings",
                    "decorator": "decorators",
                }[role]
                f[key].append(record)
            for file, reason in refused.items():
                self._facts[file]["refused"] = reason
        # Resolve calls only for parsed target files. The index retains original
        # repository scope, so imports to a file outside this measured set bind.
        for file, f in self._facts.items():
            if not f["language"]:
                continue
            functions = self.index.functions_in(file)
            structure = self.index._file_structure(file)
            for assignment in f["assignments"]:
                a, b = assignment["target_range"]
                # Reuse JVN's actual binding names. Property receivers and
                # destructuring keys are not local variable definitions.
                targets = {name.name for name in structure.local_names if a <= name.line <= b}
                if re.fullmatch(r"[A-Za-z_$][\w$]*", assignment["target_text"]):
                    targets.add(assignment["target_text"])
                targets.update(
                    span.name for span in structure.declarations if a <= span.start <= b and span.is_named
                )
                assignment["targets"] = sorted(targets)
            f["functions"] = [{"range": [s.start, s.end], "name": s.name} for s in functions]
            edges = self.index.callee_edges(Span(file, 1, len(self.index.lines(file))))
            by_line = defaultdict(list)
            for edge in edges:
                if edge.binding.target is not None:
                    by_line[edge.line].append(
                        {"name": edge.name, "target": edge.binding.target.key, "status": edge.binding.status}
                    )
            for call in f["calls"]:
                call["delegates"] = by_line[call["range"][0]]

    def facts(self, file):
        return self._facts[file]


def _add(selected, interval, visible):
    a, b = interval
    selected.update(n for n in range(a, b + 1) if n in visible)


def _condition_closure(selected, facts, visible):
    # Conditions include if/elif/else, loop and exception branch headers. Keep
    # headers and closing delimiters, never the entire enclosing body.
    for condition in sorted(facts.get("conditions", []), key=lambda c: c["range"][1] - c["range"][0]):
        a, b = condition["range"]
        if any(a <= n <= b for n in selected):
            _add(selected, condition["header"], visible)
            if facts.get("language") != "python":
                _add(selected, [b, b], visible)


def selected_lines(source, facts, query, rule):
    visible = set(source)
    if not visible:
        return set()
    threshold = int(rule[1:]) if rule.startswith("G") else 0
    if threshold and len(visible) <= threshold:
        return visible
    stage = "F" if threshold else rule
    if not facts.get("language") or facts.get("refused"):
        # Text fallback has no asserted control/data flow. Preserve first line,
        # then exact lexical matches with two lines of context and markers.
        selected = {min(visible)}
    else:
        selected = set()
        for sig in facts["signatures"]:
            if sig["range"][0] in visible:
                _add(selected, sig["header"], visible)
                if facts["language"] != "python":
                    _add(selected, [sig["range"][1], sig["range"][1]], visible)
        for decorator in facts["decorators"]:
            _add(selected, decorator["range"], visible)
        if not selected:
            selected.add(min(visible))
    if stage >= "B":
        for n, text in source.items():
            if matches(text, query):
                _add(selected, [n - 2, n + 2], visible)
    if stage >= "C":
        _condition_closure(selected, facts, visible)
    if stage >= "D":
        for exit_node in facts.get("exits", []):
            _add(selected, exit_node["range"], visible)
        _condition_closure(selected, facts, visible)
    if stage >= "E":
        for call in facts.get("calls", []):
            if call.get("delegates"):
                _add(selected, call["range"], visible)
        _condition_closure(selected, facts, visible)
    if stage >= "F":
        before = set(selected)
        assignments = facts.get("assignments", [])
        for identifier in facts.get("identifiers", []):
            n = identifier["range"][0]
            if n not in before:
                continue
            name = identifier["text"]
            scopes = facts.get("functions", [])

            def scope_at(position, scopes=scopes):
                return min(
                    (s["range"] for s in scopes if s["range"][0] <= position <= s["range"][1]),
                    key=lambda r: r[1] - r[0],
                    default=None,
                )

            prior = [
                a
                for a in assignments
                if name in a["targets"] and a["range"][1] < n and scope_at(a["range"][0]) == scope_at(n)
            ]
            if prior:
                _add(selected, max(prior, key=lambda a: a["range"][1])["range"], visible)
        _condition_closure(selected, facts, visible)
    return selected


def render_excerpt(source, facts, query, rule):
    return render_selection(source, selected_lines(source, facts, query, rule))


def render_selection(source, selected):
    """Price and expose every omission using the same renderer for both studies."""
    selected = set(selected)
    assert selected <= source.keys()
    body = []
    for a, b in ranges(source):
        cursor = a
        while cursor <= b:
            kept = cursor in selected
            end = cursor
            while end < b and (end + 1 in selected) == kept:
                end += 1
            if kept:
                body.extend(f"{n}: {source[n]}" for n in range(cursor, end + 1))
            else:
                body.append(f"... ELIDED lines {cursor}-{end} ({end - cursor + 1} lines) ...")
            cursor = end + 1
    return {
        "runs": ranges(selected),
        "body": "\n".join(body),
        "shown_lines": len(selected),
        "omitted_lines": len(source) - len(selected),
    }
