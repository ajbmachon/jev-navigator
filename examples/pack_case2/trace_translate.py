"""Parse recorded simple shell commands as data, never execute them.

The historical extractor joined newlines and counted redirections as rg paths.
This adapter preserves the supplied approach order and only repairs arguments
when one original search has an unambiguous matching operation and pattern.
"""

import re
import shlex

TOKEN = re.compile(
    r"(?P<space>[ \t\r]+)|(?P<separator>\n|&&|\|\||[;|])"
    r"|(?P<redirect>[0-9]*(?:>>?|<<?)(?:&[0-9-]+)?)"
    r"""|(?P<word>(?:[^\s;&|<>"'\\]+|'[^']*'|"(?:\\.|[^"\\])*"|\\.)+)"""
)
VALUES = {"-g", "--glob", "-e", "-t", "-T", "-m", "-A", "-B", "-C", "--type", "--max-count", "--max-columns"}


def commands(command):
    if command.startswith("/bin/zsh -lc "):
        command = shlex.split(command)[2]
    result, words = [], []
    discard_target = False
    cursor = 0
    while cursor < len(command):
        token = TOKEN.match(command, cursor)
        if token is None:
            raise ValueError(f"Unsupported shell syntax at offset {cursor}")
        cursor = token.end()
        if token.lastgroup == "space":
            continue
        if token.lastgroup == "separator":
            if discard_target:
                raise ValueError("Redirection has no target")
            if words:
                result.append(words)
            words = []
        elif token.lastgroup == "redirect":
            discard_target = "&" not in token.group()
        elif discard_target:
            discard_target = False
        else:
            word = shlex.split(token.group())[0]
            if "$(" in word or "`" in word:
                raise ValueError("Shell substitutions are not admissible replay arguments")
            words.append(word)
    if discard_target:
        raise ValueError("Redirection has no target")
    if words:
        result.append(words)
    return result


def search_arguments(words):
    globs, positionals = [], []
    offset = 1
    explicit_pattern = None
    while offset < len(words):
        word = words[offset]
        if word in VALUES:
            if offset + 1 == len(words):
                raise ValueError(f"Missing argument for {word}")
            value = words[offset + 1]
            if word in {"-g", "--glob"}:
                globs.append(value)
            elif word == "-e":
                explicit_pattern = value
            offset += 2
        else:
            if not word.startswith("-"):
                positionals.append(word)
            offset += 1
    if "--files" in words:
        return "matching_files", {
            "globs": [g for g in globs if g not in {"AGENTS.md", "CLAUDE.md"}],
            "scopes": positionals,
        }
    if explicit_pattern is None:
        if not positionals:
            raise ValueError("Search has no pattern")
        explicit_pattern = positionals.pop(0)
    return "find_text", {"pattern": explicit_pattern, "scopes": positionals}


def repair(proposal, command):
    matches = []
    for words in commands(command):
        if words[0] not in {"rg", "grep"}:
            continue
        operation, args = search_arguments(words)
        identity = "pattern" if operation == "find_text" else "globs"
        if operation == proposal["operation"] and args[identity] == proposal["arguments"].get(identity):
            matches.append(args)
    unique = {repr(args): args for args in matches}
    if len(unique) != 1:
        return proposal, "No unambiguous original search. Supplied arguments retained."
    corrected = next(iter(unique.values()))
    return {**proposal, "arguments": corrected}, "" if corrected == proposal[
        "arguments"
    ] else "shell extraction corrected"
