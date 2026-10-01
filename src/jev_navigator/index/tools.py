"""Thin wrappers over the command-line tools the index runs: ast-grep, ripgrep and git."""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from functools import cache
from pathlib import Path

from .spans import TextHit

logger = logging.getLogger(__name__)

AST_GREP = "ast-grep"
RIPGREP = "rg"
# `--no-config` keeps ripgrep from reading `RIPGREP_CONFIG_PATH`: over an untrusted repository, a
# config file could otherwise inject flags such as `--pre=<program>`, which runs an arbitrary
# program. It also keeps a personal rg config from changing what the index sees.
_RIPGREP_SAFE = (RIPGREP, "--no-config")
_NO_MATCHES_EXIT = 1


class ToolFailedError(RuntimeError):
    """A command-line tool failed for a reason other than finding nothing."""


def run_command(arguments: Sequence[str], cwd: Path, *, no_match_exit: int | None = None) -> str:
    """The command's output; ``no_match_exit`` is the exit code a search tool uses for "nothing found"."""
    completed = subprocess.run(list(arguments), cwd=cwd, capture_output=True, text=True)
    if completed.returncode not in (0, no_match_exit):
        detail = completed.stderr.strip()[:300]
        raise ToolFailedError(f"{arguments[0]} exited {completed.returncode}: {detail}")
    return completed.stdout


@cache
def ast_grep_version() -> str:
    return run_command([AST_GREP, "--version"], Path.cwd()).strip()


def ast_grep_rules(rules_yaml: str, files: Sequence[str], cwd: Path, config: str | None = None) -> list[dict]:
    """The matches of ``rules_yaml`` over ``files``. ``config``, when given, is sgconfig YAML text
    (a ``languageGlobs`` remapping, say); it is written to a temporary file outside every repository
    and passed with ``-c``."""
    if not files:
        return []
    with ExitStack() as resources:
        command = [AST_GREP, "scan", "--inline-rules", rules_yaml]
        if config is not None:
            directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="jev-navigator-sgconfig-"))
            path = Path(directory) / "sgconfig.yml"
            path.write_text(config)
            command += ["--config", str(path)]
        output = run_command(
            [*command, "--json=compact", *files],
            cwd,
            no_match_exit=_NO_MATCHES_EXIT,
        )
    return _json_list(output)


def ripgrep_fixed(text: str, files: Sequence[str], cwd: Path, max_hits: int) -> list[TextHit]:
    """The lines holding ``text``. JSON events are split at newlines only, since a line of code may
    hold a Unicode line separator that ``str.splitlines`` would split."""
    if not files:
        return []
    output = run_command(
        [*_RIPGREP_SAFE, "--json", "--fixed-strings", "--max-count", str(max_hits), "--", text, *files],
        cwd,
        no_match_exit=_NO_MATCHES_EXIT,
    )
    events = (json.loads(line) for line in output.split("\n") if line.strip())
    return [_text_hit(event["data"]) for event in events if event.get("type") == "match"]


def ripgrep_files(text: str, files: Sequence[str], cwd: Path) -> tuple[str, ...]:
    """Every supplied file containing the exact text, without a result-count cutoff."""
    if not files:
        return ()
    output = run_command(
        [*_RIPGREP_SAFE, "--files-with-matches", "--null", "--fixed-strings", "--", text, *files],
        cwd,
        no_match_exit=_NO_MATCHES_EXIT,
    )
    return tuple(path.removeprefix("./") for path in output.split("\0") if path)


def listed_files(cwd: Path, prefixes: Sequence[str] = ()) -> tuple[str, ...]:
    """Regular, non-symlink files owned by this working directory, including hidden paths.

    A Git worktree uses its tracked and untracked, non-ignored inventory, which naturally excludes
    nested repositories and managed worktrees. A non-Git directory uses ripgrep's ignore policy.
    """
    try:
        inside_git = git(["rev-parse", "--is-inside-work-tree"], cwd).strip() == "true"
    except ToolFailedError:
        inside_git = False
    if inside_git:
        output = git(["ls-files", "-z", "-c", "-o", "--exclude-standard", "--", *prefixes], cwd)
    else:
        output = run_command(
            [
                *_RIPGREP_SAFE,
                "--files",
                "--hidden",
                "--null",
                "--glob",
                "!.git",
                "--glob",
                "!.git/**",
                *prefixes,
            ],
            cwd,
        )
    files = []
    for raw in output.split("\0"):
        path = raw.removeprefix("./")
        candidate = cwd / path
        if path and candidate.is_file() and not candidate.is_symlink():
            files.append(path)
    return tuple(sorted(dict.fromkeys(files)))


def _text_hit(match: dict) -> TextHit:
    return TextHit(_decoded(match["path"]), match["line_number"], _decoded(match["lines"]).rstrip("\r\n"))


def _decoded(field: dict) -> str:
    """ripgrep reports a path or line that is not valid UTF-8 as base64 ``bytes`` instead of ``text``;
    it is decoded the way the index reads files, with invalid bytes replaced."""
    if "text" in field:
        return field["text"]
    return base64.b64decode(field["bytes"]).decode("utf-8", errors="replace")


def git(arguments: Sequence[str], cwd: Path) -> str:
    return run_command(["git", *arguments], cwd)


def _json_list(output: str) -> list[dict]:
    return json.loads(output) if output.strip() else []


def export_blobs(repository: Path, blobs: Mapping[str, str], destination: Path) -> None:
    """Writes each blob, keyed by its path, into ``destination``, all read with one
    ``git cat-file --batch``. Blobs are asked for by object id, so any byte in a path is safe. A path
    that would leave ``destination`` and an object git does not have both raise ``ToolFailedError``."""
    if not blobs:
        return
    requests = "".join(f"{object_id}\n" for object_id in blobs.values()).encode()
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repository,
        input=requests,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ToolFailedError(
            f"git cat-file exited {completed.returncode}: {completed.stderr.decode()[:300]}"
        )
    for path, content in zip(blobs, _batch_contents(completed.stdout), strict=True):
        _write_inside(destination, path, content)


def _batch_contents(output: bytes) -> list[bytes]:
    """The object contents in ``git cat-file --batch`` output, in request order."""
    contents = []
    position = 0
    while position < len(output):
        header_end = output.index(b"\n", position)
        object_id, _, details = output[position:header_end].partition(b" ")
        if details == b"missing":
            raise ToolFailedError(f"git has no object {object_id.decode()}")
        start = header_end + 1
        end = start + int(details.split()[1])
        contents.append(output[start:end])
        position = end + 1
    return contents


def _write_inside(destination: Path, path: str, content: bytes) -> None:
    root = destination.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ToolFailedError(f"{path!r} would be written outside {root}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
