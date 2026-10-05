"""Thin wrappers over the command-line tools the index runs: ast-grep, ripgrep and git."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import IO

import msgspec

from .file_shape import MAX_PARSE_PEAK_MB, Placement, placement_of
from .spans import TextHit

AST_GREP = "ast-grep"
RIPGREP = "rg"
# `--no-config` keeps ripgrep from reading `RIPGREP_CONFIG_PATH`: over an untrusted repository, a
# config file could otherwise inject flags such as `--pre=<program>`, which runs an arbitrary
# program. It also keeps a personal rg config from changing what the index sees.
RIPGREP_SAFE = (RIPGREP, "--no-config")
_NO_MATCHES_EXIT = 1
_SCANNED_FILE_PREFIX = "sg: entity|file|"
NEUTRAL_AST_GREP_CONFIG = "ruleDirs: []\n"
"""The smallest sgconfig ast-grep accepts. Passed with ``--config`` it replaces the discovery of the
analysed repository's own sgconfig.yml, which is customer content: its ``languageGlobs`` would change
what a file is parsed as, and its ``customLanguages`` makes ast-grep load a library the repository
names. Confirmed with ast-grep 0.45.1 that ``--config`` replaces discovery and is not merged with it."""
NOT_UTF8_REASON = "not parsed: not valid UTF-8"
NOT_PARSED_REASON = "not parsed: ast-grep skipped the file and printed nothing for it"
"""ast-grep 0.45.1 skips a file it was handed on its command line, exits 0 and prints nothing, when the
file has more than 3,000,000 bytes and more than 200,000 lines (found by bisection on synthetic files:
both limits must be exceeded; ``--stdin`` is not affected). Even a rule on ``kind: program`` matches
nothing then, so a skipped file reads exactly like a file without functions. Only ``--inspect=entity``
tells them apart: it prints one ``entity|file|PATH`` line for every file ast-grep actually scanned. A
file that is not valid UTF-8 is skipped the same way (``NOT_UTF8_REASON``). A file of that
size can be within the single-file limit (110,000 small functions, 3.4 MB, estimate about 300 MB) and so be
parsed alone, which is why every run, side by side or alone, is checked."""
MAX_FILES_PER_COMMAND = 300
MAX_ARGUMENT_BYTES = 128 * 1024


class ToolFailedError(RuntimeError):
    """A command-line tool failed for a reason other than finding nothing."""


def run_command(
    arguments: Sequence[str], cwd: Path, *, no_match_exit: int | None = None, stdin: str | None = None
) -> str:
    """The command's output; ``no_match_exit`` is the exit code a search tool uses for "nothing found";
    ``stdin``, when given, is written to the command's standard input."""
    return command_output(arguments, cwd, no_match_exit=no_match_exit, stdin=stdin).decode()


def command_output(
    arguments: Sequence[str], cwd: Path, *, no_match_exit: int | None = None, stdin: str | None = None
) -> bytes:
    """``run_command``'s output as the bytes the command wrote, for output that quotes file content."""
    completed = subprocess.run(
        list(arguments), cwd=cwd, input=None if stdin is None else stdin.encode(), capture_output=True
    )
    if completed.returncode not in (0, no_match_exit):
        raise _tool_failure(arguments[0], completed.returncode, completed.stderr.decode(errors="replace"))
    return completed.stdout


def _tool_failure(tool: str, returncode: int, stderr: str) -> ToolFailedError:
    """The failure with the tool's whole error output, whose cause is often its last line."""
    return ToolFailedError(f"{tool} exited {returncode}: {stderr.strip()}")


@cache
def ast_grep_version() -> str:
    return run_command([AST_GREP, "--version"], Path.cwd()).strip()


def ast_grep_rules(
    rules_yaml: str,
    files: Sequence[str],
    cwd: Path,
    config: str | None = None,
    *,
    refused: dict[str, str],
    decode: Callable[[str], dict] = json.loads,
) -> Iterator[dict]:
    """The matches of ``rules_yaml`` over ``files``, one at a time as ast-grep prints them, so no
    process's whole output is ever held. Every parse passes through here, placed by its estimated parse
    peak (``file_shape``): files within ``MAX_PARSE_PEAK_MB`` are parsed side by side; a file over it
    but within ``single_parse_limit_mb()`` is parsed alone, one at a time; a file over that
    is never handed to ast-grep and is added to ``refused`` with its reason when the iteration starts,
    so read ``refused`` after the matches. A file that ast-grep itself skipped without parsing
    (``NOT_PARSED_REASON``) is added when its process ends, so it is never taken for a file without
    symbols. ast-grep always runs with a JVN-owned sgconfig: ``config``,
    when given, is sgconfig YAML text (a ``languageGlobs`` remapping, say), otherwise
    ``NEUTRAL_AST_GREP_CONFIG``. It is written to a temporary file outside every repository and passed
    with ``--config``, so the repository being analysed never configures the parser. ``decode`` turns
    one printed match into the dict the caller reads, in every run, side by side or alone; a caller
    that reads few fields passes a decoder that skips the rest."""
    placed = _placed(files, cwd, single_parse_limit_mb())
    refused.update(placed.refused)
    if not placed.side_by_side and not placed.alone:
        return
    with ExitStack() as resources:
        directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="jev-navigator-sgconfig-"))
        path = Path(directory) / "sgconfig.yml"
        path.write_text(NEUTRAL_AST_GREP_CONFIG if config is None else config)
        command = [AST_GREP, "scan", "--inline-rules", rules_yaml, "--config", str(path)]
        for chunk in file_chunks(placed.side_by_side):
            yield from _scanned(command, chunk, cwd, refused, decode)
        for file in placed.alone:
            yield from _scanned([*command, "--threads", "1"], [file], cwd, refused, decode)


def _scanned(
    command: Sequence[str],
    files: Sequence[str],
    cwd: Path,
    refused: dict[str, str],
    decode: Callable[[str], dict],
) -> Iterator[dict]:
    """The matches of one ast-grep run over ``files``, each decoded with ``decode``; a file it skipped
    without parsing is added to ``refused`` when the run ends."""
    yield from _json_lines(
        [*command, "--json=stream", "--inspect=entity", "--", *files],
        cwd,
        decode=decode,
        on_stderr=lambda inspection: refused.update(_skipped_files(files, inspection, cwd)),
    )


def single_parse_limit_mb() -> float:
    """The most one file may take when parsed alone. It is the side-by-side bound, so no file is parsed
    alone, until JVN's memory settings provide a single-file limit."""
    return MAX_PARSE_PEAK_MB


@dataclass
class _Placed:
    side_by_side: list[str] = field(default_factory=list)
    alone: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)


def _placed(files: Sequence[str], cwd: Path, single_parse_limit: float) -> _Placed:
    placed = _Placed()
    for file in files:
        try:
            placement, reason = placement_of(cwd, file, single_parse_limit)
        except OSError as error:
            placement, reason = Placement.REFUSED, f"could not be measured: {type(error).__name__}: {error}"
        if placement is Placement.SIDE_BY_SIDE:
            placed.side_by_side.append(file)
        elif placement is Placement.ALONE:
            placed.alone.append(file)
        else:
            placed.refused[file] = reason
    return placed


def file_chunks(files: Sequence[str], *, bytes_only: bool = False) -> Iterator[Sequence[str]]:
    """``files`` in order, split so no command gets more than ``MAX_ARGUMENT_BYTES`` of paths, and,
    unless ``bytes_only``, no more than ``MAX_FILES_PER_COMMAND`` of them, which bounds what one
    parser process holds. A text search holds no file, so only the argument limit applies to it."""
    most_files = None if bytes_only else MAX_FILES_PER_COMMAND
    start, size = 0, 0
    for position, file in enumerate(files):
        length = len(file.encode()) + 1
        too_many = most_files is not None and position - start >= most_files
        full = too_many or size + length > MAX_ARGUMENT_BYTES
        if position > start and full:
            yield files[start:position]
            start, size = position, 0
        size += length
    if start < len(files):
        yield files[start:]


def _skipped_files(chunk: Sequence[str], inspection: str, cwd: Path) -> dict[str, str]:
    """The non-empty files of ``chunk`` that ast-grep's ``--inspect=entity`` output does not list as
    scanned. ast-grep lists no empty file either, and an empty file has nothing to find."""
    scanned = {
        line.removeprefix(_SCANNED_FILE_PREFIX).rsplit(": language=", 1)[0]
        for line in inspection.splitlines()
        if line.startswith(_SCANNED_FILE_PREFIX)
    }
    reasons = {file: _not_parsed_reason(cwd / file) for file in chunk if file not in scanned}
    return {file: reason for file, reason in reasons.items() if reason is not None}


def _not_parsed_reason(path: Path) -> str | None:
    """Why ast-grep skipped the file, or None when it had nothing to parse: the file is empty, or it
    left the disk during the scan, which the index reports as disappeared."""
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    if not content:
        return None
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return NOT_UTF8_REASON
    return NOT_PARSED_REASON


def _json_lines(
    arguments: Sequence[str],
    cwd: Path,
    *,
    decode: Callable[[str], dict],
    on_stderr: Callable[[str], None] | None = None,
) -> Iterator[dict]:
    """Each line the command prints, decoded with ``decode`` while it runs. stderr goes to a file, so a full
    stderr pipe cannot stall the command; the process is killed if the reader stops early. A line
    that is no JSON (the process died partway through it) fails with the process's exit code and
    stderr, which say why it stopped. ``on_stderr`` gets the whole stderr text once the command has
    ended successfully."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(list(arguments), cwd=cwd, stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            for line in process.stdout:
                if line.strip():
                    yield _json_object(line, process, errors, arguments[0], decode)
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdout.close()
            returncode = process.wait()
        if returncode not in (0, _NO_MATCHES_EXIT):
            raise _tool_failed(arguments[0], returncode, errors)
        if on_stderr is not None:
            on_stderr(_stderr_text(errors))


def _json_object(
    line: str, process: subprocess.Popen, errors: IO[bytes], tool: str, decode: Callable[[str], dict]
) -> dict:
    """A whole line the decoder rejects for a missing or mistyped field raises as it is: the tool
    printed it in full, so the decoder's expectation, not the tool, is what failed."""
    try:
        return decode(line)
    except msgspec.ValidationError:
        raise
    except ValueError as malformed:
        process.kill()
        raise _tool_failed(tool, process.wait(), errors) from malformed


def _tool_failed(tool: str, returncode: int, errors: IO[bytes]) -> ToolFailedError:
    """The failure with the command's own error text, without ``--inspect=entity``'s list of scanned
    files, which can run to one line per file of the scope."""
    lines = _stderr_text(errors).splitlines()
    message = "\n".join(line for line in lines if not line.startswith(_SCANNED_FILE_PREFIX))
    return _tool_failure(tool, returncode, message)


def _stderr_text(errors: IO[bytes]) -> str:
    errors.seek(0)
    return errors.read().decode(errors="replace")


def ripgrep_fixed(
    text: str, files: Sequence[str], cwd: Path, max_hits: int, context_bytes: int, *, whole_word: bool = False
) -> list[TextHit]:
    """``ripgrep_windows`` for the exact ``text``; ``whole_word`` keeps only hits no word character
    touches."""
    hit = literal_pattern(text)
    if whole_word:
        hit = rf"(?:^|[^\w\n]){hit}(?:[^\w\n]|$)"
    return ripgrep_windows(hit, files, cwd, max_hits, context_bytes)


def ripgrep_windows(
    hit_pattern: str, files: Sequence[str], cwd: Path, max_hits: int, context_bytes: int
) -> list[TextHit]:
    """The lines matching the ripgrep regular expression ``hit_pattern``, at most ``max_hits`` per
    file, each as the bytes around one hit: up to ``context_bytes`` before and after, so a one-line
    bundle costs no more than a short line. The match runs on to the end of the line, so each line
    matches once, and ``--replace`` prints only the hit and its context, each on its own line;
    ripgrep's JSON would carry the whole line."""
    if not files:
        return []
    context = f"(?-u:.){{0,{context_bytes}}}"
    pattern = f"(?P<before>{context})(?P<hit>{hit_pattern})(?P<after>{context})(?-u:.)*"
    command = [*RIPGREP_SAFE, "--only-matching", "--line-number", "--with-filename", "--null"]
    command += ["--max-count", str(max_hits), "--replace", _WINDOW_FIELDS, "--regexp", pattern, "--"]
    windows: dict[tuple[str, int], TextHit] = {}
    for chunk in file_chunks(files, bytes_only=True):
        output = command_output([*command, *chunk], cwd, no_match_exit=_NO_MATCHES_EXIT)
        for window in _windows(output):
            windows.setdefault((window.file, window.line), window)
    return list(windows.values())


def literal_pattern(text: str) -> str:
    """``text`` as a ripgrep regular expression matching exactly it: characters other than letters,
    digits and underscores are written as code points, so none is read as syntax."""
    return "".join(char if char.isalnum() or char == "_" else f"\\x{{{ord(char):x}}}" for char in text)


# ripgrep ends a printed replacement with a newline unless it already ends with one, so the record
# ends with a fixed mark: an empty ``after`` would otherwise leave the record without its own end.
_WINDOW_FIELDS = "${before}\n${hit}\n${after}|"


def _windows(output: bytes) -> Iterator[TextHit]:
    """The windows ripgrep's ``--null`` printer gives for ``_WINDOW_FIELDS``: ``path NUL line:before``,
    then the hit, then the text after followed by ``|``, each ending in a newline. None of the three
    holds a newline, and a path ends at its NUL, so a newline in a path cannot split a record. Bytes
    that are not UTF-8 are decoded the way the index reads files, with invalid bytes replaced."""
    position = 0
    while position < len(output):
        path_end = output.index(b"\0", position)
        number_end = output.index(b":", path_end)
        before_end = output.index(b"\n", number_end)
        hit_end = output.index(b"\n", before_end + 1)
        record_end = output.index(b"|\n", hit_end + 1)
        before = output[number_end + 1 : before_end]
        after = output[hit_end + 1 : record_end]
        yield TextHit(
            output[position:path_end].decode(errors="replace").removeprefix("./"),
            int(output[path_end + 1 : number_end]),
            (before + output[before_end + 1 : hit_end] + after).decode(errors="replace").rstrip("\r"),
        )
        position = record_end + 2


def ripgrep_files(texts: str | Sequence[str], files: Sequence[str], cwd: Path) -> tuple[str, ...]:
    """Every supplied file containing any of the exact ``texts``, without a result-count cutoff. Only
    paths come back, and ripgrep stops reading a file at its first match, so a 20 MB one-line bundle
    costs what a small file does; a search printing lines would print that line with every match."""
    patterns = [texts] if isinstance(texts, str) else list(texts)
    if not files or not patterns:
        return ()
    found: list[str] = []
    with _pattern_file(patterns) as pattern_path:
        command = [*RIPGREP_SAFE, "--files-with-matches", "--null", "--fixed-strings", "-f", pattern_path]
        for chunk in file_chunks(files, bytes_only=True):
            output = run_command([*command, "--", *chunk], cwd, no_match_exit=_NO_MATCHES_EXIT)
            found += [path.removeprefix("./") for path in output.split("\0") if path]
    return tuple(found)


@contextmanager
def _pattern_file(texts: Sequence[str]) -> Iterator[str]:
    """The path of a temporary ripgrep pattern file holding ``texts``, one per line, so their number
    never meets the argument limit."""
    if any("\n" in text for text in texts):
        raise ValueError("a text searched for by file cannot hold a line break")
    with tempfile.NamedTemporaryFile("w", prefix="jev-navigator-patterns-", suffix=".txt") as pattern_file:
        pattern_file.write("".join(f"{text}\n" for text in texts))
        pattern_file.flush()
        yield pattern_file.name


def inside_git_worktree(cwd: Path) -> bool:
    """Whether ``cwd`` lies in a Git worktree. Only git's own "not a git repository" means no; any
    other failure, such as a repository git refuses for dubious ownership, is raised, so it is never
    listed as a plain directory. Git runs in the C locale so that message is never translated."""
    completed = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    if completed.returncode == 0:
        return completed.stdout.strip() == "true"
    if "not a git repository" in completed.stderr:
        return False
    raise _tool_failure("git", completed.returncode, completed.stderr)


def head_commit(cwd: Path) -> str:
    """HEAD's commit in the Git worktree at ``cwd``; empty before its first commit, which is the one
    case `git rev-parse -q --verify` reports with exit 1 and no message. Any other failure raises."""
    return run_command(["git", "rev-parse", "-q", "--verify", "HEAD"], cwd, no_match_exit=1).strip()


def git(arguments: Sequence[str], cwd: Path, *, stdin: str | None = None) -> str:
    return run_command(["git", *arguments], cwd, stdin=stdin)


def export_blobs(repository: Path, blobs: Mapping[str, str], destination: Path) -> None:
    """Writes each blob, keyed by its path, into ``destination``, all read with one
    ``git cat-file --batch``. Blobs are asked for by object id, so any byte in a path is safe. A path
    that would leave ``destination`` and an object git does not have both raise ``ToolFailedError``."""
    if not blobs:
        return
    for path, content in zip(blobs, _cat_file_batch(repository, blobs.values()), strict=True):
        _write_inside(destination, path, content)


def git_blob(repository: Path, object_id: str) -> bytes:
    """The bytes of one blob, asked for by object id; an object git does not have raises
    ``ToolFailedError``."""
    (content,) = _cat_file_batch(repository, [object_id])
    return content


def _cat_file_batch(repository: Path, object_ids: Iterable[str]) -> list[bytes]:
    requests = "".join(f"{object_id}\n" for object_id in object_ids).encode()
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repository,
        input=requests,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise _tool_failure("git cat-file", completed.returncode, completed.stderr.decode())
    return _batch_contents(completed.stdout)


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
