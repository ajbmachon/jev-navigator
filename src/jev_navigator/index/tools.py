"""Thin wrappers over the command-line tools the index runs: ast-grep, ripgrep and git. Every process
starts through ``memory_limit.started``, so JVN's memory allowance and ceiling cover all of them."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import IO

import msgspec

from .. import memory_limit
from ..runaway_guards import PARSE_GUARD_SECONDS, parse_guard_reason
from .file_shape import Placement, placement_of
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


class _ParseGuardTrippedError(Exception):
    """An ast-grep run outlasted the parse guard and was stopped; ``unfinished`` are its files that
    neither printed a match nor were listed as scanned."""

    def __init__(self, unfinished: list[str]) -> None:
        super().__init__(f"{len(unfinished)} file(s) unfinished when the parse guard stopped ast-grep")
        self.unfinished = unfinished


def run_command(
    arguments: Sequence[str],
    cwd: Path,
    *,
    no_match_exit: int | None = None,
    stdin: str | None = None,
    timeout: float | None = None,
) -> str:
    """The command's output; ``no_match_exit`` is the exit code a search tool uses for "nothing found";
    ``stdin``, when given, is written to the command's standard input. A command still running after
    ``timeout`` seconds is stopped and raises ``subprocess.TimeoutExpired``."""
    options = {"no_match_exit": no_match_exit, "stdin": stdin, "timeout": timeout}
    return command_output(arguments, cwd, **options).decode()


def command_output(
    arguments: Sequence[str],
    cwd: Path,
    *,
    no_match_exit: int | None = None,
    stdin: str | None = None,
    timeout: float | None = None,
) -> bytes:
    """``run_command``'s output as the bytes the command wrote, for output that quotes file content."""
    with memory_limit.started(
        arguments,
        cwd=cwd,
        stdin=None if stdin is None else subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        output, error_output = process.communicate(None if stdin is None else stdin.encode(), timeout=timeout)
    if process.returncode not in (0, no_match_exit):
        raise _tool_failure(arguments[0], process.returncode, error_output.decode(errors="replace"))
    return output


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
    guard_seconds: float = PARSE_GUARD_SECONDS,
) -> Iterator[dict]:
    """The matches of ``rules_yaml`` over ``files``, one at a time as ast-grep prints them, so no
    process's whole output is ever held. Every parse passes through here, placed by its estimated parse
    peak (``file_shape``) against JVN's memory settings (``memory_limit.MemoryLimit``): files within
    ``MAX_PARSE_PEAK_MB`` are parsed side by side, as many at once as the allowance affords; a file over
    it but within ``single_parse_mb`` is parsed alone, on one thread, one at a time, after them; a file
    over that is never handed to ast-grep and is added to ``refused`` with its reason when the iteration
    starts, so read ``refused`` after the matches. A file that ast-grep itself skipped without parsing
    (``NOT_PARSED_REASON``) is added when its process ends, so it is never taken for a file without
    symbols. One scan runs at a time in a process (``memory_limit.parsing``). ast-grep always runs with a
    JVN-owned sgconfig: ``config``, when given, is sgconfig YAML text (a ``languageGlobs`` remapping,
    say), otherwise ``NEUTRAL_AST_GREP_CONFIG``. It is written to a temporary file outside every
    repository and passed with ``--config``, so the repository being analysed never configures the
    parser. ``decode`` turns one printed match into the dict the caller reads, in every run, side by
    side or alone; a caller that reads few fields passes a decoder that skips the rest.

    Every run is stopped once it outlasts ``guard_seconds`` (``runaway_guards``). A file parsed alone
    that trips the guard is added to ``refused`` with ``parse_guard_reason``. Side-by-side files parse
    in well under a second each, so a stopped side-by-side run holds a runaway file: each of its files
    that had not finished is parsed again alone under the same guard. ast-grep prints a file's matches
    together once its parse ends and lists the file as scanned at the same moment, so a file that
    printed a match or was listed had finished and is never parsed twice."""
    limit = memory_limit.process_guard().limit
    placed = _placed(files, cwd, limit.single_parse_mb)
    refused.update(placed.refused)
    if not placed.side_by_side and not placed.alone:
        return
    with ExitStack() as resources:
        directory = resources.enter_context(tempfile.TemporaryDirectory(prefix="jev-navigator-sgconfig-"))
        path = Path(directory) / "sgconfig.yml"
        path.write_text(NEUTRAL_AST_GREP_CONFIG if config is None else config)
        resources.enter_context(memory_limit.parsing())
        command = [AST_GREP, "scan", "--inline-rules", rules_yaml, "--config", str(path)]
        scan = _Scan([*command, "--threads", "1"], cwd, refused, decode, guard_seconds)
        side_by_side = [*command, "--threads", str(limit.parse_threads)]
        for chunk in file_chunks(placed.side_by_side):
            yield from scan.side_by_side(side_by_side, chunk)
        for file in placed.alone:
            yield from scan.alone(file)


@dataclass(frozen=True)
class _Scan:
    """What every ast-grep run of one ``ast_grep_rules`` call shares."""

    alone_command: Sequence[str]
    cwd: Path
    refused: dict[str, str]
    decode: Callable[[str], dict]
    guard_seconds: float

    def side_by_side(self, command: Sequence[str], files: Sequence[str]) -> Iterator[dict]:
        try:
            yield from self.run(command, files)
        except _ParseGuardTrippedError as tripped:
            for file in tripped.unfinished:
                yield from self.alone(file)

    def alone(self, file: str) -> Iterator[dict]:
        try:
            yield from self.run(self.alone_command, [file])
        except _ParseGuardTrippedError:
            self.refused[file] = parse_guard_reason(self.guard_seconds)

    def run(self, command: Sequence[str], files: Sequence[str]) -> Iterator[dict]:
        """The matches of one ast-grep run over ``files``, each decoded with ``decode``; a file it
        skipped without parsing is added to ``refused`` when the run ends."""
        yield from _json_lines(
            [*command, "--json=stream", "--inspect=entity", "--", *files],
            self.cwd,
            decode=self.decode,
            on_stderr=lambda inspection: self.refused.update(_skipped_files(files, inspection, self.cwd)),
            guarded=_Guarded(files, self.guard_seconds),
        )


@dataclass
class _Guarded:
    """One run's files, the ones that printed a match, and whether the guard stopped the run."""

    files: Sequence[str]
    seconds: float
    printed: set[str] = field(default_factory=set)
    tripped: threading.Event = field(default_factory=threading.Event)

    def unfinished(self, inspection: str) -> list[str]:
        finished = self.printed | _scanned_files(inspection)
        return [file for file in self.files if file not in finished]


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
    scanned = _scanned_files(inspection)
    reasons = {file: _not_parsed_reason(cwd / file) for file in chunk if file not in scanned}
    return {file: reason for file, reason in reasons.items() if reason is not None}


def _scanned_files(inspection: str) -> set[str]:
    """The files ``--inspect=entity`` lists as scanned."""
    return {
        line.removeprefix(_SCANNED_FILE_PREFIX).rsplit(": language=", 1)[0]
        for line in inspection.splitlines()
        if line.startswith(_SCANNED_FILE_PREFIX)
    }


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
    guarded: _Guarded,
    on_stderr: Callable[[str], None] | None = None,
) -> Iterator[dict]:
    """Each line the command prints, decoded with ``decode`` while it runs. stderr goes to a file, so a full
    stderr pipe cannot stall the command; the process is killed if the reader stops early. A line
    that is no JSON (the process died partway through it) fails with the process's exit code and
    stderr, which say why it stopped. ``on_stderr`` gets the whole stderr text once the command has
    ended successfully. A command still running after ``guarded.seconds`` is stopped and raises
    ``_ParseGuardTrippedError`` naming its unfinished files, after the matches it printed."""
    with tempfile.TemporaryFile() as errors:
        with memory_limit.started(
            arguments, cwd=cwd, stdout=subprocess.PIPE, stderr=errors, text=True
        ) as process:
            guard = threading.Timer(guarded.seconds, _stop, (process, guarded.tripped))
            guard.start()
            try:
                for line in process.stdout:
                    if line.strip():
                        match = _json_object(line, process, errors, arguments[0], decode)
                        guarded.printed.add(match.get("file"))
                        yield match
            finally:
                guard.cancel()
        if guarded.tripped.is_set():
            raise _ParseGuardTrippedError(guarded.unfinished(_stderr_text(errors)))
        if process.returncode not in (0, _NO_MATCHES_EXIT):
            raise _tool_failed(arguments[0], process.returncode, errors)
        if on_stderr is not None:
            on_stderr(_stderr_text(errors))


def _stop(process: subprocess.Popen, tripped: threading.Event) -> None:
    tripped.set()
    process.kill()


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
    text: str,
    files: Sequence[str],
    cwd: Path,
    max_hits: int | None,
    context_bytes: int,
    *,
    whole_word: bool = False,
) -> list[TextHit]:
    """The lines holding ``text``, at most ``max_hits`` per file or every one when it is None, each as
    the bytes around one hit: up to ``context_bytes`` before and after, so a one-line bundle costs no
    more than a short line. ``whole_word`` keeps only hits no word character touches. The match runs
    on to the end of the line, so each line matches once, and ``--replace`` prints only its window;
    ripgrep's JSON would carry the whole line."""
    pattern = _hit_window(text, context_bytes, whole_word)
    per_file = [] if max_hits is None else ["--max-count", str(max_hits)]
    command = [
        *RIPGREP_SAFE,
        "--text",
        "--only-matching",
        "--line-number",
        "--with-filename",
        "--null",
        *per_file,
    ]
    command += ["--replace", "$window", "--regexp", pattern, "--"]
    hits: dict[tuple[str, int], TextHit] = {}
    for chunk in file_chunks(files, bytes_only=True):
        for hit in _windows(command_output([*command, *chunk], cwd, no_match_exit=_NO_MATCHES_EXIT)):
            hits.setdefault((hit.file, hit.line), hit)
    return list(hits.values())


def ripgrep_term_lines(texts: Sequence[str], files: Sequence[str], cwd: Path) -> list[TextHit]:
    """Line identities matching any literal, without printing potentially huge source lines."""
    pattern = "(?:" + "|".join(re.escape(text) for text in texts) + ")"
    command = [
        *RIPGREP_SAFE,
        "--text",
        "--only-matching",
        "--line-number",
        "--with-filename",
        "--null",
        "--replace",
        "$term",
        "--file",
        "-",
        "--",
    ]
    hits = {}
    for chunk in file_chunks(files, bytes_only=True):
        for hit in _windows(
            command_output(
                [*command, *chunk], cwd, no_match_exit=_NO_MATCHES_EXIT, stdin=f"(?P<term>{pattern})(?-u:.)*"
            )
        ):
            hits.setdefault((hit.file, hit.line), hit)
    return list(hits.values())


def _hit_window(text: str, context_bytes: int, whole_word: bool) -> str:
    """A regular expression capturing, as ``window``, ``text`` with up to ``context_bytes`` of any
    bytes on either side, then matching the rest of the line. The text's characters other than
    letters, digits and underscores are written as code points, so no character of it is read as
    syntax."""
    context = f"(?-u:.){{0,{context_bytes}}}"
    literal = "".join(char if char.isalnum() or char == "_" else f"\\x{{{ord(char):x}}}" for char in text)
    if whole_word:
        literal = rf"(?:^|\W){literal}(?:\W|$)"
    return f"(?P<window>{context}{literal}{context})(?-u:.)*"


def _windows(output: bytes) -> Iterator[TextHit]:
    """The hits of ripgrep's ``--null`` printer, ``path NUL line:window`` per line of output. A window
    holds no newline, and a path ends at its NUL, so a newline in a path cannot split a record.
    Bytes that are not UTF-8 are decoded the way the index reads files, with invalid bytes replaced."""
    position = 0
    while position < len(output):
        path_end = output.index(b"\0", position)
        number_end = output.index(b":", path_end)
        window_end = output.find(b"\n", number_end)
        window_end = len(output) if window_end < 0 else window_end
        path = output[position:path_end].decode(errors="replace").removeprefix("./")
        window = output[number_end + 1 : window_end].decode(errors="replace").rstrip("\r")
        yield TextHit(path, int(output[path_end + 1 : number_end]), window)
        position = window_end + 1


def ripgrep_lines(texts: Sequence[str], files: Sequence[str], cwd: Path) -> list[TextHit]:
    """Every line of the supplied files holding any of the exact ``texts``, all texts searched in one
    pass over the files."""
    if not files or not texts:
        return []
    hits = []
    with _pattern_file(texts) as patterns:
        command = [*RIPGREP_SAFE, "--json", "--fixed-strings", "-f", patterns]
        for chunk in file_chunks(files, bytes_only=True):
            hits += _match_lines(run_command([*command, "--", *chunk], cwd, no_match_exit=_NO_MATCHES_EXIT))
    return hits


def ripgrep_files(texts: str | Sequence[str], files: Sequence[str], cwd: Path) -> tuple[str, ...]:
    """Every supplied file containing any of the exact ``texts``, without a result-count cutoff. Only
    paths come back, and ripgrep stops reading a file at its first match, so a 20 MB one-line bundle
    costs what a small file does; ``ripgrep_lines`` would print that line with every match."""
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


def _match_lines(output: str) -> list[TextHit]:
    """The match events of ripgrep's JSON output. Events are split at newlines only, since a line of
    code may hold a Unicode line separator that ``str.splitlines`` would split."""
    events = (json.loads(line) for line in output.split("\n") if line.strip())
    return [_text_hit(event["data"]) for event in events if event.get("type") == "match"]


def inside_git_worktree(cwd: Path) -> bool:
    """Whether ``cwd`` lies in a Git worktree. Only git's own "not a git repository" means no; any
    other failure, such as a repository git refuses for dubious ownership, is raised, so it is never
    listed as a plain directory. Git runs in the C locale so that message is never translated."""
    with memory_limit.started(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    ) as process:
        output, error_output = process.communicate()
    if process.returncode == 0:
        return output.strip() == "true"
    if "not a git repository" in error_output:
        return False
    raise _tool_failure("git", process.returncode, error_output)


def head_commit(cwd: Path) -> str:
    """HEAD's commit in the Git worktree at ``cwd``; empty before its first commit, which is the one
    case `git rev-parse -q --verify` reports with exit 1 and no message. Any other failure raises."""
    return run_command(["git", "rev-parse", "-q", "--verify", "HEAD"], cwd, no_match_exit=1).strip()


def _text_hit(match: dict) -> TextHit:
    return TextHit(_decoded(match["path"]), match["line_number"], _decoded(match["lines"]).rstrip("\r\n"))


def _decoded(field: dict) -> str:
    """ripgrep reports a path or line that is not valid UTF-8 as base64 ``bytes`` instead of ``text``;
    it is decoded the way the index reads files, with invalid bytes replaced."""
    if "text" in field:
        return field["text"]
    return base64.b64decode(field["bytes"]).decode("utf-8", errors="replace")


def git(
    arguments: Sequence[str], cwd: Path, *, stdin: str | None = None, timeout: float | None = None
) -> str:
    return run_command(["git", *arguments], cwd, stdin=stdin, timeout=timeout)


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
    with memory_limit.started(
        ["git", "cat-file", "--batch"],
        cwd=repository,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        output, error_output = process.communicate(requests)
    if process.returncode != 0:
        raise _tool_failure("git cat-file", process.returncode, error_output.decode(errors="replace"))
    return _batch_contents(output)


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
