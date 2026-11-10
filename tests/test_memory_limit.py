"""JVN's memory limit, proved with real processes: a child that really grows, real ast-grep parses,
and real JVN processes contending for real slot files."""

from __future__ import annotations

import asyncio
import json
import mmap
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from git_repos import commit_files
from test_oversized_guard import _real_code_file

from jev_navigator import cli, memory_limit
from jev_navigator.connectors import CommandConnector
from jev_navigator.directives.find_code import Outcome, SearchBudget, find_code, find_code_async
from jev_navigator.directives.places import place_for_line
from jev_navigator.index import tools
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.fact_cache import FactCache
from jev_navigator.index.spans import Span
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.round import RoundRegistration, freeze
from jev_navigator.memory_limit import MemoryLimit, MemoryLimitReachedError
from jev_navigator.runaway_guards import PARSE_GUARD_SECONDS
from jev_navigator.testing import ScriptedJevClient

MB = 2**20

GROWS_TO_600_MB_THEN_STAYS = """\
import time
held = []
for _ in range(20):
    held.append(b"x" * (30 * 2**20))
    time.sleep(0.03)
time.sleep(30)
"""

GROWS_WHEN_TRIGGERED_THEN_STAYS = """\
import sys
import time
from pathlib import Path
trigger = Path(sys.argv[1])
while not trigger.exists():
    time.sleep(0.02)
held = []
for _ in range(20):
    held.append(b"x" * (30 * 2**20))
    time.sleep(0.03)
time.sleep(30)
"""

GROWS_TO_50_MB_THEN_ENDS = """\
import time
held = b"x" * (50 * 2**20)
time.sleep(0.5)
"""

PRINTS_A_LINE_THEN_STAYS = """\
import time
print('{"line": 1}', flush=True)
time.sleep(30)
"""

TAKES_A_SLOT_AND_HOLDS_IT = """\
import subprocess
import sys
from jev_navigator import memory_limit
helper = [sys.executable, "-c", "import sys; sys.stdin.read()"]
with memory_limit.started(helper, stdin=subprocess.PIPE):
    print("holding", flush=True)
    sys.stdin.read()
"""

TAKES_A_SLOT = """\
import time
from pathlib import Path
from jev_navigator.index import tools
started = time.monotonic()
tools.git(["--version"], Path.cwd())
print(f"took a slot after {time.monotonic() - started:.1f} s", flush=True)
"""

WAITS_FOR_A_SLOT = """\
import time
from pathlib import Path
from jev_navigator.index import tools
print("asking for a slot", flush=True)
started = time.monotonic()
tools.git(["--version"], Path.cwd())
print(f"took a slot after {time.monotonic() - started:.1f} s", flush=True)
"""

STATEMENT = "export function launch(){return 1};"


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every command a process was started with; each one still runs."""
    started: list[list[str]] = []

    class RecordedPopen(subprocess.Popen):
        def __init__(self, arguments, *args, **kwargs) -> None:
            started.append(list(arguments))
            super().__init__(arguments, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", RecordedPopen)
    return started


def _scans(commands: list[list[str]]) -> list[list[str]]:
    return [command for command in commands if command[:2] == [tools.AST_GREP, "scan"]]


def _touched(megabytes: int) -> mmap.mmap:
    """``megabytes`` of memory this process really holds until the map is closed. A freed Python object
    can stay in the footprint and be reused, which would hide a later test's growth."""
    held = mmap.mmap(-1, megabytes * MB)
    for page in range(0, megabytes * MB, mmap.PAGESIZE):
        held[page] = 1
    return held


@contextmanager
def _holding(megabytes: int) -> Iterator[None]:
    """This process grows by ``megabytes`` of touched memory, returned to the system on exit."""
    held = _touched(megabytes)
    try:
        yield
    finally:
        held.close()


def _three_file_repository(root: Path) -> Path:
    """The target is in ``admit``, two steps from the start; opening ``admit`` looks up ``record`` in a
    file nothing before it needed, so the second round must start a process."""
    commit_files(
        root,
        {
            "app/__init__.py": "",
            "app/entry.py": "from .policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "app/policy.py": (
                "from .audit import record\n\ndef admit(item):\n    record(item)\n    return len(item) <= 3\n"
            ),
            "app/audit.py": "def record(item):\n    return item\n",
        },
    )
    return root


def _limit_the_process(
    monkeypatch: pytest.MonkeyPatch, slots: Path, allowance_mb: int, ceiling_mb: int, wait_seconds: float = 0
) -> None:
    monkeypatch.setenv("JEV_NAVIGATOR_MEMORY_ALLOWANCE_MB", str(allowance_mb))
    monkeypatch.setenv("JEV_NAVIGATOR_MEMORY_CEILING_MB", str(ceiling_mb))
    monkeypatch.setenv("JEV_NAVIGATOR_MEMORY_WAIT_SECONDS", str(wait_seconds))
    monkeypatch.setenv("JEV_NAVIGATOR_MEMORY_SLOTS_DIR", str(slots))


def _jvn_process(script: str, slots: Path, **settings: str) -> subprocess.Popen:
    environment = {
        **os.environ,
        "JEV_NAVIGATOR_MEMORY_SLOTS_DIR": str(slots),
        **{f"JEV_NAVIGATOR_MEMORY_{name.upper()}": value for name, value in settings.items()},
    }
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def _model_step(arguments: list[str], folder: Path) -> str:
    return CommandConnector(arguments, name="model step").complete("a prompt")


@pytest.mark.parametrize("run", [tools.run_command, _model_step], ids=["index tool", "model step connector"])
def test_a_child_that_grows_past_the_allowance_is_stopped_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run
) -> None:
    # Arrange
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=150, ceiling_mb=150)
    started = time.monotonic()

    # Act
    with pytest.raises(MemoryLimitReachedError) as stopped:
        run([sys.executable, "-c", GROWS_TO_600_MB_THEN_STAYS], tmp_path)

    # Assert
    message = str(stopped.value)
    assert "memory allowance of 150 MB" in message
    assert "child-process" in message
    assert "stopped" in message and os.path.basename(sys.executable) in message
    assert time.monotonic() - started < 5


def test_a_reader_that_stops_early_does_not_wait_for_its_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    started_processes: list[subprocess.Popen] = []

    class RecordedPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            started_processes.append(self)

    monkeypatch.setattr(subprocess, "Popen", RecordedPopen)
    lines = tools._json_lines(
        [sys.executable, "-c", PRINTS_A_LINE_THEN_STAYS],
        tmp_path,
        decode=json.loads,
        guarded=tools._Guarded((), PARSE_GUARD_SECONDS),
    )
    started = time.monotonic()

    # Act
    first = next(lines)
    lines.close()

    # Assert
    assert first == {"line": 1}
    assert time.monotonic() - started < 5
    assert [process.returncode for process in started_processes] == [-signal.SIGKILL]


def test_a_real_parse_over_the_allowance_is_stopped_and_caches_no_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: one line of 39,000 characters is within the parse guard's side-by-side share, so
    # ast-grep parses it, and its real parse peak is far over a 40 MB allowance.
    bundle = "dist/bundle.js"
    commit_files(
        tmp_path / "repo", {bundle: (STATEMENT * 2000)[:39_000], "src/small.py": "def a():\n    return 1\n"}
    )
    facts = tmp_path / "facts"
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=40, ceiling_mb=40)
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=facts)

    # Act
    with pytest.raises(MemoryLimitReachedError, match="ast-grep"):
        index.functions_in(bundle)

    # Assert
    content = (tmp_path / "repo" / bundle).read_bytes()
    assert FactCache(facts).load(bundle, content) is None
    assert bundle not in index.parsed_files
    assert bundle not in index.unavailable_files


def test_host_growth_with_a_live_index_does_not_refuse_a_second_pack(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=100, ceiling_mb=100)
    index = CodeIndex(sample_repo, ["app/orders.py", "app/validation.py"], fact_cache_dir=tmp_path / "facts")
    first = index.functions_in("app/orders.py")

    with _holding(200):
        second = index.functions_in("app/validation.py")
        assert tools.git(["--version"], sample_repo).startswith("git version")

    assert [span.name for span in first] == ["place", "cancel"]
    assert second
    assert index.parsed_files == {"app/orders.py", "app/validation.py"}


def test_a_second_jvn_process_waits_for_a_full_ceiling_then_refuses_naming_the_holder(tmp_path: Path) -> None:
    # Arrange: one slot, held by a real JVN process.
    slots = tmp_path / "slots"
    one_slot = {"allowance_mb": "512", "ceiling_mb": "512"}
    holder = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, **one_slot)
    assert holder.stdout.readline() == "holding\n"

    # Act
    started = time.monotonic()
    refused = _jvn_process(TAKES_A_SLOT, slots, wait_seconds="2", **one_slot)
    _, refusal = refused.communicate(timeout=30)
    waited = time.monotonic() - started
    holder.kill()
    holder.wait()

    # Assert
    assert refused.returncode == 1
    assert "MemoryLimitReachedError" in refusal
    assert f"process {holder.pid} holds" in refusal
    assert "ceiling of 512 MB" in refusal
    assert 2 <= waited < 20


def test_a_waiting_process_takes_the_slot_once_the_holder_ends(tmp_path: Path) -> None:
    # Arrange
    slots = tmp_path / "slots"
    one_slot = {"allowance_mb": "512", "ceiling_mb": "512"}
    holder = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, **one_slot)
    assert holder.stdout.readline() == "holding\n"
    waiting = _jvn_process(WAITS_FOR_A_SLOT, slots, wait_seconds="60", **one_slot)
    assert waiting.stdout.readline() == "asking for a slot\n"
    time.sleep(0.5)

    # Act
    holder.stdin.close()
    holder.wait()
    output, errors = waiting.communicate(timeout=60)

    # Assert
    assert waiting.returncode == 0, errors
    assert output.startswith("took a slot after ")
    assert float(output.split()[4]) >= 0.4


def test_a_killed_holder_leaves_no_stale_slot(tmp_path: Path) -> None:
    # Arrange
    slots = tmp_path / "slots"
    one_slot = {"allowance_mb": "512", "ceiling_mb": "512"}
    holder = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, **one_slot)
    assert holder.stdout.readline() == "holding\n"

    # Act
    os.kill(holder.pid, signal.SIGKILL)
    holder.wait()
    successor = _jvn_process(TAKES_A_SLOT, slots, wait_seconds="0", **one_slot)
    output, errors = successor.communicate(timeout=30)

    # Assert
    assert successor.returncode == 0, errors
    assert output.startswith("took a slot after ") and float(output.split()[4]) < 1


def test_the_ceiling_admits_as_many_processes_as_its_settings_give_slots_and_no_more(tmp_path: Path) -> None:
    # Arrange: the slot count comes from the settings, never from the test.
    slots_dir = tmp_path / "slots"
    settings = {"allowance_mb": "512", "ceiling_mb": "1536"}
    slots = MemoryLimit.from_env(
        {f"JEV_NAVIGATOR_MEMORY_{name.upper()}": value for name, value in settings.items()}
    ).slots
    holders = [_jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots_dir, **settings) for _ in range(slots)]

    # Act
    try:
        admitted = [holder.stdout.readline() for holder in holders]
        one_more = _jvn_process(TAKES_A_SLOT, slots_dir, wait_seconds="0", **settings)
        _, refusal = one_more.communicate(timeout=30)
    finally:
        for holder in holders:
            holder.kill()
            holder.wait()

    # Assert
    assert admitted == ["holding\n"] * slots
    assert one_more.returncode == 1
    assert f"hold its {slots} slots of 512 MB each" in refusal
    assert all(str(holder.pid) in refusal for holder in holders)


def test_the_watchdog_keeps_watching_while_another_thread_waits_for_a_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: this process holds the only slot and runs a watched child. Then its slot file is
    # deleted, a real JVN process takes the slot, and another thread here waits to take one again.
    slots, trigger = tmp_path / "slots", tmp_path / "grow"
    _limit_the_process(monkeypatch, slots, allowance_mb=150, ceiling_mb=150, wait_seconds=6)
    refusals: dict[str, tuple[float, str]] = {}

    def refused_as(name: str, call) -> threading.Thread:
        def run() -> None:
            try:
                call()
            except MemoryLimitReachedError as error:
                refusals[name] = (time.monotonic(), str(error))

        thread = threading.Thread(target=run)
        thread.start()
        return thread

    grower = [sys.executable, "-c", GROWS_WHEN_TRIGGERED_THEN_STAYS, str(trigger)]
    child = refused_as("child", lambda: tools.run_command(grower, tmp_path))
    time.sleep(0.5)
    (slots / "slot-0").unlink()
    newcomer = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, allowance_mb="150", ceiling_mb="150")
    assert newcomer.stdout.readline() == "holding\n"
    waiter = refused_as("waiter", lambda: tools.git(["--version"], tmp_path))
    time.sleep(0.5)

    # Act
    trigger.touch()
    grew_at = time.monotonic()
    child.join(timeout=60)
    waiter.join(timeout=60)
    newcomer.kill()
    newcomer.wait()

    # Assert
    stopped_at, reason = refusals["child"]
    assert "it stopped" in reason
    assert stopped_at - grew_at < 3
    assert "ceiling of 150 MB" in refusals["waiter"][1]


def test_memory_held_before_the_slot_is_not_charged_to_jvn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commands: list[list[str]]
) -> None:
    # Arrange: a host that already holds 300 MB when it first uses JVN.
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=100, ceiling_mb=100)

    # Act
    with _holding(300):
        output = tools.git(["--version"], tmp_path)
        again = tools.git(["--version"], tmp_path)

    # Assert
    assert output.startswith("git version") and again == output
    assert commands == [["git", "--version"], ["git", "--version"]]


def _opened(repository: Path, facts: Path, how: str) -> CodeIndex:
    if how == "from_git":
        return CodeIndex.from_git(repository, fact_cache_dir=facts)
    return CodeIndex(repository, ["app/orders.py"], fact_cache_dir=facts)


@pytest.mark.parametrize("how", ["from_git", "with_files"])
def test_what_a_host_grows_between_two_searches_is_not_charged_to_the_second(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    # Arrange: a first index takes the slot, parses and is dropped; then the host grows for its own
    # reasons, past the whole allowance.
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=100, ceiling_mb=100)
    first = _opened(sample_repo, tmp_path / "first", how).functions_in("app/orders.py")

    # Act
    with _holding(200):
        second = _opened(sample_repo, tmp_path / "second", how).functions_in("app/orders.py")

    # Assert
    assert [span.name for span in second] == [span.name for span in first] == ["place", "cancel"]


def test_a_process_started_while_no_index_is_alive_is_charged_only_for_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the process took its slot, then grew 60 MB of its own; with no index alive none of it is
    # JVN's, but with the child's own 50 MB it is over the allowance.
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=100, ceiling_mb=100)
    tools.git(["--version"], tmp_path)

    # Act
    with _holding(60):
        output = tools.run_command([sys.executable, "-c", GROWS_TO_50_MB_THEN_ENDS], tmp_path)

    # Assert
    assert output == ""


def test_a_slot_folder_that_is_a_symbolic_link_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the folder is a predictable path under /tmp, so another user could plant a link there.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "slots").symlink_to(elsewhere)
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=512, ceiling_mb=512)

    # Act
    with pytest.raises(PermissionError, match="symbolic link"):
        tools.git(["--version"], tmp_path)

    # Assert
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize(("allowance_mb", "threads"), [(1024, "3"), (512, "1"), (2048, "7")])
def test_ast_grep_gets_the_threads_its_allowance_affords(
    sample_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commands: list[list[str]],
    allowance_mb: int,
    threads: str,
) -> None:
    # Arrange
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=allowance_mb, ceiling_mb=allowance_mb)
    index = CodeIndex.from_git(sample_repo, fact_cache_dir=tmp_path / "facts")

    # Act
    functions = index.functions_in("app/orders.py")

    # Assert
    assert [command[command.index("--threads") + 1] for command in _scans(commands)] == [threads]
    assert [span.name for span in functions] == ["place", "cancel"]


def test_a_ceiling_below_the_allowance_is_refused() -> None:
    with pytest.raises(ValueError, match="ceiling"):
        MemoryLimit.from_env(
            {"JEV_NAVIGATOR_MEMORY_ALLOWANCE_MB": "1024", "JEV_NAVIGATOR_MEMORY_CEILING_MB": "512"}
        )


def test_the_defaults_are_a_gigabyte_per_process_eight_slots_and_two_minutes() -> None:
    limit = MemoryLimit.from_env({})

    assert (limit.allowance_mb, limit.ceiling_mb, limit.slots, limit.wait_seconds) == (1024, 8192, 8, 120.0)
    assert (limit.parse_threads, limit.single_parse_mb) == (3, 754)


def test_the_largest_file_parsed_alone_follows_the_allowance() -> None:
    larger = MemoryLimit.from_env(
        {"JEV_NAVIGATOR_MEMORY_ALLOWANCE_MB": "2048", "JEV_NAVIGATOR_MEMORY_CEILING_MB": "8192"}
    )

    assert larger.single_parse_mb == 1778


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_a_breach_during_a_round_ends_the_search_failed_and_its_resume_finishes_it(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    # Arrange: the uninterrupted search, with its own index and client.
    def answer(question_id: str, question, state) -> float:
        if "candidates[" in question["instructions"]:
            return 0.9
        return 0.95 if "len(order.items) <= limit" in state["slice"]["code"] else 0.05

    def search(index: CodeIndex, client: ScriptedJevClient, starts, resume=None):
        arguments = (index, Judge(client), "the item limit check", starts)
        options = {"budget": SearchBudget(beam_width=1), "resume": resume}
        if entry == "sync":
            return find_code(*arguments, **options)
        return asyncio.run(find_code_async(*arguments, **options))

    def fresh_index() -> CodeIndex:
        return CodeIndex.from_git(sample_repo, fact_cache_dir=tmp_path / f"facts-{time.monotonic_ns()}")

    def start_in(index: CodeIndex):
        return [place_for_line(index, "app/orders.py", 6, "start")]

    uninterrupted_client = ScriptedJevClient(nouls=answer)
    whole = fresh_index()
    uninterrupted = search(whole, uninterrupted_client, start_in(whole))
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=100, ceiling_mb=100)
    index = fresh_index()
    starts = start_in(index)
    client = ScriptedJevClient(nouls=answer)

    # A real JVN-owned child crosses the allowance while the first scan opens.
    def exceed_allowance(name, phase, count):
        if phase == "started":
            tools.run_command([sys.executable, "-c", GROWS_TO_600_MB_THEN_STAYS], sample_repo)

    index.scan_observer = exceed_allowance
    failed = search(index, client, starts)
    index.scan_observer = None
    asked_before_resume = len(client.requests)
    resumed = search(index, client, [], resume=failed)

    # Assert
    assert failed.outcome == Outcome.FAILED
    assert isinstance(failed.failure, MemoryLimitReachedError)
    assert [entry.place.key for entry in failed.not_inspected] == [place.key for place in starts]
    assert asked_before_resume == 0
    assert resumed.outcome == uninterrupted.outcome == Outcome.FOUND
    assert [visit.place_key for visit in resumed.found] == [visit.place_key for visit in uninterrupted.found]
    assert len(client.requests) == len(uninterrupted_client.requests)


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_any_other_error_while_opening_a_round_still_raises(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    # Arrange: the start is parsed, then the parser and ripgrep disappear from the PATH.
    index = CodeIndex.from_git(sample_repo, fact_cache_dir=tmp_path / "facts")
    starts = [place_for_line(index, "app/orders.py", 6, "start")]
    only_git = tmp_path / "bin"
    only_git.mkdir()
    (only_git / "git").symlink_to(shutil.which("git"))
    monkeypatch.setenv("PATH", str(only_git))
    arguments = (index, Judge(ScriptedJevClient()), "the item limit check", starts)

    # Act
    with pytest.raises(FileNotFoundError):
        if entry == "sync":
            find_code(*arguments, budget=SearchBudget(beam_width=1))
        else:
            asyncio.run(find_code_async(*arguments, budget=SearchBudget(beam_width=1)))


def test_a_test_takes_its_memory_slot_in_the_suites_folder_and_never_in_the_machines(
    tmp_path: Path, private_memory_slots: Path
) -> None:
    # Act
    tools.git(["--version"], tmp_path)

    # Assert
    def held_here(folder: Path) -> list[str]:
        own = f"{os.getpid()}\n"
        return [slot.name for slot in sorted(folder.glob("slot-*")) if slot.read_text() == own]

    machine = memory_limit.slots_directory({})
    assert held_here(private_memory_slots)
    assert not machine.is_dir() or held_here(machine) == []


def test_scans_in_parallel_threads_of_one_process_take_turns_and_all_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: eight scans of their own copy of twenty standard-library files. Run together, their
    # parsers outgrow an allowance of 96 MB; one at a time, each fits in it with room to spare.
    library = Path(sysconfig.get_paths()["stdlib"])
    names = sorted(path.name for path in library.glob("*.py"))[:20]
    indexes = [_copied_index(library, names, tmp_path / f"copy-{number}") for number in range(8)]
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=96, ceiling_mb=96)
    outcomes: list[str] = []

    def scan(index: CodeIndex) -> None:
        try:
            index.functions_in_files(names)
            outcomes.append("finished")
        except MemoryLimitReachedError as refusal:
            outcomes.append(str(refusal))

    # Act
    workers = [threading.Thread(target=scan, args=(index,)) for index in indexes]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    # Assert
    assert outcomes == ["finished"] * 8


class _ConcurrentScans:
    """Counts the ast-grep scans this process runs at once, and notes when one parsing a file alone
    (``--threads 1``) has started; every scan still runs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.running = 0
        self.most = 0
        self.alone_started = threading.Event()
        counting = threading.Lock()
        scans = self

        class CountedPopen(subprocess.Popen):
            def __init__(self, arguments, *args, **kwargs) -> None:
                self.counted = arguments[:2] == [tools.AST_GREP, "scan"]
                if self.counted:
                    with counting:
                        scans.running += 1
                        scans.most = max(scans.most, scans.running)
                super().__init__(arguments, *args, **kwargs)
                if self.counted and arguments[arguments.index("--threads") + 1] == "1":
                    scans.alone_started.set()

            def wait(self, timeout=None):
                returncode = super().wait(timeout)
                if self.counted:
                    self.counted = False
                    with counting:
                        scans.running -= 1
                return returncode

        monkeypatch.setattr(subprocess, "Popen", CountedPopen)


def test_a_file_parsed_alone_in_one_thread_and_a_scan_in_another_take_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: one index holds 3 MB of real code, estimated at 270 MB, over the side-by-side share, so
    # it is parsed alone; another holds twenty small files, and its scan starts while that parse runs.
    scans = _ConcurrentScans(monkeypatch)
    commit_files(tmp_path / "big", {"src/big.py": _real_code_file(3_000_000)})
    big = CodeIndex.from_git(tmp_path / "big", fact_cache_dir=tmp_path / "big-facts")
    library = Path(sysconfig.get_paths()["stdlib"])
    names = sorted(path.name for path in library.glob("*.py"))[:20]
    small = _copied_index(library, names, tmp_path / "small")
    parsed_alone: list[Span] = []
    alone = threading.Thread(target=lambda: parsed_alone.extend(big.functions_in("src/big.py")))

    # Act
    alone.start()
    scans.alone_started.wait(30)
    side_by_side = small.functions_in_files(names)
    alone.join()

    # Assert
    assert scans.alone_started.is_set()
    assert scans.most == 1
    assert parsed_alone and side_by_side


def _copied_index(library: Path, names: list[str], root: Path) -> CodeIndex:
    root.mkdir()
    for name in names:
        shutil.copy(library / name, root / name)
    return CodeIndex(root, names, fact_cache_dir=root.parent / f"{root.name}-facts")


ONE_QUESTION = {"keep": {"type": "noul", "instructions": "Does the comment hold?"}}


@pytest.mark.parametrize("caller", ["navigator_fingerprint", "frozen_round"])
def test_the_git_calls_outside_the_index_wait_for_a_memory_slot_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caller: str
) -> None:
    # Arrange: another JVN process holds the only slot, and this one may not wait.
    slots = tmp_path / "slots"
    one_slot = {"allowance_mb": "512", "ceiling_mb": "512"}
    holder = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, **one_slot)
    assert holder.stdout.readline() == "holding\n"
    _limit_the_process(monkeypatch, slots, allowance_mb=512, ceiling_mb=512)

    # Act
    try:
        with pytest.raises(MemoryLimitReachedError) as refused:
            if caller == "navigator_fingerprint":
                cli._navigator_provenance()
            else:
                freeze(
                    tmp_path / "round",
                    RoundRegistration(("case",), ONE_QUESTION, {"escalate_below": 0.8}, "", "3.13"),
                )
    finally:
        holder.kill()
        holder.wait()

    # Assert
    assert str(holder.pid) in str(refused.value)


def test_the_process_guard_follows_the_settings_it_is_read_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Arrange
    _limit_the_process(monkeypatch, tmp_path / "a", allowance_mb=512, ceiling_mb=1024)
    first = memory_limit.process_guard()
    _limit_the_process(monkeypatch, tmp_path / "b", allowance_mb=256, ceiling_mb=256)

    # Act
    second = memory_limit.process_guard()

    # Assert
    assert first is not second
    assert (second.limit.allowance_mb, second.directory) == (256, tmp_path / "b")


def test_host_growth_while_a_child_runs_does_not_stop_the_child(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=100, ceiling_mb=100)
    index = CodeIndex(sample_repo, ["app/orders.py"], fact_cache_dir=tmp_path / "facts")
    index.functions_in("app/orders.py")
    ready = threading.Event()
    release = threading.Event()

    def grow():
        with _holding(200):
            ready.set()
            release.wait(5)

    with memory_limit.started([sys.executable, "-c", "import time; time.sleep(0.5)"]) as child:
        grower = threading.Thread(target=grow)
        grower.start()
        try:
            assert ready.wait(2)
            assert child.wait(timeout=3) == 0
        finally:
            release.set()
            grower.join()
    assert index.parsed_files == {"app/orders.py"}
