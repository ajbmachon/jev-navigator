"""JVN child-process allowances and their shared machine ceiling.

Every process JVN starts is started through ``started``: the index's ast-grep, ripgrep and git, the
git call that records a pack's source revision (``cli``), and a model step's command-line connector
(``connectors``). The first started process takes one of the machine's memory slots, waiting up to
``wait_seconds`` for room. The process keeps the slot until it exits, and the operating system releases
the lock then, however the process ends, so no slot outlives its holder. The ceiling holds
``ceiling_mb // allowance_mb`` slots, one ``flock``-ed file each, in a folder that no HOME setting
moves.

While a started process runs, a watchdog thread measures the physical footprint of every process
JVN started, every ``SAMPLE_SECONDS``. Over the allowance it kills those processes, and each affected
``started`` block raises ``MemoryLimitReachedError``.

The allowance and shared slots cover JVN's child processes. The embedding host owns its Python
process's total memory, including JVN's indexes and the host's other caches, and must enforce that
budget at its job or process boundary. Process footprint cannot distinguish those allocations.
An index's lifetime does not make unrelated host growth JVN's. The same division applies to the
standalone command: its launcher owns the Python process budget.

The footprint is the operating system's physical footprint: libproc's ``ri_phys_footprint`` on macOS,
which counts compressed pages that the resident size leaves out, and resident plus swapped memory on
Linux.
"""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import os
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .index.file_shape import MAX_PARSE_PEAK_MB

ALLOWANCE_MB = 1024
"""What the child processes of one JVN host may hold together. Measured on 04.10.2026 with the streaming
parser (#50) and the declaration-rule fix (#72), parsing every file of an app-sized scope: the largest
measured 15.6 MB scope peaked at 333 MB; generated bundles parsed side by side peaked at 471 MB.
Sample provenance is retained in ``measurements/runtime-limits/REPORT.md``."""

CEILING_MB = 8192
"""The shared ceiling of JVN child-process reservations: eight slots at the default allowance. All
agent work on the machine shares 10 to 15 GB, and André chose 8 GB for JVN (04.10.2026) because a slot
is a reservation: with four, a fifth run waited while four runs of about 300 MB used only 1.2 GB."""

WAIT_SECONDS = 120.0
"""How long a process waits for a free slot before it refuses (André, 04.10.2026)."""

SAMPLE_SECONDS = 0.05
"""How often the watchdog measures. At the 800 MB per second measured in the crash of 03.10.2026, a
process overshoots its allowance by about 40 MB before it is stopped."""

SLOT_POLL_SECONDS = 0.2

PARSE_HEADROOM_MB = 270
"""Conservative headroom when sizing ast-grep concurrency and individual files. Originally based on
a measured 261 MB Python parsing share, it remains parser sizing slack, not a host-heap charge.
Sample provenance is retained in ``measurements/runtime-limits/REPORT.md``."""

ENVIRONMENT_NAMES = {
    "allowance_mb": "JEV_NAVIGATOR_MEMORY_ALLOWANCE_MB",
    "ceiling_mb": "JEV_NAVIGATOR_MEMORY_CEILING_MB",
    "wait_seconds": "JEV_NAVIGATOR_MEMORY_WAIT_SECONDS",
}
SLOTS_DIR_VARIABLE = "JEV_NAVIGATOR_MEMORY_SLOTS_DIR"

CHILD_ENVIRONMENT_NAMES = frozenset({"PATH", "HOME", "TMPDIR", "LANG", "LANGUAGE", "TZ"})
CHILD_ENVIRONMENT_PREFIXES = ("LC_", "GIT_")
"""What a process JVN starts over code inherits from its host: where programs are found, the home git
reads its configuration from, the temporary folder, the locale (``LC_*``) and git's own settings
(``GIT_*``, such as which configuration files git reads and so which repositories it trusts). A host may
hold model keys, bearer tokens and proxy credentials while the code JVN reads is untrusted, so
ast-grep, ripgrep and git get nothing else. A host that keeps a credential in a ``GIT_*`` variable
hands it to JVN's git. A model step's command-line connector passes the host's whole environment
itself, because the model CLI needs its credentials."""

_MB = 2**20


class MemoryLimitReachedError(RuntimeError):
    """JVN stopped because its memory allowance or the machine's JVN ceiling was reached. Resuming
    continues the work once there is room."""


@dataclass(frozen=True)
class MemoryLimit:
    allowance_mb: int = ALLOWANCE_MB
    ceiling_mb: int = CEILING_MB
    wait_seconds: float = WAIT_SECONDS

    def __post_init__(self) -> None:
        if self.allowance_mb <= 0:
            raise ValueError(f"the memory allowance must be positive, got {self.allowance_mb} MB")
        if self.ceiling_mb < self.allowance_mb:
            raise ValueError(
                f"the memory ceiling of {self.ceiling_mb} MB is below the allowance of {self.allowance_mb} MB"
            )
        if self.wait_seconds < 0:
            raise ValueError(f"the wait for a memory slot cannot be negative, got {self.wait_seconds} s")

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> MemoryLimit:
        """Library defaults, overridden by the ``JEV_NAVIGATOR_MEMORY_*`` variables."""
        environment = os.environ if environment is None else environment
        found: dict[str, float] = {
            field: (float if field == "wait_seconds" else int)(environment[name])
            for field, name in ENVIRONMENT_NAMES.items()
            if name in environment
        }
        return cls(**found)

    @property
    def slots(self) -> int:
        return self.ceiling_mb // self.allowance_mb

    @property
    def parse_threads(self) -> int:
        """How many files ast-grep may parse at once: the allowance less parser headroom, divided by the
        largest parse the guard admits; at least one. It assumes one parse per process, which
        ``parsing`` keeps."""
        return max(1, int((self.allowance_mb - PARSE_HEADROOM_MB) // MAX_PARSE_PEAK_MB))

    @property
    def single_parse_mb(self) -> int:
        """The largest estimated parse peak of a file ast-grep parses alone, on one thread with no other
        file beside it: the allowance less parser headroom. The parse guard refuses any file above it."""
        return self.allowance_mb - PARSE_HEADROOM_MB


def slots_directory(environment: Mapping[str, str] | None = None) -> Path:
    """``$JEV_NAVIGATOR_MEMORY_SLOTS_DIR``, or a folder of this user in /tmp. Never under HOME: processes
    started with a HOME of their own would each get a ceiling of their own."""
    environment = os.environ if environment is None else environment
    return Path(environment.get(SLOTS_DIR_VARIABLE) or f"/tmp/jev-navigator-memory-{os.getuid()}")


def child_environment() -> dict[str, str]:
    """The entries of this process's environment that ``CHILD_ENVIRONMENT_NAMES`` and
    ``CHILD_ENVIRONMENT_PREFIXES`` admit, as a new dict."""
    return {
        name: value
        for name, value in os.environ.items()
        if name in CHILD_ENVIRONMENT_NAMES or name.startswith(CHILD_ENVIRONMENT_PREFIXES)
    }


def process_guard() -> MemoryGuard:
    """This process's guard for the settings its environment holds now; a process's settings do not
    change, so it is always the same guard."""
    key = (MemoryLimit.from_env(), slots_directory())
    with _GUARDS_LOCK:
        guard = _GUARDS.get(key)
        if guard is None:
            guard = _GUARDS[key] = MemoryGuard(*key)
        return guard


def started(arguments: Sequence[str], **options) -> Iterator[subprocess.Popen]:
    """``MemoryGuard.started`` of this process's guard."""
    return process_guard().started(arguments, **options)


def check() -> None:
    """``MemoryGuard.check`` of this process's guard."""
    process_guard().check()


@contextmanager
def parsing() -> Iterator[None]:
    """This process's one parse at a time. ``parse_threads`` and ``single_parse_mb`` each spend the
    whole allowance on one ast-grep, so a scan in another thread waits for this one to finish instead
    of both running and the allowance stopping every scan."""
    with _ONE_PARSE:
        yield


class MemoryGuard:
    def __init__(self, limit: MemoryLimit, directory: Path) -> None:
        self.limit = limit
        self.directory = directory
        self._lock = threading.Lock()
        self._slot_lock = threading.Lock()
        self._slot: _Slot | None = None
        self._children: set[subprocess.Popen] = set()
        self._stopped: dict[subprocess.Popen, str] = {}

    def check(self) -> None:
        """Reserve this host's child-process allowance in the shared ceiling."""
        self._hold_slot()

    @contextmanager
    def started(self, arguments: Sequence[str], **options) -> Iterator[subprocess.Popen]:
        """The process ``subprocess.Popen(arguments, **options)``, watched until the block ends. Without an
        ``env`` option it gets ``child_environment()``. On leaving the block the process has ended: it is
        killed if the block raised, and waited for otherwise. A process the watchdog stopped raises
        ``MemoryLimitReachedError`` here, never its partial output."""
        self.check()
        process = subprocess.Popen(list(arguments), **{"env": child_environment(), **options})
        done = self._watch(process)
        try:
            yield process
        except BaseException:
            process.kill()
            raise
        finally:
            _close_streams(process)
            process.wait()
            reason = self._unwatch(process, done)
            if reason is not None:
                raise MemoryLimitReachedError(reason)

    @property
    def _allowance_bytes(self) -> int:
        return self.limit.allowance_mb * _MB

    def _hold_slot(self) -> None:
        """Take a slot, replacing a deleted slot file. Waiting holds only the slot lock, so the
        watchdog keeps watching running children."""
        with self._slot_lock:
            if self._slot is not None and self._slot.still_ours():
                return
            if self._slot is not None:
                self._slot.file.close()
                self._slot = None
            self._slot = _take_slot(self.directory, self.limit)

    def _watch(self, process: subprocess.Popen) -> threading.Event:
        with self._lock:
            self._children.add(process)
        done = threading.Event()
        threading.Thread(target=self._sample_until, args=(done,), name="jvn-memory", daemon=True).start()
        return done

    def _unwatch(self, process: subprocess.Popen, done: threading.Event) -> str | None:
        done.set()
        with self._lock:
            self._children.discard(process)
            return self._stopped.pop(process, None)

    def _sample_until(self, done: threading.Event) -> None:
        while not done.wait(SAMPLE_SECONDS):
            with self._lock:
                children = tuple(self._children)
            sizes = {child: footprint(child.pid) for child in children}
            if sum(sizes.values()) > self._allowance_bytes:
                self._stop(sizes)
                return

    def _stop(self, sizes: Mapping[subprocess.Popen, int]) -> None:
        reason = self._over_allowance(sizes)
        with self._lock:
            stopping = [child for child in sizes if child in self._children]
            for child in stopping:
                self._stopped[child] = reason
        for child in stopping:
            child.kill()

    def _over_allowance(self, sizes: Mapping[subprocess.Popen, int]) -> str:
        held = ", ".join(
            f"{_name(child)} (process {child.pid}) held {_mb(size)}" for child, size in sizes.items()
        )
        names = ", ".join(_name(child) for child in sizes)
        return (
            f"JVN reached its child-process memory allowance of {self.limit.allowance_mb:,} MB "
            f"with {_mb(sum(sizes.values()))} in use: {held}; it stopped {names}. "
            f"The host's Python process is outside this allowance; its launcher owns that budget. "
            f"Narrow the scope or raise {ENVIRONMENT_NAMES['allowance_mb']}, then resume"
        )


@dataclass(frozen=True)
class _Slot:
    path: Path
    file: BinaryIO

    def still_ours(self) -> bool:
        """A slot file deleted while held (a /tmp cleaner, say) keeps its lock, but a new file at its
        path lets another process in, so a lock counts only on the file its path still names."""
        try:
            return os.stat(self.path, follow_symlinks=False).st_ino == os.fstat(self.file.fileno()).st_ino
        except FileNotFoundError:
            return False


def _take_slot(directory: Path, limit: MemoryLimit) -> _Slot:
    """One of the ceiling's slots, locked for as long as this process lives, waiting up to the limit's
    wait for one to free."""
    _make_private(directory)
    deadline = time.monotonic() + limit.wait_seconds
    while True:
        holders = []
        for number in range(limit.slots):
            path = directory / f"slot-{number}"
            slot = _lock(path)
            if slot is not None:
                slot.file.truncate(0)
                slot.file.write(f"{os.getpid()}\n".encode())
                slot.file.flush()
                return slot
            holders.append(_holder(path))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MemoryLimitReachedError(_ceiling_full(directory, limit, holders))
        time.sleep(min(SLOT_POLL_SECONDS, remaining))


def _make_private(directory: Path) -> None:
    """The default folder is a predictable path in the shared /tmp, so another user could create it
    first: it must be a real folder of this user, never a link to somewhere else."""
    with contextlib.suppress(FileExistsError):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = os.lstat(directory)
    if stat.S_ISLNK(status.st_mode):
        raise PermissionError(f"JVN's memory slot folder {directory} is a symbolic link")
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
        raise PermissionError(f"JVN's memory slot folder {directory} is not a folder of this user")


def _lock(path: Path) -> _Slot | None:
    slot = _Slot(path, os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "r+b"))
    try:
        fcntl.flock(slot.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        slot.file.close()
        return None
    if not slot.still_ours():
        slot.file.close()
        return None
    return slot


def _holder(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _ceiling_full(directory: Path, limit: MemoryLimit, holders: Sequence[str]) -> str:
    named = [holder for holder in holders if holder]
    if len(holders) == 1:
        held = f"process {named[0] if named else 'unknown'} holds its one slot of {limit.allowance_mb:,} MB"
    else:
        held = (
            f"processes {_listed(named) if named else 'unknown'} hold its {len(holders)} slots of "
            f"{limit.allowance_mb:,} MB each"
        )
    return (
        f"JVN's shared memory ceiling of {limit.ceiling_mb:,} MB is full: {held} in {directory}. "
        f"Waited {limit.wait_seconds:g} s. Resume once one of them ends, or raise "
        f"{ENVIRONMENT_NAMES['ceiling_mb']}"
    )


def _listed(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _close_streams(process: subprocess.Popen) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _name(process: subprocess.Popen) -> str:
    return Path(str(process.args[0])).name


def _mb(size: int) -> str:
    return f"{size / _MB:,.0f} MB"


if sys.platform == "darwin":
    _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
    _RUSAGE_INFO_V2 = 2

    class _RusageInfoV2(ctypes.Structure):
        """``struct rusage_info_v2`` from <sys/resource.h>; ``rest`` holds the ten counters after the
        footprint."""

        _fields_ = [
            ("uuid", ctypes.c_uint8 * 16),
            *(
                (name, ctypes.c_uint64)
                for name in (
                    "user_time",
                    "system_time",
                    "pkg_idle_wkups",
                    "interrupt_wkups",
                    "pageins",
                    "wired_size",
                    "resident_size",
                    "phys_footprint",
                )
            ),
            ("rest", ctypes.c_uint64 * 10),
        ]

    def footprint(pid: int) -> int:
        """The process's physical footprint in bytes, compressed pages included; 0 once it has ended."""
        usage = _RusageInfoV2()
        if _LIBPROC.proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(usage)) != 0:
            return 0
        return usage.phys_footprint

elif sys.platform.startswith("linux"):

    def footprint(pid: int) -> int:
        """The process's resident plus swapped memory in bytes; 0 once it has ended."""
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return 0
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        return sum(int(fields[name].split()[0]) * 1024 for name in ("VmRSS", "VmSwap") if name in fields)

else:

    def footprint(pid: int) -> int:
        raise OSError(f"JVN measures memory only on macOS and Linux, not on {sys.platform}")


_GUARDS: dict[tuple[MemoryLimit, Path], MemoryGuard] = {}
_GUARDS_LOCK = threading.Lock()
_ONE_PARSE = threading.Lock()
