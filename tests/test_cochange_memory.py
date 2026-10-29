"""Real Git queries retain requested results without a full peer cache per file."""

from __future__ import annotations

import tracemalloc
from pathlib import Path

from git_repos import commit_all, write_files

from jev_navigator.index.code_index import CodeIndex


def test_small_cochange_queries_do_not_retain_every_files_full_population(tmp_path: Path) -> None:
    names = tuple(f"source-{number:04d}.py" for number in range(300))
    write_files(tmp_path, {name: f"value = {number}\n" for number, name in enumerate(names)})
    commit_all(tmp_path)
    index = CodeIndex.from_git(tmp_path, max_files=len(names))
    tracemalloc.start()
    try:
        for name in names:
            expected = tuple((peer, 1) for peer in names if peer != name)[:5]
            assert index.co_changed_files(name, limit=5) == expected
        retained, _ = tracemalloc.get_traced_memory()
        # The original cache retains 300 complete peer tuples, over 8 MiB.
        # This bound measures retained Python memory, never limits customer evidence.
        assert retained < 2 * 1024**2, f"Co-change queries retained {retained} bytes"
        assert index.co_changed_files(names[-1], limit=len(names)) == tuple((peer, 1) for peer in names[:-1])
    finally:
        tracemalloc.stop()
