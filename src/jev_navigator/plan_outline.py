"""Compact source facts for a planner, streamed one file at a time.

This is a map of actual declarations and imports, without source bodies or claims
about what a directory means. The host chooses files and measures its tokenizer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .index.code_index import CodeIndex
from .index.languages import language_read


def outline_lines(index: CodeIndex, files: Iterable[str] | None = None) -> Iterator[str]:
    """Yield a path, declarations with extents, and resolved imports for each file.

    Unknown files are refused. Text files carry their real format blocks. Parsing is
    lazy and per file. No repository-wide parser response or source body is assembled.
    """
    for file in dict.fromkeys(index.files if files is None else files):
        if file not in index.files:
            raise ValueError(f"{file!r} is not a file in the index")
        yield f"{file}\n"
        if language_read(file):
            for span in index.definitions_in(file):
                yield f"  {span.name}:{span.start}-{span.end}\n"
            for imported in index.imports(file):
                yield f"  ->{imported}\n"
        elif file not in index.text_files_left_out((file,)):
            for block in index.text_blocks_in(file):
                yield f"  {block.name or '<text>'}:{block.start}-{block.end}\n"
