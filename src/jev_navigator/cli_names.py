"""``jvn names``: every real spelling of a word in scope, rarest first, with no model call."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .index.code_index import CodeIndex
from .index.spellings import Spelling, lookup_keys

DEFAULT_MAX_FILES = 5


def add_names_command(commands: argparse._SubParsersAction, max_files_type: type) -> None:
    names = commands.add_parser(
        "names",
        help="list every real spelling of a word in scope, rarest first (no model calls)",
        description="Look a word up in the spelling map: identifiers, config keys, string and comment "
        "words, and file names that share its word parts, whatever their case, separators or plural.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="jvn names website\njvn names web_site --prefix src/ --max-files 2\n"
        'JSON: {"command":"names","target":"website"}\n'
        "createWebsite, WEBSITE_ID, websites and website.ts all spell website.",
    )
    names.add_argument("target", help="The word or name whose spellings to list")
    names.add_argument("--repo", default=".", help="Source directory (default: current directory)")
    names.add_argument("--prefix", action="append", default=[], help="Source scope; repeatable")
    names.add_argument(
        "--max-files",
        type=max_files_type,
        default=DEFAULT_MAX_FILES,
        metavar="N|none",
        help=f"Files listed per spelling (default: {DEFAULT_MAX_FILES}; 'none' lists every file)",
    )


def run_names(args: argparse.Namespace) -> int:
    try:
        index = CodeIndex.from_directory(Path(args.repo).resolve(), prefixes=tuple(args.prefix))
        spellings = index.names(args.target, max_files=args.max_files)
    except KeyboardInterrupt:
        print("jvn names: cancelled", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - the command reports any failure and exits 1
        print(f"jvn names: {error}", file=sys.stderr)
        return 1
    keys = sorted(lookup_keys(args.target))
    if args.json:
        print(json.dumps({"target": args.target, "keys": keys, "spellings": [asdict(s) for s in spellings]}))
    else:
        print("\n".join(_lines(args.target, keys, spellings)))
    return 0


def _lines(target: str, keys: list[str], spellings: tuple[Spelling, ...]) -> list[str]:
    heading = (
        f'{len(spellings)} spellings of "{target}" (keys: {", ".join(keys)}), rarest first; 0 model calls'
    )
    return [heading, *(_line(spelling) for spelling in spellings)]


def _line(spelling: Spelling) -> str:
    kind = "file name, " if spelling.file_name else ""
    places = ", ".join(
        f"{place.file}:{','.join(map(str, place.lines))}" if place.lines else place.file
        for place in spelling.places
    )
    more = f" (+{spelling.capped} more files)" if spelling.capped else ""
    files = "1 file" if spelling.files == 1 else f"{spelling.files} files"
    return f"{spelling.word}  [{kind}{files}]  {places}{more}"
