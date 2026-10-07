import asyncio
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import replay


def main():
    out = Path.home() / ".local/share/jvn-takeover/2026-10-03/search-design/recipes"
    records = list(replay.rows(out / "dev110-candidates.jsonl"))[-3:]
    inputs = replay.load(replay.BASE / "runs/pack-49b78955/pack-inputs-dev110.json")["cases"]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        old = root / "old_replay.py"
        old.write_text(
            subprocess.check_output(["git", "show", "c694249:evaluations/pack_recipes/replay.py"], text=True)
        )
        spec = importlib.util.spec_from_file_location("old_replay", old)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        before = root / "before"
        after = root / "after"
        before.mkdir()
        after.mkdir()
        old_results = asyncio.run(baseline.native_replay(records, {}, {}, inputs, before))
        fixed_results = asyncio.run(replay.native_replay(records, {}, {}, inputs, after))
        assert len(old_results) == len(fixed_results) == 3
        assert all(r["failure"] == "no readable anchor" for r in fixed_results)
        assert not (before / "native-replay.json").exists()
        assert len(replay.load(after / "native-replay.json")) == 3
        (out / "native-tail-regression.json").write_text(
            json.dumps(
                {
                    "baseline": "c694249",
                    "real_engine_case": records[0]["case"],
                    "returned_before": len(old_results),
                    "saved_before": 0,
                    "saved_after": len(fixed_results),
                    "provider_calls": 0,
                },
                indent=2,
            )
            + "\n"
        )
        print("Real Engine no-anchor terminal checkpoint regression passes; missing-anchor records retained.")


if __name__ == "__main__":
    main()
