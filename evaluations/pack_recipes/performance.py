"""Compare the original anchor collector and its correction on the same real index."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import LineAnchor
from jev_navigator.recipe_configuration import SearchRecipe
from jev_navigator.sources import ANCHORS, FILES, Seeds


def compare(baseline):
    old_source = subprocess.check_output(
        [
            "git",
            "show",
            f"{baseline}:src/jev_navigator/recipe_configuration.py",
        ],
        text=True,
    )
    module = types.ModuleType("jev_navigator.baseline_recipe_configuration")
    module.__package__ = "jev_navigator"
    sys.modules[module.__name__] = module
    exec(compile(old_source, "baseline_recipe_configuration.py", "exec"), module.__dict__)
    with tempfile.TemporaryDirectory(prefix="jvn-recipe-performance-") as folder:
        root = Path(folder)
        (root / "api.py").write_text(
            "\n".join(
                f"def operation_{number}(value):\n    return value + {number}\n" for number in range(256)
            )
        )
        index = CodeIndex.from_directory(root)
        seeds = Seeds(
            files=("api.py",), anchors=tuple(LineAnchor("api.py", number * 3 + 2) for number in range(128))
        )
        measurements = {}
        identities = []
        for name, owner in (("before", module.SearchRecipe), ("after", SearchRecipe)):
            started = time.perf_counter()
            result = owner("local", 1, (ANCHORS, FILES)).gather(index, seeds, box_chars=76_800)
            measurements[name] = time.perf_counter() - started
            identities.append([(unit.id, unit.content_sha256) for unit in result.units])
        assert identities[0] == identities[1], "The correction changed candidate evidence"
        return {
            "baseline": baseline,
            "functions": 256,
            "anchors": 128,
            "identical_evidence": True,
            "seconds": measurements,
            "speedup": measurements["before"] / measurements["after"],
            "provider_calls": 0,
            "spent_usd": 0,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="7e50e42")
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline), indent=2))
