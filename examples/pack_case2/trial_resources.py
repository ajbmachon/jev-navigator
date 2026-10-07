"""The resource floor shared by trial phases, independent of any provider or planner."""

import re
import shutil
import subprocess
from pathlib import Path


def resources():
    disk = shutil.disk_usage(Path.home()).free
    vm = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+)", vm)[1])
    counts = {name: int(n) for name, n in re.findall(r"(Pages [^:]+):\s+(\d+)", vm)}
    memory = sum(counts.get(f"Pages {name}", 0) for name in ("free", "inactive", "speculative")) * page
    if disk < 30 * 10**9 or memory < 8 * 10**9:
        raise RuntimeError(f"Resource stop: disk={disk}, available memory={memory}")
