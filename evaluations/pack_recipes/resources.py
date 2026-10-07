"""Resource admission for the parallel offline development measurements."""

import shutil
import subprocess
from pathlib import Path


def check_resources():
    disk = shutil.disk_usage(Path.cwd()).free
    memory = subprocess.check_output(["memory_pressure"], text=True)
    percent = int(memory.rsplit("System-wide memory free percentage:", 1)[1].split("%", 1)[0])
    size = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    available = size * percent / 100
    if disk < 30_000_000_000 or available < 8_000_000_000:
        raise RuntimeError(f"Resource stop: free disk {disk}, available memory {available}")
    return {"free_disk_bytes": disk, "memory_pressure_available_bytes": available}
