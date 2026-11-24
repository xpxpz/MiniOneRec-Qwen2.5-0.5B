"""Log commands, elapsed time and sampled single-GPU memory per stage."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def gpu_sample():
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=name,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        name, memory = result.stdout.strip().rsplit(",", 1)
        return name, int(memory)
    return None


def main(stage, log_dir, command):
    uses_gpu = stage not in {"prepare", "convert"}
    if uses_gpu and not shutil.which("nvidia-smi"):
        raise RuntimeError("NVIDIA driver and nvidia-smi required for formal GPU stages")
    sample = gpu_sample() if uses_gpu else None
    if uses_gpu and sample is None:
        raise RuntimeError("Cannot query GPU 0")
    peak = sample[1] if sample else None
    start = time.monotonic()
    with subprocess.Popen(command) as process:
        while process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                current = gpu_sample() if sample else None
                if current:
                    peak = max(peak, current[1])
    elapsed = time.monotonic() - start
    entry = {
        "stage": stage, "command": command, "exit_code": process.returncode,
        "wall_seconds": elapsed, "gpu_name": sample[0] if sample else None,
        "gpu_count": int(uses_gpu), "gpu_hours": elapsed / 3600 if uses_gpu else 0,
        "peak_sampled_device_mib": peak, "cloud_cost": None,
    }
    with (Path(log_dir) / "stages.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    return process.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[4:]))
