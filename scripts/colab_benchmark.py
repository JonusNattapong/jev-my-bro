from __future__ import annotations

import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.3.4"], check=True)
Path("/content/jevbro").mkdir(exist_ok=True)
shutil.copyfile("/content/schema.py", "/content/jevbro/schema.py")
Path("/content/jevbro/__init__.py").write_text("")
model_root = snapshot_download(
    "JonusNattapong/jev-my-bro",
    local_dir="/content/jev-model",
)
sys.argv = [
    "benchmark_inference.py",
    "--model",
    f"{model_root}/laya-model",
    "--data",
    "/content/test.jsonl",
    "--device",
    "cuda",
    "--batch-sizes",
    "1,4,16",
    "--warmup",
    "2",
    "--repeats",
    "10",
    "--report",
    "/content/benchmark-gpu.json",
]
_ = runpy.run_path("/content/benchmark_inference.py", run_name="__main__")
