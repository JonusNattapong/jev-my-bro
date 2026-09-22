"""Prepare the remote v5 data directory before file uploads."""

from pathlib import Path


Path("/content/jev-my-bro/data/reviewed_v5").mkdir(parents=True, exist_ok=True)
print("prepared")
