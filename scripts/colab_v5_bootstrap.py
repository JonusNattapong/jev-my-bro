"""Clone the repository before uploading the reviewed v5 data."""

from pathlib import Path
import subprocess


root = Path("/content/jev-my-bro")
if not root.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/JonusNattapong/jev-my-bro.git", str(root)],
        check=True,
    )
print(root)
