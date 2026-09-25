"""Check Git's publication candidates without reading local credentials or recordings."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


PRIVATE_DIRECTORIES = {".venv", ".runtime", "__pycache__", ".pytest_cache", "node_modules"}
PRIVATE_SUFFIXES = (
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".mp4",
    ".zip", ".tar", ".tar.gz", ".tgz",
    ".sqlite", ".sqlite3", ".sqlite-wal", ".sqlite-shm", ".sqlite3-wal", ".sqlite3-shm",
    ".db", ".db-wal", ".db-shm", ".pfx", ".p12", ".key",
)


def publication_issues(paths: list[str]) -> list[str]:
    issues = []
    for name in paths:
        path = PurePosixPath(name.replace("\\", "/").lower())
        private_env = (path.name == ".env" or path.name.startswith(".env.")) and path.name != ".env.example"
        private_dataset = "data" in path.parts and path.name in {"manifest.jsonl", "examples.jsonl"}
        if (
            private_env
            or private_dataset
            or PRIVATE_DIRECTORIES.intersection(path.parts)
            or path.name.endswith(PRIVATE_SUFFIXES)
        ):
            issues.append(f"Local/private artifact would be published: {name}")
    return issues


def screenshot_issues(root: Path) -> list[str]:
    readme = (root / "README.md").read_text(encoding="utf-8")
    issues = []
    for name in sorted(set(re.findall(r"docs/images/[a-zA-Z0-9_.-]+\.png", readme))):
        image = root / name
        if not image.is_file():
            issues.append(f"Missing README screenshot: {name}")
            continue
        with image.open("rb") as stream:
            header = stream.read(24)
        if (
            len(header) != 24
            or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR"
            or not int.from_bytes(header[16:20], "big")
            or not int.from_bytes(header[20:24], "big")
        ):
            issues.append(f"Invalid PNG screenshot: {name}")
    return issues


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    paths = [name for name in output.decode("utf-8").split("\0") if name]
    issues = publication_issues(paths) + screenshot_issues(root)
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    print(f"Publication checks passed for {len(paths)} candidate files and README screenshot references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
