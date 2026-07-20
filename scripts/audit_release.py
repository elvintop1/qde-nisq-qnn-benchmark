from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRECTORIES = {"outputs", "results", "artifacts", "checkpoints", "data"}
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".qpy",
    ".log",
}
MAX_SOURCE_SIZE = 5 * 1024 * 1024


def tracked_paths() -> list[Path]:
    process = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if process.returncode == 0 and process.stdout:
        return [ROOT / item.decode() for item in process.stdout.split(b"\0") if item]
    return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]


def main() -> int:
    violations: list[str] = []
    for path in tracked_paths():
        relative = path.relative_to(ROOT)
        if FORBIDDEN_DIRECTORIES.intersection(relative.parts):
            violations.append(f"generated-data directory: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"generated-data suffix: {relative}")
        if path.stat().st_size > MAX_SOURCE_SIZE:
            violations.append(f"file exceeds 5 MiB: {relative}")
    if violations:
        print("Release audit failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Release audit passed: no result artifacts or oversized files are tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
