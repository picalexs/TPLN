"""Run all report figure scripts in this folder.

Each ``fig_*.py`` is invoked as a subprocess so failures are isolated.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    scripts = sorted(p for p in HERE.glob("fig_*.py") if p.is_file())
    failures: list[tuple[str, str]] = []
    started = time.time()
    for s in scripts:
        t0 = time.time()
        print(f"[run] {s.name} ...", flush=True)
        proc = subprocess.run([sys.executable, str(s)], capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            failures.append((s.name, proc.stderr.strip() or proc.stdout.strip()))
            print(f"  FAILED in {dt:.1f}s")
        else:
            print(f"  ok ({dt:.1f}s)")
    total = time.time() - started
    print(f"\nFinished in {total:.1f}s. {len(scripts) - len(failures)} succeeded, "
          f"{len(failures)} failed.")
    for name, err in failures:
        print(f"\n--- {name} ---\n{err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
