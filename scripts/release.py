"""Cut a new agent version: quality gates, version bump, changelog, git tag.

  python scripts/release.py 0.2.0 "Calibrated opponent mix from game 1-3 transactions"

Refuses to release if the tree is dirty, tests fail, or the case 0 evaluation
(replay of the cached extraction) fails its gates or regresses beyond tolerance.
Then push with:  git push && git push --tags
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True)


def main() -> int:
    if len(sys.argv) < 3 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        print(__doc__)
        return 2
    version, note = sys.argv[1], " ".join(sys.argv[2:])
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip():
        print("working tree is dirty; commit first")
        return 1
    run(sys.executable, "-m", "pytest", "-q")
    cached = list((ROOT / "results" / "extractions").glob("case0__*.json"))
    if cached:
        r = run(sys.executable, "-m", "c2f", "eval", "--case-id", "case0", "--replay",
                "--fail-on-regression", check=False)
        if r.returncode != 0:
            print("case 0 quality gate failed; not releasing")
            return 1
    else:
        print("WARNING: no cached case 0 extraction yet, eval gate skipped")

    init = ROOT / "c2f" / "__init__.py"
    init.write_text(re.sub(r'__version__ = "[^"]+"', f'__version__ = "{version}"', init.read_text()))
    log = ROOT / "CHANGELOG.md"
    text = log.read_text()
    entry = f"## v{version} ({time.strftime('%Y-%m-%d %H:%M')})\n\n- {note}\n\n"
    log.write_text(text.replace("<!-- new entries below -->\n\n", f"<!-- new entries below -->\n\n{entry}", 1))
    run("git", "add", "-A", "c2f/__init__.py", "CHANGELOG.md", "results")
    run("git", "commit", "-m", f"Release v{version}: {note}")
    run("git", "tag", "-a", f"v{version}", "-m", note)
    print(f"\nReleased v{version}. Now: git push && git push --tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
