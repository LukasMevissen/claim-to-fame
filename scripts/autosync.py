"""Keep this checkout on the latest main: commit your work, pull everyone else's, push.

    python scripts/autosync.py              # one sync cycle (what the scheduled task runs)
    python scripts/autosync.py install      # Windows: run every 2 min in the background
    python scripts/autosync.py uninstall
    python scripts/autosync.py loop         # macOS/Linux: keep it running in a terminal
    python scripts/autosync.py status       # last result and recent log
    python scripts/autosync.py pause | resume

One cycle (only while you are on main):
1. Commit your saved changes as "Auto-sync (<your name>): <files>", once no changed file
   was touched for 30 s, so a burst of saves lands in one commit.
2. Fetch and rebase onto origin/main.
3. Run the tests; push only if they pass (main is what the bot is deployed from).

Never committed from here:
- Generated data the server pushes after every game (GENERATED below). Upstream always
  wins; a local copy is backed up to .git/autosync-backup/ before it is replaced, and kept
  as is when upstream did not touch that file.
- Anything that looks like a secret (.env, API keys, the values in your .env).

It stops and tells you (log, status, Windows notification, optional C2F_ALERT_WEBHOOK) on a
merge conflict, failing tests or a suspected secret. Your commits are never lost: a
conflicting rebase is aborted and your commit stays local until you resolve it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
TASK_NAME = "c2f-autosync"

# Written by the tournament server and pushed after every game; never committed from here.
GENERATED = (
    "data/transactions/",
    "results/market/",
    "results/extractions/",
    "results/opponent_belief.json",
    "results/ab_table.csv",
    "runs/submissions.jsonl",
)
SECRET_NAME = re.compile(r"(^|/)(\.env(\..+)?|[^/]*\.pem|[^/]*\.key|id_rsa[^/]*|credentials[^/]*\.json)$", re.I)
SECRET_TEXT = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}")
MAX_FILE_MB = 25
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0   # no console flash under pythonw


def is_generated(path: str) -> bool:
    return any(path == g or (g.endswith("/") and path.startswith(g)) for g in GENERATED)


def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class GitError(RuntimeError):
    pass


class FetchError(GitError):
    pass


@dataclass
class Sync:
    repo: Path = ROOT
    branch: str = "main"
    remote: str = "origin"
    settle_seconds: float = 30
    gate_cmd: list[str] | None = None            # None: pytest with the repo's venv
    notify: Callable[[str, str], None] | None = None
    push_retries: int = 3

    # ---------- git plumbing ----------

    def git(self, *args: str, check: bool = True, input: str | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LITERAL_PATHSPECS="1", GIT_EDITOR="true")
        r = subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", input=input, env=env, creationflags=NO_WINDOW, timeout=180)
        if check and r.returncode != 0:
            raise GitError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[:500]}")
        return r

    def out(self, *args: str) -> str:
        return self.git(*args).stdout.strip()

    @property
    def gitdir(self) -> Path:
        return Path(self.out("rev-parse", "--absolute-git-dir"))

    @property
    def upstream(self) -> str:
        return f"{self.remote}/{self.branch}"

    def changes(self) -> list[tuple[str, str]]:
        """(XY status, path) for every changed or untracked (not ignored) path."""
        raw = self.git("status", "--porcelain=v1", "-z", "--untracked-files=all").stdout.split("\0")
        res, i = [], 0
        while i < len(raw):
            entry = raw[i]
            i += 1
            if not entry:
                continue
            xy, path = entry[:2], entry[3:]
            if xy[0] in "RC":
                i += 1                      # skip the rename source
            res.append((xy, path))
        return res

    def in_progress(self) -> str | None:
        g = self.gitdir
        for name in ("rebase-merge", "rebase-apply", "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD"):
            if (g / name).exists():
                return name
        return None

    def count(self, rng: str) -> int:
        return int(self.out("rev-list", "--count", rng))

    def fetch(self) -> None:
        r = self.git("fetch", "--quiet", self.remote, self.branch, check=False)
        if r.returncode != 0:
            raise FetchError((r.stderr or r.stdout).strip()[:300])

    def age(self, path: str) -> float:
        p = self.repo / path
        return time.time() - p.stat().st_mtime if p.exists() else float("inf")

    # ---------- safety checks ----------

    def suspected_secrets(self, paths: list[str]) -> list[str]:
        values = {v for k, v in read_env(self.repo / ".env").items()
                  if any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(v) >= 12}
        hits = []
        for path in paths:
            p = self.repo / path
            if SECRET_NAME.search(path):
                hits.append(f"{path} (secret-looking file name)")
                continue
            if not p.is_file():
                continue
            size = p.stat().st_size
            if size > MAX_FILE_MB * 1024 * 1024:
                hits.append(f"{path} ({size / 2**20:.0f} MB, too large to auto-commit)")
                continue
            if size > 5 * 1024 * 1024:
                continue
            data = p.read_bytes()
            if b"\0" in data[:8000]:
                continue                     # binary
            text = data.decode("utf-8", errors="replace")
            if SECRET_TEXT.search(text) or any(v in text for v in values):
                hits.append(f"{path} (contains what looks like an API key)")
        return hits

    def gate(self) -> tuple[bool, str]:
        cmd = self.gate_cmd
        if cmd is None:
            if not (self.repo / "tests").is_dir():
                return True, "no tests"
            venv = self.repo / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            # The bot's tests; autosync's own tests are slow (real git repos) and CI runs them.
            cmd = [str(venv if venv.exists() else sys.executable), "-m", "pytest", "-q", "-x",
                   "-p", "no:cacheprovider", "--ignore=tests/test_autosync.py"]
        try:
            r = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", creationflags=NO_WINDOW, timeout=600)
        except subprocess.TimeoutExpired:
            return False, "tests timed out after 10 min"
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-8:])
        return r.returncode == 0, tail

    # ---------- steps ----------

    def commit_mine(self, paths: list[str]) -> bool:
        self.git("add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul", input="\0".join(paths))
        staged = [p for p in self.git("diff", "--cached", "--name-only", "-z").stdout.split("\0") if p]
        gen = [p for p in staged if is_generated(p)]
        if gen:                               # staged by hand: the server owns these
            self.git("restore", "--staged", "--pathspec-from-file=-", "--pathspec-file-nul", input="\0".join(gen))
        staged = [p for p in staged if not is_generated(p)]
        if not staged:
            return False
        name = self.git("config", "user.name", check=False).stdout.strip() or "unknown"
        shown = ", ".join(sorted(staged)[:3]) + (f" +{len(staged) - 3} more" if len(staged) > 3 else "")
        self.git("commit", "-q", "-m", f"Auto-sync ({name}): {shown}")
        return True

    def backup(self, path: str, stamp: str) -> Path | None:
        src = self.repo / path
        if not src.exists():
            return None
        dst = self.gitdir / "autosync-backup" / stamp / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def rebase_onto_upstream(self, generated: list[tuple[str, str]]) -> tuple[bool, str]:
        """Rebase local commits onto upstream. Dirty generated files are moved out of the way:
        replaced by upstream when upstream changed them (local copy backed up), otherwise put
        back after the rebase. Returns (ok, problem description)."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        incoming = set(p for p in self.git("diff", "--name-only", "-z", "HEAD", self.upstream).stdout.split("\0") if p)
        restore: list[tuple[str, Path | None]] = []
        for xy, path in generated:
            saved = self.backup(path, stamp)
            if xy == "??":
                if path in incoming:          # upstream adds the same file: upstream wins
                    (self.repo / path).unlink(missing_ok=True)
                continue
            if xy[0] == "A":                  # staged new file, not in HEAD
                self.git("rm", "-q", "--cached", "--", path)
                if path in incoming:
                    (self.repo / path).unlink(missing_ok=True)
                continue
            self.git("restore", "--staged", "--worktree", "--source=HEAD", "--", path)
            if path not in incoming:
                restore.append((path, saved))  # nobody else changed it: keep the local version
        r = self.git("rebase", "--quiet", self.upstream, check=False)
        problem = ""
        if r.returncode != 0:
            conflicts = [p for p in self.git("diff", "--name-only", "--diff-filter=U", "-z", check=False)
                         .stdout.split("\0") if p]
            if conflicts:
                problem = (f"your commits conflict with {self.upstream} in {', '.join(conflicts)}. "
                           "Resolve by hand: git pull --rebase")
            else:
                problem = f"git rebase failed: {(r.stderr or r.stdout).strip()[:300]}"
            if self.in_progress():
                self.git("rebase", "--abort", check=False)
        for path, saved in restore:
            dst = self.repo / path
            if saved is None:
                dst.unlink(missing_ok=True)   # was deleted locally
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, dst)
        return r.returncode == 0, problem

    # ---------- one cycle ----------

    def cycle(self) -> dict:
        if sys.platform == "win32" and self.git("config", "core.longpaths", check=False).stdout.strip() != "true":
            # Deep checkouts exceed MAX_PATH; without this `git rebase` dies with "Filename too long".
            self.git("config", "core.longpaths", "true")
        if (self.gitdir / "autosync.paused").exists():
            return self.done("paused", "paused; run: python scripts/autosync.py resume")
        busy = self.in_progress()
        if busy:
            return self.done("waiting", f"a git operation is in progress ({busy}); finish it, then sync resumes")
        branch = self.git("symbolic-ref", "--short", "-q", "HEAD", check=False).stdout.strip()
        if branch != self.branch:
            return self.done("off-main", f"on '{branch or 'detached HEAD'}', not {self.branch}: not syncing")

        changes = self.changes()
        mine = [p for xy, p in changes if not is_generated(p)]
        generated = [(xy, p) for xy, p in changes if is_generated(p)]
        if mine:
            young = [p for p in mine if self.age(p) < self.settle_seconds]
            if young:
                return self.done("settling", f"waiting until edits settle ({young[0]}"
                                 + (f" +{len(young) - 1}" if len(young) > 1 else "") + ")")
            secrets = self.suspected_secrets(mine)
            if secrets:
                return self.done("blocked", "not committing, possible secret: " + "; ".join(secrets)
                                 + ". Remove it or add the file to .gitignore.", alert=True)
            self.commit_mine(mine)

        self.fetch()
        behind = self.count(f"HEAD..{self.upstream}")
        if behind:
            ok, problem = self.rebase_onto_upstream(generated)
            if not ok:
                return self.done("conflict" if "conflict" in problem else "rebase-failed", problem, alert=True)
        for attempt in range(self.push_retries):
            ahead = self.count(f"{self.upstream}..HEAD")
            if not ahead:
                return self.done("ok", f"up to date with {self.upstream}" + (f", pulled {behind} commit(s)" if behind else ""))
            ok, tail = self.gate()
            if not ok:
                return self.done("tests-failing", f"{ahead} commit(s) kept local, not pushed: tests fail.\n{tail}", alert=True)
            r = self.git("push", "--quiet", self.remote, f"HEAD:{self.branch}", check=False)
            if r.returncode == 0:
                return self.done("ok", f"pushed {ahead} commit(s)" + (f", pulled {behind}" if behind else ""))
            # Someone pushed in between: take their commits and try again.
            self.fetch()
            ok, problem = self.rebase_onto_upstream([(xy, p) for xy, p in self.changes() if is_generated(p)])
            if not ok:
                return self.done("conflict" if "conflict" in problem else "rebase-failed", problem, alert=True)
        return self.done("push-failed", (r.stderr or r.stdout).strip()[:300], alert=True)

    # ---------- reporting ----------

    def done(self, state: str, detail: str, alert: bool = False) -> dict:
        path = self.gitdir / "autosync-status.json"
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
        head = self.git("rev-parse", "--short", "HEAD", check=False).stdout.strip()
        st = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "state": state, "detail": detail, "head": head,
              "alert_key": f"{state}:{head}:{detail.splitlines()[0][:120]}" if alert else None}
        path.write_text(json.dumps(st, indent=2), encoding="utf-8")
        self.log(f"{state}: {detail.splitlines()[0]}")
        if alert and st["alert_key"] != prev.get("alert_key"):     # once per new problem
            (self.notify or notify)(f"Auto-sync: {state}", detail.splitlines()[0])
        return st

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True) if sys.stdout else None
        p = self.gitdir / "autosync.log"
        if p.exists() and p.stat().st_size > 1_000_000:
            p.replace(p.with_suffix(".log.1"))
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run_once(self) -> dict:
        """One cycle with a lock, so overlapping scheduled runs never collide."""
        lock = self.gitdir / "autosync.lock"
        if lock.exists() and time.time() - lock.stat().st_mtime > 900:
            lock.unlink(missing_ok=True)          # stale lock from a killed run
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return {"state": "locked", "detail": "another sync is running"}
        os.close(fd)
        try:
            return self.cycle()
        except FetchError as e:                    # offline or GitHub hiccup: try again next cycle
            return self.done("offline", f"could not reach {self.remote}: {e}")
        except Exception as e:  # noqa: BLE001
            return self.done("error", f"{type(e).__name__}: {e}", alert=True)
        finally:
            lock.unlink(missing_ok=True)


def notify(title: str, body: str) -> None:
    """Windows notification + optional webhook. Best effort, never raises."""
    url = os.environ.get("C2F_ALERT_WEBHOOK") or read_env(ROOT / ".env").get("C2F_ALERT_WEBHOOK")
    if url:
        try:
            import urllib.request
            req = urllib.request.Request(url, data=json.dumps({"content": f"{title}: {body}", "text": f"{title}: {body}"}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception:  # noqa: BLE001
            pass
    if sys.platform == "win32":
        ps = ("[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime]|Out-Null;"
              "$x=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
              "$t=$x.GetElementsByTagName('text');$t.Item(0).AppendChild($x.CreateTextNode($env:C2F_T1))|Out-Null;"
              "$t.Item(1).AppendChild($x.CreateTextNode($env:C2F_T2))|Out-Null;"
              "$id='{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe';"
              "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($id).Show([Windows.UI.Notifications.ToastNotification]::new($x))")
        try:
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=20,
                           env=dict(os.environ, C2F_T1=title, C2F_T2=body[:250]), capture_output=True, creationflags=NO_WINDOW)
        except Exception:  # noqa: BLE001
            pass


# ---------- scheduling ----------

def _pythonw() -> str:
    venv = ROOT / ".venv" / "Scripts"
    for cand in (venv / "pythonw.exe", Path(sys.executable).with_name("pythonw.exe")):
        if cand.exists():
            return str(cand)
    return sys.executable


def _powershell(script: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                          text=True, env=dict(os.environ, **env), creationflags=NO_WINDOW)


def install(every_minutes: int) -> int:
    if sys.platform != "win32":
        print("Scheduled task is Windows only. Elsewhere run:  python scripts/autosync.py loop")
        return 2
    ps = ("$a=New-ScheduledTaskAction -Execute $env:C2F_PY -Argument ('\"'+$env:C2F_SCRIPT+'\" once') -WorkingDirectory $env:C2F_REPO;"
          "$t=New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(20) -RepetitionInterval (New-TimeSpan -Minutes ([int]$env:C2F_EVERY));"
          "$s=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) "
          "-StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries;"
          "Register-ScheduledTask -TaskName $env:C2F_TASK -Action $a -Trigger $t -Settings $s -Force "
          "-Description 'claim-to-fame: commit, pull and push main (scripts/autosync.py)' | Out-Null")
    r = _powershell(ps, C2F_PY=_pythonw(), C2F_SCRIPT=str(Path(__file__).resolve()), C2F_REPO=str(ROOT),
                    C2F_EVERY=str(every_minutes), C2F_TASK=TASK_NAME)
    if r.returncode != 0:
        print(r.stderr.strip())
        return 1
    print(f"Installed scheduled task '{TASK_NAME}': syncs every {every_minutes} min while you are logged in.")
    print("Check it any time:  python scripts/autosync.py status")
    return 0


def uninstall() -> int:
    r = _powershell("Unregister-ScheduledTask -TaskName $env:C2F_TASK -Confirm:$false", C2F_TASK=TASK_NAME)
    print("Removed." if r.returncode == 0 else r.stderr.strip())
    return r.returncode


def status(s: Sync) -> int:
    p = s.gitdir / "autosync-status.json"
    print(p.read_text(encoding="utf-8") if p.exists() else "no sync has run yet")
    log = s.gitdir / "autosync.log"
    if log.exists():
        print("\nrecent log:")
        print("\n".join(log.read_text(encoding="utf-8").splitlines()[-12:]))
    if sys.platform == "win32":
        r = _powershell("(Get-ScheduledTask -TaskName $env:C2F_TASK -ErrorAction Stop).State", C2F_TASK=TASK_NAME)
        print(f"\nscheduled task: {r.stdout.strip() if r.returncode == 0 else 'not installed'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", default="once",
                    choices=["once", "loop", "install", "uninstall", "status", "pause", "resume"])
    ap.add_argument("--every", type=float, default=2, help="minutes between syncs (loop, install)")
    args = ap.parse_args(argv)
    s = Sync()
    if args.cmd == "once":
        st = s.run_once()
        return 0 if st.get("state") in ("ok", "settling", "locked", "offline", "paused", "off-main") else 1
    if args.cmd == "loop":
        while True:
            s.run_once()
            time.sleep(args.every * 60)
    if args.cmd == "install":
        return install(max(1, int(args.every)))
    if args.cmd == "uninstall":
        return uninstall()
    if args.cmd == "status":
        return status(s)
    flag = s.gitdir / "autosync.paused"
    if args.cmd == "pause":
        flag.touch()
        print("Paused. Resume with: python scripts/autosync.py resume")
    else:
        flag.unlink(missing_ok=True)
        print("Resumed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
