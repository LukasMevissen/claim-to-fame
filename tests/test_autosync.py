"""scripts/autosync.py against real throwaway git repos: a bare 'GitHub', my clone, a teammate's clone."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("autosync", Path(__file__).parents[1] / "scripts" / "autosync.py")
autosync = importlib.util.module_from_spec(_spec)
sys.modules["autosync"] = autosync          # dataclasses look the module up while executing it
_spec.loader.exec_module(autosync)

PASS = [sys.executable, "-c", "pass"]
FAIL = [sys.executable, "-c", "import sys; sys.exit(1)"]


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def clone(remote, dst, name):
    git(dst.parent, "clone", "-q", str(remote), str(dst))
    git(dst, "config", "user.name", name)
    git(dst, "config", "user.email", f"{name}@example.invalid")
    git(dst, "config", "core.autocrlf", "false")
    return dst


def write(repo, path, text):
    p = repo / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def push_change(repo, path, text, msg="teammate change"):
    write(repo, path, text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)
    git(repo, "push", "-q", "origin", "HEAD:main")


@pytest.fixture
def repos(tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q")
    git(seed, "symbolic-ref", "HEAD", "refs/heads/main")
    for k, v in (("user.name", "seed"), ("user.email", "seed@example.invalid"), ("core.autocrlf", "false")):
        git(seed, "config", k, v)
    write(seed, "app.py", "x = 1\n")
    write(seed, ".gitignore", ".env\n")
    write(seed, "results/market/summary.json", "{}\n")
    write(seed, "runs/submissions.jsonl", "")
    git(seed, "add", "-A")
    git(seed, "commit", "-qm", "init")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-q", "origin", "main")
    me = clone(remote, tmp_path / "me", "Lukas")
    other = clone(remote, tmp_path / "other", "Diego")
    return remote, me, other


def make_sync(repo, alerts, **kw):
    kw.setdefault("settle_seconds", 0)
    kw.setdefault("gate_cmd", PASS)
    return autosync.Sync(repo=repo, notify=lambda title, body: alerts.append(f"{title}: {body}"), **kw)


def remote_log(remote):
    return git(remote, "log", "--format=%an|%s", "main")


def test_commits_and_pushes_my_work(repos):
    remote, me, other = repos
    alerts = []
    write(me, "app.py", "x = 2\n")
    write(me, "docs/notes.md", "new file\n")
    st = make_sync(me, alerts).run_once()
    assert st["state"] == "ok" and "pushed 1" in st["detail"]
    top = remote_log(remote).splitlines()[0]
    assert top.startswith("Lukas|Auto-sync (Lukas): app.py, docs/notes.md")
    git(other, "pull", "-q")
    assert (other / "app.py").read_text() == "x = 2\n"
    assert not alerts


def test_pulls_teammates_work(repos):
    remote, me, other = repos
    push_change(other, "app.py", "x = 3\n")
    st = make_sync(me, []).run_once()
    assert st["state"] == "ok" and "pulled 1" in st["detail"]
    assert (me / "app.py").read_text() == "x = 3\n"


def test_my_work_is_rebased_on_top_of_teammates(repos):
    remote, me, other = repos
    push_change(other, "other.py", "y = 1\n")
    write(me, "mine.py", "z = 1\n")
    assert make_sync(me, []).run_once()["state"] == "ok"
    subjects = [l.split("|")[1] for l in remote_log(remote).splitlines()]
    assert subjects[:2] == ["Auto-sync (Lukas): mine.py", "teammate change"]   # linear, mine on top


def test_generated_data_is_never_committed_and_upstream_wins(repos):
    remote, me, other = repos
    write(me, "results/market/summary.json", '{"local": true}\n')
    push_change(other, "results/market/summary.json", '{"server": true}\n', "Auto-sync: market")
    s = make_sync(me, [])
    assert s.run_once()["state"] == "ok"
    assert (me / "results/market/summary.json").read_text() == '{"server": true}\n'
    backups = list((me / ".git" / "autosync-backup").rglob("summary.json"))
    assert backups and backups[0].read_text() == '{"local": true}\n'
    assert "Lukas" not in remote_log(remote)


def test_local_generated_file_survives_when_upstream_did_not_touch_it(repos):
    remote, me, other = repos
    write(me, "runs/submissions.jsonl", '{"local": 1}\n')
    push_change(other, "app.py", "x = 9\n")
    assert make_sync(me, []).run_once()["state"] == "ok"
    assert (me / "app.py").read_text() == "x = 9\n"
    assert (me / "runs/submissions.jsonl").read_text() == '{"local": 1}\n'   # kept, still uncommitted
    assert "Lukas" not in remote_log(remote)


def test_untracked_generated_file_that_upstream_adds_is_replaced(repos):
    remote, me, other = repos
    write(me, "data/transactions/game9.json", "local\n")
    push_change(other, "data/transactions/game9.json", "server\n")
    assert make_sync(me, []).run_once()["state"] == "ok"
    assert (me / "data/transactions/game9.json").read_text() == "server\n"


def test_conflict_aborts_keeps_my_commit_and_alerts_once(repos):
    remote, me, other = repos
    push_change(other, "app.py", "x = 'theirs'\n")
    write(me, "app.py", "x = 'mine'\n")
    alerts = []
    s = make_sync(me, alerts)
    st = s.run_once()
    assert st["state"] == "conflict" and "app.py" in st["detail"]
    assert not (me / ".git" / "rebase-merge").exists()                 # not left half-rebased
    assert git(me, "log", "-1", "--format=%s") == "Auto-sync (Lukas): app.py"   # my work is safe
    assert (me / "app.py").read_text() == "x = 'mine'\n"
    assert "Lukas" not in remote_log(remote)
    assert s.run_once()["state"] == "conflict"
    assert len(alerts) == 1                                            # no alert spam every 2 min


def test_secret_is_never_committed(repos):
    remote, me, other = repos
    write(me, "config_local.py", 'OPENAI = "sk-' + "a" * 40 + '"\n')
    alerts = []
    st = make_sync(me, alerts).run_once()
    assert st["state"] == "blocked" and "config_local.py" in st["detail"]
    assert git(me, "status", "--porcelain").startswith("??")            # still uncommitted
    assert "Lukas" not in remote_log(remote) and alerts


def test_value_from_dotenv_is_never_committed(repos):
    remote, me, other = repos
    write(me, ".env", "TEAM_API_KEY=team-key-1234567890\n")             # ignored itself
    write(me, "notes.md", "debug: team-key-1234567890\n")
    st = make_sync(me, []).run_once()
    assert st["state"] == "blocked" and "notes.md" in st["detail"]


def test_waits_while_files_are_still_being_edited(repos):
    remote, me, other = repos
    write(me, "app.py", "x = 5\n")
    st = make_sync(me, [], settle_seconds=60).run_once()
    assert st["state"] == "settling"
    assert git(me, "status", "--porcelain") == "M app.py"


def test_failing_tests_block_the_push_not_the_commit(repos):
    remote, me, other = repos
    write(me, "app.py", "x = 6\n")
    alerts = []
    st = make_sync(me, alerts, gate_cmd=FAIL).run_once()
    assert st["state"] == "tests-failing"
    assert git(me, "log", "-1", "--format=%s") == "Auto-sync (Lukas): app.py"
    assert "Lukas" not in remote_log(remote) and alerts


def test_does_nothing_off_main(repos):
    remote, me, other = repos
    git(me, "switch", "-q", "-c", "experiment")
    write(me, "app.py", "x = 7\n")
    assert make_sync(me, []).run_once()["state"] == "off-main"
    assert git(me, "status", "--porcelain") == "M app.py"


def test_pause_and_resume(repos):
    remote, me, other = repos
    s = make_sync(me, [])
    (me / ".git" / "autosync.paused").touch()
    write(me, "app.py", "x = 8\n")
    assert s.run_once()["state"] == "paused"
    (me / ".git" / "autosync.paused").unlink()
    assert s.run_once()["state"] == "ok"


def test_offline_is_quiet(repos, tmp_path):
    remote, me, other = repos
    git(me, "remote", "set-url", "origin", str(tmp_path / "nowhere.git"))
    alerts = []
    assert make_sync(me, alerts).run_once()["state"] == "offline"
    assert not alerts
