"""Team dashboard: scores, opponent strategies, a combined ToDo list and the program diagram.

    python scripts/dashboard.py                  # build dashboard/data.json once
    python scripts/dashboard.py serve            # http://localhost:8777, rebuilt every 2 min
    python scripts/dashboard.py serve --port 9000 --interval 60 --no-fetch

The page (dashboard/index.html) re-reads data.json every 30 s, so with `serve` running it
stays current while games are played and teammates push.

Sources, all already in the repo or public:
    scores      leaderboard api/matrix (net per team and game); offline: the same numbers
                recomputed from data/transactions/ (received as handyman - paid as insurer)
    strategies  results/ab_table.csv (every team's a and b per line vs the revealed t),
                data/transactions/ (accept / reject decisions), results/opponent_belief.json
    todo        TODO.md (one section per contributor, edited by hand, shared by autosync)
                + tasks derived from the pipeline state (game reviewed? belief refit? ...)
                + each contributor's recent commits (what they are working on right now)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics as st
import subprocess

import threading
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dashboard"
API = "https://c2f.public.quantco.cloud/leaderboard/api"
US = "GemischtesHack"
TEAM_MEMBERS = ("LukasMevissen", "Diegocentrico2001", "milanova-n")
# git author names that belong to the same person
ALIASES = {"Lukas Mevissen": "LukasMevissen"}
RECENT_GAMES = 5


# ------------------------------------------------------------------ scores

def scores_from_transactions(tx_dir: Path) -> dict:
    """Net payoff per team and game: received as handyman minus paid as insurer."""
    games: dict[int, Counter] = {}
    for f in tx_dir.glob("game*.json"):
        doc = json.loads(f.read_text())
        net: Counter = Counter()
        for r in doc.get("items", []):
            net[r["issuer"]] += r["amount"]
            net[r["reviewer"]] -= r["amount"]
        games[int(doc["game_id"])] = net
    ids = sorted(games)
    teams = sorted({t for g in games.values() for t in g})
    return {"games": ids, "source": "data/transactions",
            "teams": {t: [round(games[g].get(t, 0.0), 2) for g in ids] for t in teams}}


def scores_from_api(timeout: float = 10) -> dict:
    url = f"{API}/matrix?page=1&game_limit=1000"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        doc = json.load(r)
    return {"games": doc["game_ids"], "source": "leaderboard api/matrix",
            "teams": {it["team_name"]: [round(c, 2) for c in it["cells"]] for it in doc["items"]}}


def scores(tx_dir: Path, fetch: bool) -> dict:
    if fetch:
        try:
            return scores_from_api()
        except Exception as e:  # offline, API down: same numbers from disk
            out = scores_from_transactions(tx_dir)
            out["warning"] = f"leaderboard unreachable ({type(e).__name__}), using data/transactions"
            return out
    return scores_from_transactions(tx_dir)


# ------------------------------------------------------------------ strategies

def _f(x: str) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_ab(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# Labels are absolute on the price level (a vs the revealed t, on covered lines) and
# relative to the field on acceptance, because every team accepts 30-50% of bills and
# what matters is who is stricter or looser than the rest.

def craftsman_label(m: dict) -> tuple[str, str]:
    """(label, one-line reason) for the handyman role."""
    if m["absent_rate"] >= 0.3:
        return "Often absent", f"no quotes on {m['absent_rate']:.0%} of lines"
    over, mk = m["over_covered"], m["markup"]
    if over is None or mk is None:
        return "Too little data", "fewer than a handful of judged charges"
    if mk >= 1.0 and over >= 0.5:
        return "Overcharger", f"median {mk:.2f}x t, {over:.0%} of covered charges above t"
    if mk < 0.75:
        return "Undercutter", f"median {mk:.2f}x t, safe but leaves money on the table"
    if over <= 0.3:
        return "Fair pricer", f"median {mk:.2f}x t, only {over:.0%} above t"
    return "Threshold hunter", f"median {mk:.2f}x t, {over:.0%} above t: prices right at the edge"


def insurer_label(m: dict, field: dict) -> tuple[str, str]:
    """(label, reason) for the insurer role, relative to the field medians in `field`."""
    acc, wrong, pen = m["accept_rate"], m["wrong_accept_share"], m["penalty_rate"]
    d = acc - field["accept_rate"]
    lab = "Strict gatekeeper" if d <= -0.05 else "Lenient payer" if d >= 0.05 else "Selective"
    tags = []
    if wrong is not None and wrong >= field["wrong_accept_share"] + 0.04:
        tags.append(f"pays many inflated bills ({wrong:.0%} of accepts)")
    if pen >= field["penalty_rate"] + 0.03:
        tags.append(f"rejects many fair bills ({pen:.0%} penalties)")
    return lab, "; ".join(tags) or f"accepts {acc:.0%} of bills (field {field['accept_rate']:.0%})"


def _field(ms: list[dict]) -> dict:
    def med(k):
        xs = [m[k] for m in ms if m[k] is not None and m["absent_rate"] < 0.3]
        return st.median(xs) if xs else 0.0
    return {k: med(k) for k in ("accept_rate", "wrong_accept_share", "penalty_rate")}


def strategies(ab_rows: list[dict], tx_dir: Path, belief_path: Path) -> dict:
    """Per opponent: what they do as handyman (a) and as insurer (b), whole tournament
    and the last RECENT_GAMES games, so a change of strategy shows up."""
    a_vs_t = {}            # (team, game, item) -> fair / fraud / unknown
    by_team: dict[str, list[dict]] = defaultdict(list)
    for r in ab_rows:
        g = int(r["round"])
        a_vs_t[(r["team"], g, int(r["line_item"]))] = r["a_vs_t"]
        by_team[r["team"]].append(r | {"g": g})

    tx: list[dict] = []
    for doc in tx_docs(tx_dir):
        tx += [r | {"g": int(doc["game_id"])} for r in doc.get("items", [])]
    last_game = max((r["g"] for r in tx), default=0)

    belief = {}
    if belief_path.exists():
        belief = json.loads(belief_path.read_text()).get("teams", {})

    def metrics(team: str, since: int) -> dict:
        rows = [r for r in by_team.get(team, []) if r["g"] >= since]
        present = [r for r in rows if r["absent"] != "True"]
        covered = [r for r in present if (_f(r["t_low"]) or 0) > 0]
        judged = [r for r in covered if r["a_vs_t"] in ("fair", "fraud")]
        ratios = []
        for r in covered:
            a, lo, hi = _f(r["a"]), _f(r["t_low"]), _f(r["t_high"])
            if a:
                ratios.append(a / ((lo + hi) / 2 if hi else lo))
        issued = [r for r in tx if r["issuer"] == team and r["g"] >= since]
        reviewed = [r for r in tx if r["reviewer"] == team and r["g"] >= since]
        accepted = [r for r in reviewed if r["accepted"]]
        wrong = [r for r in accepted
                 if a_vs_t.get((r["issuer"], r["g"], r["line_item_index"])) == "fraud"]
        known = [r for r in accepted
                 if a_vs_t.get((r["issuer"], r["g"], r["line_item_index"])) in ("fair", "fraud")]
        penalties = [r for r in reviewed if not r["accepted"] and r["amount"] > 0]
        # insurer cost by state: what share of everything the team paid went where
        cost = defaultdict(float)
        for r in reviewed:
            if not r["amount"]:
                continue
            if not r["accepted"]:
                cost["penalty"] += r["amount"]           # fair bill rejected: pays 1.5a
            else:
                v = a_vs_t.get((r["issuer"], r["g"], r["line_item_index"]))
                cost["fair" if v == "fair" else "inflated" if v == "fraud" else "unclassified"] += r["amount"]
        total_cost = sum(cost.values())
        share = lambda k: cost[k] / total_cost if total_cost else None  # noqa: E731
        n = lambda xs: max(len(xs), 1)  # noqa: E731
        return {
            "lines": len(rows),
            "absent_rate": 1 - len(present) / n(rows) if rows else 0.0,
            "over_covered": (sum(r["a_vs_t"] == "fraud" for r in judged) / len(judged)
                             if len(judged) >= 5 else None),
            "markup": round(st.median(ratios), 2) if len(ratios) >= 5 else None,
            "charge_accept_rate": sum(r["accepted"] for r in issued) / n(issued),
            "income": round(sum(r["amount"] for r in issued), 2),
            "accept_rate": len(accepted) / n(reviewed),
            "wrong_accept_share": len(wrong) / len(known) if known else None,
            "penalty_rate": len(penalties) / n(reviewed),
            "costs": round(sum(r["amount"] for r in reviewed), 2),
            "cost_share_fair": share("fair"),
            "cost_share_inflated": share("inflated"),
            "cost_share_penalty": share("penalty"),
            "cost_share_unclassified": share("unclassified"),
        }

    teams = sorted(by_team, key=str.lower)
    full = {t: metrics(t, 1) for t in teams}
    recent = {t: metrics(t, last_game - RECENT_GAMES + 1) for t in teams}
    f_all, f_rec = _field(list(full.values())), _field(list(recent.values()))
    out = {}
    for team in teams:
        cl, cr = craftsman_label(full[team])
        il, ir = insurer_label(full[team], f_all)
        rcl, _ = craftsman_label(recent[team])
        ril, _ = insurer_label(recent[team], f_rec)
        b = belief.get(team, {})
        out[team] = {
            "is_us": team == US,
            "all": full[team], "recent": recent[team],
            "craftsman": {"label": cl, "reason": cr, "recent_label": rcl,
                          "model_markup": round(math.exp(b["alpha"]), 2) if "alpha" in b else None,
                          "model_spread": round(b["sigma"], 2) if "sigma" in b else None},
            "insurer": {"label": il, "reason": ir, "recent_label": ril,
                        "model_limit": round(math.exp(b["beta"]), 2) if "beta" in b else None,
                        "model_spread": round(b["tau"], 2) if "tau" in b else None},
        }
    return {"last_game": last_game, "recent_games": RECENT_GAMES, "field": f_all, "teams": out}


# ------------------------------------------------------------------ todo

TASK_RE = re.compile(r"^\s*[-*]\s+\[( |x|X)\]\s+(.*)$")


def parse_todo(text: str) -> list[dict]:
    """`## Owner` sections with `- [ ] task` / `- [x] task` lines. Tags: #tag, (game N)."""
    owner, out = "Team", []
    for line in text.splitlines():
        h = re.match(r"^##\s+(.+?)\s*$", line)
        if h:
            owner = h.group(1).strip()
            continue
        m = TASK_RE.match(line)
        if m:
            body = m.group(2).strip()
            out.append({"owner": owner, "text": body, "done": m.group(1) != " ",
                        "tags": re.findall(r"(?<!\S)#([\w-]+)", body), "source": "TODO.md"})
    return out


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=20).stdout
    except Exception:
        return ""


def owner_of(path: str) -> str:
    """Main git author of a file (most commits touching it)."""
    names = Counter(ALIASES.get(n, n) for n in _git("log", "--format=%an", "--", path).split("\n") if n)
    return names.most_common(1)[0][0] if names else "Team"


def summarize_subject(subj: str) -> str:
    """Auto-sync commits list every file; condense to the code files plus 'game data'."""
    m = re.match(r"^Auto-sync(?: \([^)]*\))?:\s*(.*)$", subj)
    if not m:
        return subj[:140]
    body = m.group(1)
    more = re.search(r"\+(\d+) more", body)
    body = re.sub(r"\+\d+ more", "", body)
    paths = [p for p in re.split(r"[,\s]+", body) if "/" in p or "." in p]
    data = [p for p in paths if p.startswith(("data/", "results/", "runs/"))]
    code = [p.rsplit("/", 1)[-1] for p in paths if p not in data]
    games = sorted({int(g) for p in data for g in re.findall(r"game_?0*(\d+)", p)})
    parts = list(dict.fromkeys(code))[:4]
    if data:
        parts.append(f"game data{' ' + ', '.join(map(str, games[-2:])) if games else ''}")
    if more:
        parts.append(f"+{more.group(1)} more")
    return "sync: " + (", ".join(parts) or "files")


def recent_work(limit: int = 6) -> dict[str, list[dict]]:
    log = _git("log", "--all", "--no-merges", "-n", "300", "--format=%an\x1f%cI\x1f%h\x1f%s")
    out: dict[str, list[dict]] = {m: [] for m in TEAM_MEMBERS}
    for line in log.splitlines():
        try:
            an, when, sha, subj = line.split("\x1f", 3)
        except ValueError:
            continue
        an = ALIASES.get(an, an)
        lst = out.setdefault(an, [])
        if len(lst) < limit:
            lst.append({"when": when, "sha": sha, "subject": summarize_subject(subj)})
    return out


def _max_game(names) -> int:
    nums = [int(m.group(1)) for n in names if (m := re.search(r"game_?0*(\d+)", n))]
    return max(nums, default=0)


def derived_tasks(last_game: int) -> list[dict]:
    """Open steps of the team's workflows (docs/WORKFLOW.md, docs/REPORT.md 6), read off
    the repo state. They disappear on their own when the step is done."""
    tasks = []

    def add(owner, text, tags, ref):
        tasks.append({"owner": owner, "text": text, "done": False, "tags": tags,
                      "source": "pipeline", "ref": ref})

    belief = ROOT / "results" / "opponent_belief.json"
    if belief.exists():
        fitted = max(json.loads(belief.read_text()).get("games", [0]) or [0])
        if fitted < last_game:
            add(owner_of("c2f/opponents.py"),
                f"Refit opponent belief: fitted up to game {fitted}, transactions up to {last_game} "
                "(python scripts/update_opponents.py --no-fetch)", ["market"], "c2f/opponents.py")

    reviewed = _max_game(p.name for p in (ROOT / "results" / "casebook").glob("game*.json"))
    for g in range(reviewed + 1, last_game + 1):
        add(owner_of("results/casebook"), f"Observer review of game {g} (casebook entry)",
            ["observer"], "docs/REPORT.md")

    # shadow report older (by commit time) than the newest shadow run
    def committed(path: str) -> int:
        return int(_git("log", "-1", "--format=%ct", "--", path).strip() or 0)
    shadow_files = list((ROOT / "results" / "shadow").glob("game*.json"))
    if shadow_files:
        newest = max(shadow_files, key=lambda p: (_max_game([p.name]), p.name))
        rel = str(newest.relative_to(ROOT)).replace("\\", "/")
        if committed("results/shadow/REPORT.md") < committed(rel):
            add(owner_of("c2f/shadow.py"), f"Shadow runs reached game {_max_game([newest.name])}: "
                "refresh the report (python -m c2f shadow-report)", ["shadow"],
                "results/shadow/REPORT.md")

    tx_dir = ROOT / "data" / "transactions"
    for f in sorted(tx_dir.glob("game*.json")):
        if json.loads(f.read_text()).get("complete") is False:
            add(owner_of("c2f/leaderboard.py"), f"{f.name} is incomplete: re-sync transactions",
                ["market"], str(f.relative_to(ROOT)))

    labels = [p for p in (ROOT / "data" / "labels").glob("*.json") if p.name != "TEMPLATE.json"]
    if not labels:
        add("milanova-n", "Label game 0 (data/labels/game0.json): independent labeller, "
            "see README 'Checking our prices'", ["qc"], "docs/WORKFLOW.md")

    wu = ROOT / "writeup" / "WRITEUP.md"
    if wu.exists():
        for m in re.finditer(r"\*\*(.+?)\*\*\s*_\(fill in", wu.read_text()):
            add("Team", f"Write-up: fill in '{m.group(1).rstrip('.')}'", ["writeup"], "writeup/WRITEUP.md")

    cl = ROOT / "CHANGELOG.md"
    if cl.exists() and "## unreleased" in cl.read_text():
        add("Team", "CHANGELOG has an unreleased section: cut a release (scripts/release.py)",
            ["release"], "CHANGELOG.md")

    for ref in _git("branch", "-r", "--no-merged", "origin/main", "--format=%(refname:short)").split():
        if ref.endswith("/HEAD"):
            continue
        author = ALIASES.get(a := _git("log", "-1", "--format=%an", ref).strip(), a) or "Team"
        add(author, f"Branch {ref} is not merged into main: merge or close it", ["git"], ref)
    return tasks


def todo(last_game: int) -> dict:
    path = ROOT / "TODO.md"
    manual = parse_todo(path.read_text(encoding="utf-8")) if path.exists() else []
    return {"items": manual + derived_tasks(last_game), "recent_work": recent_work(),
            "members": list(TEAM_MEMBERS)}


# ------------------------------------------------------------------ round briefing

BRIEF_DIR = OUT_DIR / "briefings"      # one cached JSON per finished game (git-ignored)
N_BRIEFINGS = 10


def our_transactions(game: int, fetch: bool) -> list[dict]:
    """Every row of the game where we are issuer or reviewer."""
    f = ROOT / "data" / "transactions" / f"game{game}.json"
    if f.exists() and json.loads(f.read_text()).get("complete", True):
        rows = json.loads(f.read_text())["items"]
        return [r for r in rows if US in (r["issuer"], r["reviewer"])]
    if not fetch:
        return []
    out, page = [], 1
    while True:
        url = f"{API}/transactions?game_id={game}&team={US}&page={page}&page_size=100"
        with urllib.request.urlopen(url, timeout=10) as r:
            doc = json.load(r)
        out += doc.get("items", [])
        if page >= doc.get("total_pages", 1):
            return out
        page += 1


def _repo_text(rel: str, refresh: bool = False) -> str | None:
    """A repo file from the working tree, or from origin/main when it is newer there.
    Lets briefings use the runner's pushed logs on a checkout that is not pulled
    (autosync paused or off); the working tree is never touched."""
    local = ROOT / rel
    txt = local.read_text(encoding="utf-8") if local.exists() else None
    if refresh:
        _git("fetch", "-q", "origin")
    remote = _git("show", f"origin/main:{rel}") or None
    if remote and (txt is None or len(remote) > len(txt)):
        return remote
    return txt


def line_names(game: int) -> dict[int, str]:
    cb = _repo_text(f"results/casebook/game{game}.json")
    if cb:
        return {int(k): v for k, v in json.loads(cb).get("lines", {}).items()}
    for ex in sorted((ROOT / "results" / "extractions").glob(f"game{game}__*.json")):
        items = json.loads(ex.read_text()).get("items", [])
        return {int(i["item_id"]): i.get("description", "") for i in items if str(i["item_id"]).isdigit()}
    return {}


def submission(game: int) -> dict | None:
    text = _repo_text("runs/submissions.jsonl", refresh=True)
    last = None
    if text:
        for line in text.splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("game_id") == game and not r.get("dry_run"):
                last = r
    return last


def make_briefing(game: int, sc: dict, strat: dict, rows: list[dict], latest: bool = True) -> dict:
    """Facts for one finished game, as a headline plus short sections of bullets."""
    gi = sc["games"].index(game)
    net = {t: v[gi] for t, v in sc["teams"].items()}
    cum = {t: sum(v[:gi + 1]) for t, v in sc["teams"].items()}
    prev = {t: sum(v[:gi]) for t, v in sc["teams"].items()}
    rank = lambda d: {t: k + 1 for k, (t, _) in enumerate(sorted(d.items(), key=lambda x: -x[1]))}  # noqa: E731
    r_game, r_cum, r_prev = rank(net), rank(cum), rank(prev)
    n = len(net)
    winner = max(net, key=net.get)
    above = sorted((t for t in cum if r_cum[t] == r_cum[US] - 1), key=cum.get)
    sections = []

    res = [f"Net {net[US]:+,.0f}, rank {r_game[US]} of {n} in this game "
           f"(winner {winner} {net[winner]:+,.0f}).",
           f"Total {cum[US]:+,.0f}, rank {r_cum[US]} of {n}"
           + (f" (was {r_prev[US]})." if gi and r_prev[US] != r_cum[US] else " (unchanged).")]
    if above:
        res.append(f"{cum[above[0]] - cum[US]:,.0f} behind {above[0]} in rank {r_cum[US] - 1}.")
    sections.append({"title": "Result", "bullets": res})

    if rows:
        names = line_names(game)
        mine = [r for r in rows if r["issuer"] == US]
        theirs = [r for r in rows if r["reviewer"] == US]
        acc = [r for r in mine if r["accepted"]]
        fair_rej = [r for r in mine if not r["accepted"] and r["amount"] > 0]
        zero = [r for r in mine if not r["accepted"] and r["amount"] == 0]
        income = sum(r["amount"] for r in mine)
        cost = sum(r["amount"] for r in theirs)
        pen = [r for r in theirs if not r["accepted"] and r["amount"] > 0]
        paid = [r for r in theirs if r["accepted"]]
        sections.append({"title": "As handyman", "bullets": [
            f"Received {income:,.0f} on {len(mine)} bills: {len(acc)} accepted, "
            f"{len(fair_rej)} rejected but fair (paid 1.5a), {len(zero)} rejected for nothing.",
        ]})
        sections.append({"title": "As insurer", "bullets": [
            f"Paid {cost:,.0f}: {len(paid)} bills accepted ({sum(r['amount'] for r in paid):,.0f}), "
            f"{len(pen)} fair bills rejected with penalty ({sum(r['amount'] for r in pen):,.0f}).",
        ]})
        per = defaultdict(float)
        for r in mine:
            per[r["line_item_index"]] += r["amount"]
        for r in theirs:
            per[r["line_item_index"]] -= r["amount"]
        ranked = sorted(per.items(), key=lambda kv: kv[1])
        lab = lambda i: f"line {i}" + (f" ({names[i][:48]})" if names.get(i) else "")  # noqa: E731
        lines = [f"Worst: {lab(i)} {v:+,.0f}" for i, v in ranked[:2] if v < 0]
        lines += [f"Best: {lab(i)} {v:+,.0f}" for i, v in ranked[::-1][:1] if v > 0]
        if lines:
            sections.append({"title": "Lines that decided it", "bullets": lines})

    sub = submission(game)
    if sub:
        tm = sub.get("timing", {}) or {}
        stages = [s for s in tm.get("stages", []) if s.get("submitted")]
        errs = [(s["stage"], e) for s in tm.get("stages", [])
                for e in ([s["error"]] if s.get("error") else []) + list(s.get("errors") or [])]
        if not tm:   # the run stopped before any stage, e.g. no case key: nothing was submitted
            ops = ["WARNING: nothing was submitted, every line counted as a = 0, b = 0."]
        else:
            ops = [f"Final stage {tm.get('final_stage', '?')} ({tm.get('model', '?')}) after "
                   f"{tm.get('since_start_s', 0):.0f} s, {tm.get('n_invoice_lines', '?')} lines, "
                   f"{len(stages)} submissions."]
            if tm.get("final_stage") in ("fixed", None) or not sub.get("ok"):
                ops.append("WARNING: no model prices went out, fallback prices only.")
        if sub.get("error"):
            ops.append(f"Error: {sub['error'][:110]}")
        for stage, e in errs:
            short = re.search(r"'message': '([^']+)'", e)
            ops.append(f"Error in stage {stage}: {(short.group(1) if short else e)[:110]}")
        sections.append({"title": "Operations", "bullets": ops})
    else:
        sections.append({"title": "Operations", "bullets": ["Submission log for this game not synced yet."]})

    shifts = [f"{t}: {v['craftsman']['label']} -> {v['craftsman']['recent_label']} as handyman"
              for t, v in strat["teams"].items() if not v["is_us"]
              and v["craftsman"]["recent_label"] != v["craftsman"]["label"]]
    shifts += [f"{t}: {v['insurer']['label']} -> {v['insurer']['recent_label']} as insurer"
               for t, v in strat["teams"].items() if not v["is_us"]
               and v["insurer"]["recent_label"] != v["insurer"]["label"]]
    if shifts and latest:
        sections.append({"title": f"Opponents changing course (last {strat['recent_games']} games)",
                         "bullets": shifts[:6]})

    move = ("up" if r_cum[US] < r_prev[US] else "down" if r_cum[US] > r_prev[US] else "steady") if gi else "steady"
    headline = (f"Game {game}: {net[US]:+,.0f} (rank {r_game[US]}/{n}), "
                f"total {cum[US]:+,.0f}, rank {r_cum[US]}/{n} {move}")
    return {"game": game, "headline": headline, "sections": sections,
            "complete": bool(rows) and sub is not None}


def briefing_markdown(b: dict) -> str:
    out = [f"**{b['headline']}**", ""]
    for s in b["sections"]:
        out.append(f"*{s['title']}*")
        out += [f"- {x}" for x in s["bullets"]]
        out.append("")
    return "\n".join(out).strip()


def briefings(sc: dict, strat: dict, fetch: bool) -> list[dict]:
    """Briefings for the last N_BRIEFINGS games, newest first. A briefing is cached once it is
    complete (our transactions and the submission log are both in)."""
    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for g in sc["games"][::-1][:N_BRIEFINGS]:
        cache = BRIEF_DIR / f"game{g}.json"
        if cache.exists():
            out.append(json.loads(cache.read_text()))
            continue
        try:
            latest = g == sc["games"][-1]
            b = make_briefing(g, sc, strat, our_transactions(g, fetch), latest)
        except Exception as e:  # network or data hiccup: show the score part at least
            b = make_briefing(g, sc, strat, [], g == sc["games"][-1])
            b["sections"].append({"title": "Note", "bullets": [f"Line details unavailable: {type(e).__name__}"]})
        if b["complete"]:
            cache.write_text(json.dumps(b, indent=1))
        out.append(b)
    return out


# ------------------------------------------------------------------ build / serve

DROP_GAMES = 3        # absent in each of a team's last 3 played games ...
DROP_SHARE = 0.8      # ... on at least 80% of the lines -> dropped out


def dropped_out(ab_rows: list[dict]) -> dict[str, int]:
    """Teams that stopped playing: {team: first game of the current absent run}.
    Games where nobody quoted (an empty case) are skipped, so they cannot fake a dropout."""
    share = defaultdict(lambda: [0, 0])          # (team, game) -> [absent lines, lines]
    played = set()
    for r in ab_rows:
        g, absent = int(r["round"]), r["absent"] == "True"
        c = share[(r["team"], g)]
        c[0] += absent
        c[1] += 1
        if not absent:
            played.add(g)
    out = {}
    for team in {t for t, _ in share}:
        games = sorted(g for t, g in share if t == team and g in played)
        absent = [g for g in games if share[(team, g)][0] >= DROP_SHARE * share[(team, g)][1]]
        if len(games) >= DROP_GAMES and set(games[-DROP_GAMES:]) <= set(absent):
            since = games[-1]
            for g in reversed(games):
                if g not in absent:
                    break
                since = g
            out[team] = since
    return out


def tx_docs(tx_dir: Path) -> list[dict]:
    """Every game's public transactions: the working tree's files, plus games that only
    origin/main has yet (a checkout that is not pulled lags behind the runner's pushes)."""
    docs = {}
    for f in tx_dir.glob("game*.json"):
        doc = json.loads(f.read_text())
        docs[int(doc["game_id"])] = doc
    try:
        rel = tx_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:                      # a directory outside the repo: files only
        return [docs[g] for g in sorted(docs)]
    for name in _git("ls-tree", "--name-only", "origin/main", rel + "/").split():
        m = re.search(r"game(\d+)\.json$", name)
        if m and int(m.group(1)) not in docs:
            text = _git("show", f"origin/main:{name}")
            if text:
                doc = json.loads(text)
                docs[int(doc["game_id"])] = doc
    return [docs[g] for g in sorted(docs)]


def fresh_ab(fetch: bool) -> list[dict]:
    """results/ab_table.csv from the working tree, or from origin/main when that is newer
    (a checkout that is not pulled would otherwise label teams on stale games)."""
    text = _repo_text("results/ab_table.csv", refresh=fetch)
    return list(csv.DictReader(io.StringIO(text))) if text else []


def build(fetch: bool = True) -> dict:
    tx_dir = ROOT / "data" / "transactions"
    sc = scores(tx_dir, fetch)
    ab_rows = fresh_ab(fetch)
    strat = strategies(ab_rows, tx_dir, ROOT / "results" / "opponent_belief.json")
    last_game = max(sc["games"][-1] if sc["games"] else 0, strat["last_game"])
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD").strip(),
        "us": US,
        "scores": sc,
        "strategies": strat,
        "todo": todo(last_game),
        "briefings": briefings(sc, strat, fetch),
        "dropped": dropped_out(ab_rows),
    }
    write(data)
    return data


ART_DIR = OUT_DIR / "artifact"


def build_artifact() -> Path:
    """Artifact copy of the page: the publisher adds the document skeleton itself, so the
    doctype/html/head/body tags go; data.js is published beside it."""
    html = (OUT_DIR / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"(?is)<!doctype html>\s*|</?html[^>]*>\s*|</?head>\s*|</?body>\s*", "", html)
    html = re.sub(r'(?i)<meta charset="utf-8">\s*<meta name="viewport"[^>]*>\s*', "", html)
    ART_DIR.mkdir(exist_ok=True)
    (ART_DIR / "index.html").write_text(html, encoding="utf-8")
    (ART_DIR / "data.js").write_text((OUT_DIR / "data.js").read_text(encoding="utf-8"), encoding="utf-8")
    return ART_DIR


def write(data: dict) -> None:
    """data.json for tools, data.js for the page (a <script> also loads from file://)."""
    OUT_DIR.mkdir(exist_ok=True)
    text = json.dumps(data, indent=1)
    for name, body in (("data.json", text), ("data.js", f"window.DASH_DATA = {text};\n")):
        tmp = OUT_DIR / (name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(OUT_DIR / name)


# ------------------------------------------------------------------ TODO.md edits

TODO_PATH = ROOT / "TODO.md"
_todo_lock = threading.Lock()


def edit_todo(text: str, action: str, owner: str, task: str, done: bool = False) -> str:
    """Return TODO.md text with one task ticked/unticked ("toggle") or appended ("add")
    in the `## owner` section (created at the end if missing). Other lines are untouched."""
    task = " ".join(task.split())
    if not task:
        raise ValueError("empty task")
    lines = text.splitlines()
    section, start, end = "Team", None, len(lines)
    for i, line in enumerate(lines):
        h = re.match(r"^##\s+(.+?)\s*$", line)
        if h:
            if section == owner and start is not None:
                end = i
                break
            section = h.group(1).strip()
            if section == owner:
                start = i
    if action == "toggle":
        lo = start + 1 if start is not None else 0
        for i in range(lo, end):
            m = TASK_RE.match(lines[i])
            if m and m.group(2).strip() == task:
                lines[i] = re.sub(r"\[( |x|X)\]", "[x]" if done else "[ ]", lines[i], count=1)
                return "\n".join(lines) + "\n"
        raise KeyError(f"task not found in section {owner!r}: {task!r}")
    if action == "add":
        new = f"- [ ] {task}"
        if start is None:
            while lines and not lines[-1].strip():
                lines.pop()
            lines += ["", f"## {owner}", new]
        else:
            at = end
            while at > start + 1 and not lines[at - 1].strip():
                at -= 1
            lines.insert(at, new)
        return "\n".join(lines) + "\n"
    raise ValueError(f"unknown action {action!r}")


class Handler(SimpleHTTPRequestHandler):
    fetch = True

    def log_message(self, *a):  # quiet
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/ping":
            return self._json(200, {"ok": True})
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/todo":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(min(n, 10_000)) or b"{}")
            with _todo_lock:
                old = TODO_PATH.read_text(encoding="utf-8") if TODO_PATH.exists() else "# Team ToDo\n"
                new = edit_todo(old, req.get("action", ""), str(req.get("owner", "Team")),
                                str(req.get("text", "")), bool(req.get("done")))
                TODO_PATH.write_text(new, encoding="utf-8")
            sc = json.loads((OUT_DIR / "data.json").read_text())["scores"]["games"] \
                if (OUT_DIR / "data.json").exists() else []
            t = todo(sc[-1] if sc else 0)
            # keep data.json in step without a full rebuild
            if (OUT_DIR / "data.json").exists():
                data = json.loads((OUT_DIR / "data.json").read_text())
                data["todo"] = t
                write(data)
            return self._json(200, {"todo": t})
        except (KeyError, ValueError) as e:
            return self._json(400, {"error": str(e)})


def serve(port: int, interval: float, fetch: bool) -> None:
    def loop():
        briefed = None   # newest game whose briefing was printed (the first build prints the latest)
        while True:
            try:
                d = build(fetch)
                print(f"[{time.strftime('%H:%M:%S')}] data rebuilt: games to "
                      f"{d['scores']['games'][-1] if d['scores']['games'] else '-'}, "
                      f"{len(d['todo']['items'])} tasks", flush=True)
                latest = d["briefings"][0] if d["briefings"] else None
                if latest and latest["game"] != briefed:
                    briefed = latest["game"]
                    print("\n" + briefing_markdown(latest) + "\n", flush=True)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] build failed: {e!r}", flush=True)
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()
    handler = partial(Handler, directory=str(OUT_DIR))
    print(f"dashboard on http://localhost:{port}  (rebuild every {interval:.0f} s, Ctrl+C stops)")
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", default="build", choices=["build", "serve", "brief", "artifact"])
    ap.add_argument("--game", type=int, default=None, help="brief: game id (default: latest)")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--interval", type=float, default=120)
    ap.add_argument("--no-fetch", action="store_true", help="do not call the leaderboard API")
    args = ap.parse_args()
    if args.cmd == "serve":
        serve(args.port, args.interval, not args.no_fetch)
    elif args.cmd in ("brief", "artifact"):
        d = build(not args.no_fetch)
        bs = {b["game"]: b for b in d["briefings"]}
        b = bs.get(args.game) if args.game else (d["briefings"][0] if d["briefings"] else None)
        if args.cmd == "artifact":
            print(f"artifact files in {build_artifact()}")
        print(briefing_markdown(b) if b else "no briefing for that game")
    else:
        d = build(not args.no_fetch)
        print(f"wrote {OUT_DIR / 'data.json'}: {len(d['scores']['teams'])} teams, "
              f"games {d['scores']['games'][:1]}..{d['scores']['games'][-1:]}, "
              f"{len(d['todo']['items'])} tasks")


if __name__ == "__main__":
    main()
