"""The belief (median, sigma, p_zero) behind every quote we submitted, for every game.

Why this exists: the run log (runs/submissions.jsonl) only stores the belief of the
submitted stage since game 44 ("estimates"). Before that only the quotes (a, b) were logged,
and results/extractions/ holds the answers of the OpenAI stages only. From game 26 on the
submitted answer usually came from the backup stage (claude-opus-5-5), which is stored
nowhere, so for games 26-43 the stored extraction is NOT what we priced from.

Sources, in this order:
    logged         "estimates" in the run log (game 44 on), or results/live/game<N>.json
    extraction     final stage "full"/"fast": the stored extraction of that stage, turned
                   into a belief with the config of that game
    reconstructed  inverted from the submitted (a, b) with the config of that game

Reconstruction. Under the fixed rule both quotes are the estimate times a known factor:
    a = m * g(sigma)      g = charge_min_ratio (0.87) while the floor binds, the confident
                          ratio (0.95) for sigma below charge_conf_sigma, otherwise one of
                          the n_candidates log-spaced search points between the 5 % and
                          95 % quantile (pricing.charge_price)
    b = m * shrink * exp(sigma * z((1/3 - p_zero) / (1 - p_zero)))   and 0 for p_zero >= 1/3
The backup model states whole-euro medians, sigma in steps of 0.05 and p_zero in steps of
0.01 (true for all 100 lines logged in games 44-55). So we enumerate (sigma, g, p_zero), keep
the combinations whose m = a / g is a whole number and that reproduce a and b to the cent,
and report all of them. One combination left: "exact". Several with the same median:
"median_exact". b = 0: only the median is identified (p_zero >= 1/3, sigma unknown).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tomllib
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

from .config import REPO_ROOT

RUN_LOG = REPO_ROOT / "runs" / "submissions.jsonl"
LIVE_DIR = REPO_ROOT / "results" / "live"
EXTRACTIONS = REPO_ROOT / "results" / "extractions"
_STD = NormalDist()
P_ZERO_GRID = [i / 100 for i in range(0, 34)]
# How often the backup model stated each value on the 100 lines logged in games 44-55. Only
# used to order several beliefs that all reproduce the same quotes (status "median_exact").
_SIGMA_SEEN = {0.1: 4, 0.2: 25, 0.25: 6, 0.3: 14, 0.35: 7, 0.4: 14, 0.45: 6, 0.5: 17, 0.6: 7}
_P_ZERO_SEEN = {0.03: 1, 0.05: 6, 0.07: 1, 0.08: 7, 0.1: 11, 0.12: 9, 0.15: 18, 0.18: 2, 0.2: 9, 0.25: 4, 0.3: 1}


def _likely(sol: tuple) -> float:
    return _SIGMA_SEEN.get(sol[1], 0.5) * _P_ZERO_SEEN.get(sol[2], 0.5) * (2.0 if sol[3] else 1.0)


# ---------------------------------------------------------------- config of a past game

def _git(*a: str) -> bytes:
    return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, check=False).stdout


@lru_cache(maxsize=1)
def _config_blobs() -> dict[str, dict]:
    """config sha (as stamped in the run log) -> parsed configs/default.toml of that version."""
    out: dict[str, dict] = {}
    for h in _git("log", "--format=%H", "--", "configs/default.toml").decode().split():
        data = _git("show", f"{h}:configs/default.toml")
        if data:
            out.setdefault(hashlib.sha256(data).hexdigest()[:12], tomllib.loads(data.decode()))
    return out


def rule_of(config: dict) -> dict:
    """The numbers of one config version that tie a belief to its quotes."""
    p, b, fb = config.get("pricing", {}), config.get("belief", {}), config.get("fallback", {})
    shrinks = [float(p.get("acceptance_shrink", 1.0))]
    if p.get("acceptance_shrink_from"):            # learned factor, capped: any step up to the cap
        cap = float(p.get("acceptance_shrink_cap", shrinks[0]))
        shrinks = sorted({round(1.0 + i / 100, 2) for i in range(0, int(round((cap - 1) * 100)) + 1)} | {shrinks[0]})
    return dict(
        min_ratio=float(p.get("charge_min_ratio", 0.0)),
        conf_sigma=float(p.get("charge_conf_sigma", 0.0)), conf_ratio=float(p.get("charge_conf_ratio", 0.0)),
        q=float(p.get("acceptance_quantile", 1 / 3)), shrinks=shrinks,
        q_lo=float(p.get("charge_floor_quantile", 0.05)), q_hi=float(p.get("charge_ceiling_quantile", 0.95)),
        n_cand=int(p.get("n_candidates", 60)), n_t=int(p.get("n_t_grid", 100)),
        n_draws=int(p.get("n_opp_draws", 15)),
        opponents_from=p.get("opponents_from", ""), team=p.get("team_name", ""),
        min_limit_obs=float(p.get("min_limit_obs", 3.0)),
        static_opponents=tuple((float(o.get("weight", 1.0)), o.get("kind", "noisy"), float(o.get("mu", 0.0)),
                                float(o.get("s", 0.3))) for o in p.get("opponents", [])),
        min_sigma=float(b.get("min_sigma", 0.15)), max_sigma=float(b.get("max_sigma", 1.2)),
        confident_below=float(b.get("confident_below", 0.0)),
        confident_min_sigma=float(b.get("confident_min_sigma", 0.1)),
        fallback=(float(fb.get("charge_price", 0.0)), float(fb.get("acceptance_limit", 0.0))))


def rule_for_sha(config_sha: str) -> dict | None:
    c = _config_blobs().get(config_sha)
    return rule_of(c) if c else None


# ---------------------------------------------------------------- forward and inverse rule

def sigma_grid(rule: dict) -> list[float]:
    lo = int(round(rule["min_sigma"] * 100))
    grid = [s / 100 for s in range(lo, int(round(rule["max_sigma"] * 100)) + 1, 5)]
    if rule["confident_below"] > 0:
        grid = [rule["confident_min_sigma"]] + grid
    return grid


def _search_points(sigma: float, rule: dict) -> list[float]:
    z_lo, z_hi = _STD.inv_cdf(rule["q_lo"]), _STD.inv_cdf(rule["q_hi"])
    n = max(rule["n_cand"], 2)
    return [math.exp(sigma * (z_lo + (z_hi - z_lo) * i / (n - 1))) for i in range(n)]


def search_index(sigma: float, rule: dict, opponents) -> int:
    """Index of the search point pricing.charge_price picks for this sigma against this
    opponent population (same grids as pricing.expected_revenue, vectorised; the result
    scales with the median, so the median is 1 here)."""
    import numpy as np
    a = np.array(_search_points(sigma, rule))[:, None, None]                    # candidates
    t = np.exp(sigma * np.array([_STD.inv_cdf((i + 0.5) / rule["n_t"]) for i in range(rule["n_t"])]))[None, :, None]
    z = np.array([_STD.inv_cdf((i + 0.5) / rule["n_draws"]) for i in range(rule["n_draws"])])[None, None, :]
    fair = a <= t
    rev = np.zeros((a.shape[0], t.shape[1]))
    total = sum(w for w, *_ in opponents)
    for w, kind, mu, sd in opponents:
        if kind == "absent":
            pay = np.where(fair, 1.5 * a, 0.0)[:, :, 0]
        elif kind == "accept_all":
            pay = np.where(fair, a, np.minimum(a, 4.0 * t))[:, :, 0]
        else:
            acc = a <= t * np.exp(mu + sd * z)
            pay = np.where(acc, np.where(fair, a, np.minimum(a, 4.0 * t)), np.where(fair, 1.5 * a, 0.0)).mean(axis=2)
        rev += w / total * pay
    return int(rev.mean(axis=1).argmax())


def opponents_from_json(text: str, rule: dict) -> tuple:
    """Opponent population of one stored results/opponent_belief.json as (weight, kind, mu, s)."""
    from .opponents import OpponentBelief, opponent_types
    types = opponent_types(OpponentBelief.from_json(text), us=rule["team"], min_limits=rule["min_limit_obs"])
    return tuple((o.weight, o.kind, o.mu, o.s) for o in types)


_ALLOWED: dict = {}


def allowed_for(git_sha: str, config_sha: str) -> dict[float, set[int]] | None:
    """Search-point indices per sigma for one game: the opponent belief file as committed at
    the stamped commit, the version before it and the one after it (the bot's checkout is
    usually a little ahead of or behind its HEAD). None if nothing can be loaded."""
    rule = rule_for_sha(config_sha)
    if rule is None:
        return None
    key = (git_sha, config_sha)
    if key in _ALLOWED:
        return _ALLOWED[key]
    pops: list[tuple] = []
    path = rule["opponents_from"]
    if path:
        before = _git("log", "-2", "--format=%H", git_sha, "--", path).decode().split()
        after = _git("log", "--reverse", "--format=%H", f"{git_sha}..HEAD", "--", path).decode().split()[:1]
        seen = set()
        for h in before + after:
            text = _git("show", f"{h}:{path}")
            if not text or hash(text) in seen:
                continue
            seen.add(hash(text))
            try:
                pops.append(opponents_from_json(text.decode("utf-8"), rule))
            except Exception:  # noqa: BLE001  an unreadable old version only narrows the candidates
                pass
    if not pops and rule["static_opponents"]:
        pops = [rule["static_opponents"]]
    _ALLOWED[key] = allowed_indices(rule, pops) if pops else None
    return _ALLOWED[key]


def allowed_indices(rule: dict, populations: list[tuple]) -> dict[float, set[int]]:
    """sigma -> search-point indices the rule picks under any of the given opponent populations."""
    return {sg: {search_index(sg, rule, pop) for pop in populations if pop} for sg in sigma_grid(rule)}


def charge_factors(sigma: float, rule: dict, allowed: dict[float, set[int]] | None = None) -> list[float]:
    """Every value a / m can take for this sigma: the floor, or a search point above it.
    allowed (from allowed_indices) restricts the search points to the ones the charge search
    really picks; without it every point above the floor is a candidate."""
    floor = rule["min_ratio"]
    if sigma < rule["conf_sigma"]:
        floor = max(floor, rule["conf_ratio"])
    pts = _search_points(sigma, rule)
    idx = range(len(pts)) if allowed is None else sorted(allowed.get(sigma, ()))
    return [floor] + [pts[i] for i in idx if floor < pts[i] <= 1.6]


def limit_of(m: float, sigma: float, p_zero: float, rule: dict, shrink: float) -> float:
    if rule["q"] <= p_zero or m <= 0:
        return 0.0
    q_pos = min(max((rule["q"] - p_zero) / (1.0 - p_zero), 1e-9), 1 - 1e-9)
    return round(m * math.exp(sigma * _STD.inv_cdf(q_pos)) * shrink, 2)


def reconstruct(a: float, b: float, rule: dict, allowed: dict[float, set[int]] | None = None) -> dict:
    """Beliefs that reproduce the submitted (a, b) to the cent. See the module docstring."""
    if a <= 0:
        return {"status": "no_charge", "median": 0.0, "sigma": None, "p_zero": None, "solutions": []}
    if (round(a, 2), round(b, 2)) == tuple(round(x, 2) for x in rule["fallback"]):
        return {"status": "fallback_price", "median": None, "sigma": None, "p_zero": None, "solutions": []}
    sols = []
    for sigma in sigma_grid(rule):
        for k, g in enumerate(charge_factors(sigma, rule, allowed)):
            if g <= 0:                              # config without a floor: search points only
                continue
            m = round(a / g)
            if m <= 0 or abs(round(g * m, 2) - a) > 0.0051:
                continue
            if b <= 0:
                sols.append((m, None, None, k == 0, None))
                continue
            for shrink in rule["shrinks"]:
                for p0 in P_ZERO_GRID:
                    if abs(limit_of(m, sigma, p0, rule, shrink) - b) <= 0.0051:
                        sols.append((m, sigma, p0, k == 0, shrink))
    sols = sorted(set(sols), key=lambda s: (-_likely(s), s[1] or 0, s[2] or 0))
    medians = sorted({s[0] for s in sols})
    listed = [{"median": float(m), "sigma": s, "p_zero": p, "floor_binds": fl, "shrink": sh}
              for m, s, p, fl, sh in sols][:12]
    if not sols:                                    # outside the grid: the floor ratio as a fallback
        m = round(a / rule["min_ratio"], 2) if rule["min_ratio"] > 0 else None
        return {"status": "approximate", "median": m, "sigma": None,
                "p_zero": None if b > 0 else ">=1/3", "solutions": []}
    on_floor = [s for s in sols if s[3]]
    best = (on_floor or sols)[0] if b <= 0 else sols[0]
    if b <= 0:
        status = "median_only" if len(medians) == 1 else "ambiguous"
        return {"status": status, "median": float(best[0]), "sigma": None, "p_zero": ">=1/3", "solutions": listed}
    distinct = {(s[0], s[1], s[2]) for s in sols}
    status = "exact" if len(distinct) == 1 else "median_exact" if len(medians) == 1 else "ambiguous"
    return {"status": status, "median": float(best[0]), "sigma": best[1], "p_zero": best[2], "solutions": listed}


# ---------------------------------------------------------------- all games

def _index(item_id: str, fallback: int) -> int:
    m = re.match(r"\s*(\d+)", str(item_id))
    return int(m.group(1)) if m else fallback


def load_run_log(path: Path | None = None) -> dict[int, dict]:
    """game -> last successful live record."""
    out: dict[int, dict] = {}
    p = path or RUN_LOG
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("ok") and not d.get("dry_run") and d.get("game_id"):
                out[int(d["game_id"])] = d
    return out


def _from_extraction(game: int, stage: str, model: str | None, rule: dict) -> dict[int, dict] | None:
    tag = "-noimg" if stage == "fast" else ""
    files = sorted(EXTRACTIONS.glob(f"game{game}{tag}__x*__{model}.json")) if model else []
    if not files:
        return None
    out = {}
    for k, it in enumerate(json.loads(files[-1].read_text(encoding="utf-8")).get("items", []), 1):
        sd = float(it.get("price_log_sd", 0.5))
        if sd < rule["confident_below"]:
            sigma = max(sd, rule["confident_min_sigma"])
        else:
            sigma = min(max(sd, rule["min_sigma"]), rule["max_sigma"])
        out[_index(it.get("item_id", k), k)] = {
            "median": round(float(it.get("quantity") or 1) * max(float(it.get("unit_price_gross_eur", 0)), 0.0), 2),
            "sigma": round(sigma, 4),
            "p_zero": round(1.0 - float(it.get("covered_prob", 1)) * float(it.get("related_prob", 1)), 4)}
    return out


def game_estimates(rec: dict) -> dict:
    """One game of the run log -> {game, source, stage, model, lines: [...]}."""
    game, t = int(rec["game_id"]), rec.get("timing") or {}
    stage, model = t.get("final_stage"), t.get("model")
    out = {"game": game, "stage": stage, "model": model, "config_sha": rec["stamp"]["config_sha"]}
    payload = {int(p["index"]): p for p in rec.get("payload") or []}
    if rec.get("estimates"):
        est = {_index(e["item_id"], k): e for k, e in enumerate(rec["estimates"], 1)}
        out["source"] = "logged"
        out["lines"] = [{"index": i, "a": p["charge_price"], "b": p["acceptance_limit"],
                         **({k: est[i][k] for k in ("median", "sigma", "p_zero")} if i in est
                            else {"median": None, "sigma": None, "p_zero": None}),
                         "status": "logged" if i in est else "not_priced"} for i, p in sorted(payload.items())]
        return out
    rule = rule_for_sha(rec["stamp"]["config_sha"])
    if rule is None:
        out.update(source="unknown_config", lines=[])
        return out
    ext = _from_extraction(game, stage, model, rule) if stage in ("full", "fast") else None
    allowed = allowed_for(rec["stamp"]["git_sha"], rec["stamp"]["config_sha"])
    lines = []
    for i, p in sorted(payload.items()):
        a, b = float(p["charge_price"]), float(p["acceptance_limit"])
        r = reconstruct(a, b, rule, allowed)
        e = (ext or {}).get(i)
        if e and e["median"] > 0:                   # stored answer of the submitted stage
            ok = abs(a / e["median"] - (r["median"] and a / r["median"] or 0)) < 0.02 or any(
                abs(a - round(g * e["median"], 2)) <= 0.011 for g in charge_factors(e["sigma"], rule, None))
            lines.append({"index": i, "a": a, "b": b, **e,
                          "status": "extraction" if ok else "extraction_mismatch"})
        else:
            lines.append({"index": i, "a": a, "b": b, "median": r["median"], "sigma": r["sigma"],
                          "p_zero": r["p_zero"], "status": r["status"],
                          **({"solutions": r["solutions"]} if r["status"] in ("ambiguous", "median_exact") else {})})
    out["source"] = "extraction" if ext else "reconstructed"
    out["lines"] = lines
    return out


def live_files(directory: Path | None = None) -> dict[int, dict]:
    """game -> submitted stage of results/live/game<N>__<role>.json (written since the fix in
    c2f/__main__.py::_save_live). With two machines the primary's file wins."""
    out: dict[int, dict] = {}
    d = directory or LIVE_DIR
    for f in sorted(d.glob("game*__*.json"), key=lambda f: "__primary" in f.name):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            stage = next(s for s in reversed(doc["stages"]) if s["submitted"] and s["lines"])
        except (ValueError, KeyError, StopIteration, OSError):
            continue
        out[int(doc["game"])] = {
            "game": int(doc["game"]), "stage": stage["stage"], "model": stage.get("model"),
            "config_sha": (doc.get("stamp") or {}).get("config_sha"), "source": "logged",
            "lines": [{"index": ln["index"], "a": ln["a"], "b": ln["b"], "median": ln["median"],
                       "sigma": ln["sigma"], "p_zero": ln["p_zero"], "status": "logged"} for ln in stage["lines"]]}
    return out


def all_games(path: Path | None = None, live_dir: Path | None = None) -> list[dict]:
    """Every game we have a record of: the run log, completed by the per-game live files
    (which also cover games another machine submitted)."""
    games = {g: game_estimates(rec) for g, rec in load_run_log(path).items()}
    for g, doc in live_files(live_dir).items():
        if g not in games or games[g]["source"] != "logged":
            games[g] = doc
    return [games[g] for g in sorted(games)]


def write(games: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """results/live_estimates/games.json (everything, with the alternative beliefs of
    ambiguous lines) and lines.csv (one row per line, for analysis)."""
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, cp = out_dir / "games.json", out_dir / "lines.csv"
    jp.write_text(json.dumps(games, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8")
    cols = ["game", "index", "a", "b", "median", "sigma", "p_zero", "status", "source", "stage", "model"]
    with cp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for g in games:
            for ln in g["lines"]:
                row = {**ln, "game": g["game"], "source": g["source"], "stage": g["stage"], "model": g["model"]}
                w.writerow(["" if row.get(c) is None else row.get(c) for c in cols])
    return jp, cp
