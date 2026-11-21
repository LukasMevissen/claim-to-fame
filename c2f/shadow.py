"""Shadow runs: other models assess the same case in parallel to the live workflow.

Nothing here is submitted. Each shadow model's estimates and the quotes they would have
produced go to results/shadow/game<N>__<model>__<effort>.json; once the game's public
transactions are in, `python -m c2f shadow-report` scores every model with the same replay
(c2f/replay.py) next to what we actually submitted, and writes results/shadow/REPORT.md.

Models come from [shadow].models in the config as "provider:model@effort", e.g.
"anthropic:claude-fable-5-2@low" or "openai:gpt-6-luna@low".
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .config import REPO_ROOT, Config

DIR = REPO_ROOT / "results" / "shadow"
RUN_LOG = REPO_ROOT / "runs" / "submissions.jsonl"


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    provider, _, rest = spec.partition(":")
    model, _, effort = rest.partition("@")
    return provider, model, effort or None


def _path(game: int, model: str, effort: str | None) -> Path:
    return DIR / f"game{game}__{model}__{effort or 'default'}.json"


def _one(case, cfg: Config, spec: str, game: int, reference: str, timeout: float) -> dict:
    from .agent import price_items
    provider, model, effort = parse_spec(spec)
    rec: dict = {"game": game, "provider": provider, "model": model, "effort": effort}
    t0 = time.monotonic()
    try:
        if provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
            from .anthropic_extract import extract_claude
            items, meta = extract_claude(case, model, timeout, reference, effort)
        elif provider == "openai":
            from .extract import extract
            items, meta = extract(case, model, timeout, 0.0, False, reference=reference,
                                  effort=effort, tag="-shadow", save=False), {}
        else:
            raise ValueError(f"unknown provider {provider!r}")
        priced = price_items(items, cfg)
        rec.update(ok=True, secs=round(time.monotonic() - t0, 1), **meta,
                   items=[asdict(i) for i in items],
                   quotes={p.item_id: [p.charge_price, p.acceptance_limit] for p in priced})
    except Exception as e:  # noqa: BLE001  a shadow failure is a result, not a crash
        rec.update(ok=False, secs=round(time.monotonic() - t0, 1), error=repr(e)[:300])
    DIR.mkdir(parents=True, exist_ok=True)
    _path(game, model, effort).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def run(game: int, cfg: Config, client=None, specs: list[str] | None = None,
        prior_only: bool = True, log=print) -> list[dict]:
    """Assess game `game` (already started) with every shadow model. Never submits.
    prior_only: the price book only shows games before this one (as in a live run)."""
    from . import pricebook
    from .api import QuantCoClient
    from .case_io import decrypt, load_case

    sh = cfg.section("shadow")
    specs = specs if specs is not None else list(sh.get("models", []))
    from .env import load_env
    load_env(REPO_ROOT / ".env")          # pick up a key added while the scheduler runs
    if not os.environ.get("ANTHROPIC_API_KEY"):
        skipped = [s for s in specs if s.startswith("anthropic:")]
        specs = [s for s in specs if not s.startswith("anthropic:")]
        if skipped:
            log(f"[shadow] game {game}: ANTHROPIC_API_KEY not set, skipping {len(skipped)} model(s)")
    if not specs:
        return []
    client = client or QuantCoClient(cfg)
    zip_path = REPO_ROOT / cfg.section("api").get("cases_dir", "data/cases") / f"case_{game:03d}.zip"
    key = client.fetch_key(game, poll_seconds=0)
    reference = pricebook.reference(before_game=game if prior_only else None,
                                    max_chars=cfg.section("llm").get("pricebook_max_chars", 24000))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        case = load_case(decrypt(zip_path, key, Path(tmp)), f"game{game}")
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            out = list(pool.map(lambda s: _one(case, cfg, s, game, reference,
                                               sh.get("timeout_seconds", 120)), specs))
    for r in out:
        log(f"[shadow] game {game} {r['model']}@{r['effort']}: "
            + (f"{len(r['items'])} items in {r['secs']}s" if r["ok"] else f"FAILED {r['error'][:120]}"))
    return out


def _live_quotes() -> dict[int, dict[int, tuple[float, float]]]:
    out: dict[int, dict[int, tuple[float, float]]] = {}
    if RUN_LOG.exists():
        for line in RUN_LOG.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("ok") and not r.get("dry_run") and r.get("payload"):
                out[int(r["game_id"])] = {int(p["index"]): (p["charge_price"], p["acceptance_limit"])
                                          for p in r["payload"]}
    return out


def report(window: float = 45.0) -> str:
    """Score every shadow model and the live submissions on the games both have. Markdown."""
    from .replay import game_net, load_summary

    summary = load_summary()
    runs: dict[str, dict[int, dict]] = {}
    for f in sorted(DIR.glob("game*__*.json")):
        r = json.loads(f.read_text())
        runs.setdefault(f"{r['model']}@{r['effort']}", {})[int(r["game"])] = r
    live = _live_quotes()
    lines = ["# Shadow model comparison", "",
             "Shadow models see the same case, prompt and price book as the live model; their",
             "quotes are never submitted. Net = replay of the quotes against the real opponents",
             "(`c2f/replay.py`), t inside the revealed bounds; two columns for the two assumptions",
             "about t where only a lower bound is known (1.15x and 1.4x that bound).",
             f"In time = answered within {window:.0f} s. Compare rows only on the same games.", ""]
    for name, by_game in sorted(runs.items()):
        games = sorted(g for g, r in by_game.items() if str(g) in summary)
        if not games:
            lines += [f"## {name}", "", "no finished games with public transactions yet", ""]
            continue
        ok = [g for g in games if by_game[g]["ok"]]
        q = {g: {int(i): tuple(v) for i, v in by_game[g]["quotes"].items()} for g in ok}

        def tot(qs, gs, up):
            return sum(game_net(qs.get(g, {}), g, summary, up) for g in gs)

        secs = [by_game[g]["secs"] for g in ok]
        both = [g for g in ok if g in live]
        lines += [f"## {name}", "",
                  f"- games answered: {len(ok)} of {len(games)} attempted"
                  + (f" (failed: {[g for g in games if g not in ok]})" if len(ok) < len(games) else ""),
                  f"- latency: median {sorted(secs)[len(secs) // 2] if secs else '-'} s, "
                  f"max {max(secs) if secs else '-'} s, in time {sum(s <= window for s in secs)}/{len(ok)}",
                  f"- net on the answered games: {tot(q, ok, 1.15):,.0f} / {tot(q, ok, 1.4):,.0f}",
                  f"- on the {len(both)} of those we also played live: shadow {tot(q, both, 1.15):,.0f} / "
                  f"{tot(q, both, 1.4):,.0f} vs live {tot(live, both, 1.15):,.0f} / {tot(live, both, 1.4):,.0f}",
                  ""]
        tin = sum(by_game[g].get("input_tokens", 0) for g in ok)
        tout = sum(by_game[g].get("output_tokens", 0) for g in ok)
        if tin:
            lines[-1:] = [f"- tokens: {tin:,} in, {tout:,} out over {len(ok)} games", ""]
        served = {by_game[g].get("served_by") for g in ok} - {None, name.split("@")[0]}
        if served:
            lines[-1:] = [f"- note: some answers were served by a fallback model: {sorted(served)}", ""]
    if not runs:
        lines.append("No shadow runs yet (results/shadow/ is empty).")
    text = "\n".join(lines) + "\n"
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "REPORT.md").write_text(text)
    return text
