"""CLI.

  python -m c2f version
  python -m c2f price --case-dir data/cases/case0
  python -m c2f eval  --game 0                                         # QC vs data/labels/game0.json
  python -m c2f eval  --game 0 --replay                                # no LLM, cached extraction (CI)
  python -m c2f live  --case-id 0 [--dry-run]                          # one game now (0 = test game)
  python -m c2f run   [--dry-run]                                      # unattended, all upcoming games
  python -m c2f market [--no-fetch] [--check]                          # transactions -> ab_table + checks
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from .env import load_env
from .agent import price_case, price_items, submission_payload
from .case_io import decrypt, invoice_positions, load_case
from .config import REPO_ROOT, load_config
from .extract import load_cached
from .versioning import stamp
from . import evaluate as ev

RUN_LOG = REPO_ROOT / "runs" / "submissions.jsonl"


def _table(priced) -> str:
    rows = [f"{'item':>8} {'p_zero':>6} {'median':>10} {'a (charge)':>11} {'b (accept)':>11}  description"]
    for p in priced:
        rows.append(f"{p.item_id:>8} {p.belief['p_zero']:>6.2f} {p.belief['median']:>10.2f} "
                    f"{p.charge_price:>11.2f} {p.acceptance_limit:>11.2f}  {p.description[:50]}")
    return "\n".join(rows)


def cmd_version(args, cfg) -> int:
    print(json.dumps(stamp(cfg).as_dict(), indent=2))
    return 0


def cmd_price(args, cfg) -> int:
    case = load_case(Path(args.case_dir), args.case_id)
    priced, timing = price_case(case, cfg, use_cache=not args.no_cache)
    print(_table(priced))
    print(json.dumps(timing))
    return 0


def cmd_eval(args, cfg) -> int:
    st = stamp(cfg)
    deadline = cfg.section("runtime").get("deadline_seconds", 50)
    if args.case_id is None:
        args.case_id = f"game{args.game}" if args.game is not None else "game0"
    if args.replay:
        model = cfg.section("llm")["model"]
        items = load_cached(args.case_id, model)
        if items is None:
            print(f"no cached extraction for {args.case_id} / {model}; run eval once with --game N")
            return 2
        t0 = time.monotonic()
        priced = price_items(items, cfg)
        timing = {"model": model, "replay": True, "pricing_s": round(time.monotonic() - t0, 2)}
    elif args.game is not None:
        # Decrypt data/cases/case_NNN.zip with the game's key (only works once the game started).
        from .api import QuantCoClient

        zip_path = REPO_ROOT / cfg.section("api").get("cases_dir", "data/cases") / f"case_{args.game:03d}.zip"
        key = QuantCoClient(cfg).fetch_key(args.game, poll_seconds=0)
        with tempfile.TemporaryDirectory() as tmp:
            case = load_case(decrypt(zip_path, key, Path(tmp)), args.case_id)
            priced, timing = price_case(case, cfg, use_cache=not args.no_cache)
    else:
        case = load_case(Path(args.case_dir), args.case_id)
        priced, timing = price_case(case, cfg, use_cache=not args.no_cache)

    labels = ev.load_labels(args.case_id)
    errs = ev.gates(priced, timing, deadline, labels)
    metrics = ev.simulate(priced, labels) if labels else None
    out = ev.write_result(args.case_id, st.as_dict(), timing, errs, metrics, priced)

    print(_table(priced))
    print(f"\n[{st.label}] timing {timing}")
    for e in errs:
        print(f"GATE FAIL: {e}")
    if metrics is None:
        print(f"no labels at data/labels/{args.case_id}.json -> only gates checked")
    else:
        print("metrics:", json.dumps(metrics))
        best = ev.previous_best(args.case_id, out)
        if best:
            prev = best["metrics"]["net_payoff"]
            delta = metrics["net_payoff"] - prev
            print(f"vs best so far ({best['file']}): {prev:.2f} -> {metrics['net_payoff']:.2f} ({delta:+.2f})")
            if args.fail_on_regression and delta < -args.tolerance * abs(prev):
                print("REGRESSION beyond tolerance")
                return 1
    print(f"saved {out.relative_to(REPO_ROOT)}")
    return 1 if errs else 0


def _log(rec: dict) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}, ensure_ascii=False) + "\n")


def _release_ok(st, args) -> bool:
    if (st.git_dirty or not st.git_tag) and not args.allow_untagged:
        print(f"refusing: running {st.label} (dirty or untagged). Release first or pass --allow-untagged.")
        return False
    return True


def play_game(client, cfg, st, game_id: int, start_time: float | None, dry_run: bool,
              zip_path: Path | None = None) -> bool:
    """Wait for the key, decrypt, then submit in stages (last write wins). Never raises.

    stage 0  fixed fallback prices for every invoice line, right after decryption
    stage 1  model prices WITHOUT the photo (faster)          } run in parallel; each is
    stage 2  model prices WITH the photo, if the case has one } submitted when it finishes
    A stage is only submitted if it priced something and no higher stage was submitted yet,
    so a slow text-only answer never overwrites the photo answer.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    zip_path = zip_path or REPO_ROOT / cfg.section("api").get("cases_dir", "data/cases") / f"case_{game_id:03d}.zip"
    rec = {"game_id": game_id, "dry_run": dry_run, "stamp": st.as_dict(), "zip": str(zip_path)}
    t0 = None
    try:
        if not zip_path.exists():
            raise FileNotFoundError(f"missing {zip_path}")
        key = client.fetch_key(game_id, start_time)
        t0 = time.monotonic()
        # The 60 s window runs from the game start, not from when the key arrived.
        start = start_time if start_time is not None else time.time()
        deadline = cfg.section("runtime").get("deadline_seconds", 50)
        fb = cfg.section("fallback")
        fallback = (fb.get("charge_price", 0.0), fb.get("acceptance_limit", 0.0))
        stages: list[dict] = []
        best = {"rank": -1, "payload": [], "priced": [], "timing": {}}

        def since() -> float:
            return round(time.time() - start, 2)

        def push(rank: int, name: str, priced, timing: dict | None, payload: list[dict]) -> None:
            info = {"stage": name, "since_start_s": since(), "n_priced": len(priced)}
            if timing and timing.get("model_errors"):
                info["errors"] = timing["model_errors"]
            if not payload or rank <= best["rank"]:
                info["submitted"] = False
            else:
                try:
                    if not dry_run:
                        client.submit(game_id, payload)
                    info["submitted"] = True
                    best.update(rank=rank, payload=payload, priced=priced, timing=timing or {})
                except Exception as e:  # noqa: BLE001
                    info["submitted"], info["error"] = False, repr(e)[:200]
            stages.append(info)
            print(f"[game {game_id}] stage {name}: {info}")

        tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        try:
            root = decrypt(zip_path, key, Path(tmpdir.name))
            case = load_case(root, f"game{game_id}")
            positions = invoice_positions(case.invoice_text)
            try:                       # digest for the price book (c2f/pricebook.py)
                from .pricebook import save_digest
                save_digest(game_id, case)
            except Exception as e:  # noqa: BLE001
                print(f"[game {game_id}] digest not saved: {e!r}")
            push(0, "fixed", [], None, submission_payload([], positions, fallback))

            budget = deadline - (time.time() - start)
            # fast = quick model without photo (safety net), full = reasoning model (+ photo)
            variants = [(1, "fast", False), (2, "full", True)]
            pool = ThreadPoolExecutor(max_workers=len(variants))
            futs = {pool.submit(price_case, case, cfg, False, budget, with_image): (rank, name)
                    for rank, name, with_image in variants}
            pending = set(futs)
            while pending:
                left = deadline - (time.time() - start)
                if left <= 0:
                    break
                done, pending = wait(pending, timeout=left, return_when=FIRST_COMPLETED)
                for f in done:
                    rank, name = futs[f]
                    try:
                        priced, timing = f.result()
                    except Exception as e:  # noqa: BLE001
                        priced, timing = [], {"model_errors": [repr(e)[:300]]}
                    push(rank, name, priced, timing,
                         submission_payload(priced, positions, fallback) if priced else [])
            for f in pending:
                stages.append({"stage": futs[f][1], "since_start_s": since(), "submitted": False,
                               "error": "missed deadline"})
                print(f"[game {game_id}] stage {futs[f][1]}: missed deadline")
            pool.shutdown(wait=False, cancel_futures=True)
        finally:
            tmpdir.cleanup()

        if best["rank"] < 0:
            raise RuntimeError(f"nothing submitted; stages: {stages}")
        timing = {**best["timing"], "n_invoice_lines": len(positions),
                  "final_stage": stages and [s["stage"] for s in stages if s.get("submitted")][-1],
                  "stages": stages, "since_start_s": since(),
                  "wall_s": round(time.monotonic() - t0, 2)}
        print(_table(best["priced"]))
        print(f"[game {game_id}] {'DRY RUN ' if dry_run else ''}final={timing['final_stage']} "
              f"{len(best['payload'])} items in {timing['since_start_s']}s")
        _log({**rec, "ok": True, "timing": timing, "payload": best["payload"],
              "n_accepted": None if dry_run else len(best["payload"])})
        return True
    except Exception as e:  # noqa: BLE001  the scheduler must survive any single game
        wall = round(time.monotonic() - t0, 2) if t0 else None
        print(f"[game {game_id}] FAILED after {wall}s: {e!r}")
        _log({**rec, "ok": False, "error": repr(e)[:500], "wall_s": wall})
        return False


def cmd_live(args, cfg) -> int:
    from .api import QuantCoClient

    st = stamp(cfg)
    if not _release_ok(st, args):
        return 2
    ok = play_game(QuantCoClient(cfg), cfg, st, int(args.case_id), None, args.dry_run,
                   Path(args.zip) if args.zip else None)
    return 0 if ok else 1


def cmd_market(args, cfg) -> int:
    """Sync public transactions of completed games, rebuild results/ab_table.csv, run the
    payoff consistency checks (results/ab_checks.csv) and refit the opponent belief.
    With --check: exit 1 if any check is an error."""
    import statistics

    from . import market

    rep = market.refresh(fetch=not args.no_fetch, refit=not args.no_refit, force=True,
                         pause=args.pause)
    me = args.team
    if args.items:
        summary = market.write_summary(market.load_games())
        print(f"{'game':>4} {'pos':>3} {'t_low':>9} {'t_high':>9} {'others med a':>12} {'our a':>9} {'our b range':>21}")
        for g, items in summary.items():
            for idx, it in items.items():
                others = [v for k, v in it["charges"].items() if k != me]
                bb = it["b_bounds"].get(me)
                print(f"{g:>4} {idx:>3} {it['t_low']:>9.2f} {str(it['t_high'] or '-'):>9} "
                      f"{(statistics.median(others) if others else 0):>12.2f} "
                      f"{str(it['charges'].get(me, '-')):>9} {str(bb or '-'):>21}")
    if args.check:
        from .consistency import summary as check_summary
        found = rep.get("findings")
        if found is None:
            print("checks did not run")
            return 1
        print(check_summary(found))
        return 1 if any(f.severity == "error" for f in found) or rep["errors"] else 0
    return 1 if rep["errors"] else 0


def _market_refresh(args, deadline: float) -> None:
    """After a game: fetch newly completed games, rebuild the table and checks, refit the
    belief. Bounded by `deadline` (no request or refit starts too close to the next game)
    and fully guarded: a failure is logged and retried on the next idle iteration."""
    if args.no_sync:
        return
    try:
        from . import market
        market.refresh(deadline=deadline, log=lambda m: print(f"[market] {m}"))
    except Exception as e:  # noqa: BLE001  never let market data stop the scheduler
        print(f"[market] refresh failed: {e!r}")


def _shadow_async(cfg, client, game_id: int) -> None:
    """After the live submission: let the shadow models assess the same case in a background
    thread (c2f/shadow.py). Never submits; any failure is only logged."""
    if not cfg.section("shadow").get("models"):
        return
    import threading

    def work() -> None:
        try:
            from . import shadow
            shadow.run(game_id, cfg, client)
        except Exception as e:  # noqa: BLE001
            print(f"[shadow] game {game_id} failed: {e!r}")

    threading.Thread(target=work, daemon=True, name=f"shadow-{game_id}").start()


def cmd_shadow(args, cfg) -> int:
    from . import shadow
    for g in args.games:
        shadow.run(g, cfg, specs=args.models or None)
    return 0


def cmd_shadow_report(args, cfg) -> int:
    from . import shadow
    print(shadow.report())
    return 0


def _already_played(dry_run: bool) -> set[int]:
    """Games the log records as submitted (same dry_run mode), so a restart skips them."""
    out: set[int] = set()
    if RUN_LOG.exists():
        for line in RUN_LOG.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("ok") and bool(r.get("dry_run")) == dry_run and r.get("game_id") is not None:
                out.add(int(r["game_id"]))
    return out


def cmd_run(args, cfg) -> int:
    """Unattended loop: for every upcoming game, wait for start_time, then play it."""
    from .api import QuantCoClient, parse_ts

    st = stamp(cfg)
    if not _release_ok(st, args):
        return 2
    client = QuantCoClient(cfg)
    done: set[int] = _already_played(args.dry_run)   # a restart must not replay a game
    print(f"[run] {st.label} dry_run={args.dry_run}; Ctrl-C to stop")
    while True:
        try:
            games = sorted(client.list_games(), key=lambda g: parse_ts(g["start_time"]))
        except Exception as e:  # noqa: BLE001
            print(f"[run] list_games failed: {e!r}; retry in 10s")
            time.sleep(10)
            continue
        now = time.time()
        # Still worth playing if it started less than ~45 s ago (60 s window).
        todo = [g for g in games if g["id"] != 0 and g["id"] not in done
                and parse_ts(g["start_time"]) > now - 45
                and (args.until is None or g["id"] <= args.until)]
        if not todo:
            print("[run] no upcoming games; exiting")
            return 0
        g = todo[0]
        start = parse_ts(g["start_time"])
        if start - now > 120:          # re-check the schedule periodically while idle
            # Idle: pull the games completed since (a game completes some minutes after its
            # start; until then it is not listed and the next idle iteration retries).
            _market_refresh(args, deadline=start - 90)
            now = time.time()
            print(f"[run] next: game {g['id']} at {g['start_time']} (in {int(start - now)}s)")
            time.sleep(max(0.0, min(start - now - 60, 600)))
            continue
        play_game(client, cfg, st, int(g["id"]), start, args.dry_run)
        done.add(int(g["id"]))
        _shadow_async(cfg, client, int(g["id"]))


def main(argv: list[str] | None = None) -> int:
    load_env(REPO_ROOT / ".env")
    ap = argparse.ArgumentParser(prog="c2f")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version")

    p = sub.add_parser("price")
    p.add_argument("--case-dir", required=True)
    p.add_argument("--case-id")
    p.add_argument("--no-cache", action="store_true")

    e = sub.add_parser("eval", help="price a case and score it against data/labels/<case-id>.json")
    e.add_argument("--game", type=int, help="decrypt data/cases/case_NNN.zip via the API key")
    e.add_argument("--case-id", help="labels/cache name; default gameN (same name live runs use)")
    e.add_argument("--case-dir", help="already decrypted case folder instead of --game")
    e.add_argument("--replay", action="store_true", help="use cached extraction, no LLM call")
    e.add_argument("--no-cache", action="store_true")
    e.add_argument("--fail-on-regression", action="store_true")
    e.add_argument("--tolerance", type=float, default=0.02)

    lv = sub.add_parser("live", help="play one game now (game 0 = permanent test game)")
    lv.add_argument("--case-id", required=True, help="game id; zip defaults to data/cases/case_XXX.zip")
    lv.add_argument("--zip")
    lv.add_argument("--dry-run", action="store_true")
    lv.add_argument("--allow-untagged", action="store_true")

    mk = sub.add_parser("market", help="sync public transactions, rebuild ab_table, check, refit")
    mk.add_argument("--no-fetch", action="store_true", help="only rebuild from data/transactions/")
    mk.add_argument("--no-refit", action="store_true", help="keep results/opponent_belief.json")
    mk.add_argument("--check", action="store_true", help="print payoff checks; exit 1 on errors")
    mk.add_argument("--items", action="store_true", help="print t bounds vs charges per item")
    mk.add_argument("--pause", type=float, default=0.3, help="seconds between API requests")
    mk.add_argument("--team", default="GemischtesHack")

    rn = sub.add_parser("run", help="unattended scheduler over all upcoming games")
    rn.add_argument("--dry-run", action="store_true")
    rn.add_argument("--allow-untagged", action="store_true")
    rn.add_argument("--until", type=int, help="stop after this game id")
    rn.add_argument("--no-sync", action="store_true",
                    help="do not refresh public transactions / ab_table / belief between games")

    sh = sub.add_parser("shadow", help="assess started games with the shadow models (never submits)")
    sh.add_argument("games", type=int, nargs="+")
    sh.add_argument("--models", nargs="*", help='override [shadow].models, e.g. anthropic:claude-fable-5-2@low')
    sub.add_parser("shadow-report", help="score shadow models vs live submissions -> results/shadow/REPORT.md")

    args = ap.parse_args(argv)
    if args.cmd == "eval" and not args.replay and not args.case_dir and args.game is None:
        ap.error("eval needs --game N, --case-dir or --replay")
    cfg = load_config(args.config)
    return {"version": cmd_version, "price": cmd_price, "eval": cmd_eval,
            "live": cmd_live, "run": cmd_run, "market": cmd_market,
            "shadow": cmd_shadow, "shadow-report": cmd_shadow_report}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
