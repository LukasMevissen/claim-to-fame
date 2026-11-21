"""CLI.

  python -m c2f version
  python -m c2f price --case-dir data/cases/case0
  python -m c2f eval  --case-id case0 --case-dir data/cases/case0     # QC vs labels
  python -m c2f eval  --case-id case0 --replay                         # no LLM, cached extraction (CI)
  python -m c2f live  --case-id 7 --zip data/cases/case7.zip [--dry-run]
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
from .case_io import decrypt, load_case
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
    if args.replay:
        model = cfg.section("llm")["model"]
        items = load_cached(args.case_id, model)
        if items is None:
            print(f"no cached extraction for {args.case_id} / {model}; run eval once with --case-dir")
            return 2
        t0 = time.monotonic()
        priced = price_items(items, cfg)
        timing = {"model": model, "replay": True, "pricing_s": round(time.monotonic() - t0, 2)}
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


def cmd_live(args, cfg) -> int:
    from .api import QuantCoClient

    st = stamp(cfg)
    if (st.git_dirty or not st.git_tag) and not args.allow_untagged:
        print(f"refusing: running {st.label} (dirty or untagged). Release first or pass --allow-untagged.")
        return 2
    t0 = time.monotonic()
    client = QuantCoClient(cfg)
    key = client.fetch_key(args.case_id)
    with tempfile.TemporaryDirectory() as tmp:
        root = decrypt(Path(args.zip), key, Path(tmp))
        case = load_case(root, str(args.case_id))
        priced, timing = price_case(case, cfg, use_cache=False)
        payload = submission_payload(priced)
        resp = None if args.dry_run else client.submit(args.case_id, payload)
    timing["wall_s"] = round(time.monotonic() - t0, 2)
    print(_table(priced))
    print(json.dumps(timing))
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "case_id": args.case_id,
                            "dry_run": args.dry_run, "stamp": st.as_dict(), "timing": timing,
                            "payload": payload, "response": resp}, ensure_ascii=False) + "\n")
    return 0


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

    e = sub.add_parser("eval")
    e.add_argument("--case-id", default="case0")
    e.add_argument("--case-dir")
    e.add_argument("--replay", action="store_true", help="use cached extraction, no LLM call")
    e.add_argument("--no-cache", action="store_true")
    e.add_argument("--fail-on-regression", action="store_true")
    e.add_argument("--tolerance", type=float, default=0.02)

    lv = sub.add_parser("live")
    lv.add_argument("--case-id", required=True)
    lv.add_argument("--zip", required=True)
    lv.add_argument("--dry-run", action="store_true")
    lv.add_argument("--allow-untagged", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "eval" and not args.replay and not args.case_dir:
        ap.error("eval needs --case-dir or --replay")
    cfg = load_config(args.config)
    return {"version": cmd_version, "price": cmd_price, "eval": cmd_eval, "live": cmd_live}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
