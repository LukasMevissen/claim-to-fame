"""Pricing agent: case -> line-item estimates -> beliefs -> (a, b) quotes."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .belief import TBelief
from .case_io import Case
from .config import Config
from .extract import ItemEstimate, extract
from .pricing import Quote, quote


@dataclass
class PricedItem:
    item_id: str
    description: str
    belief: dict
    charge_price: float
    acceptance_limit: float


def to_belief(it: ItemEstimate, cfg: Config) -> TBelief:
    b = cfg.section("belief")
    # The model's stated uncertainty is informative: stated < 0.12 -> actual log error about
    # 0.05 (16 lines), 0.12-0.2 -> 0.23, 0.3-0.4 -> 0.32, 0.4-0.6 -> 0.54. So a very confident
    # statement is trusted down to confident_min_sigma; everything else keeps the min_sigma floor.
    if it.price_log_sd < b.get("confident_below", 0.0):
        sigma = max(it.price_log_sd, b.get("confident_min_sigma", 0.1))
    else:
        sigma = min(max(it.price_log_sd, b.get("min_sigma", 0.15)), b.get("max_sigma", 1.2))
    p_zero = 1.0 - it.covered_prob * it.related_prob
    return TBelief(median=it.quantity * it.unit_price_gross_eur, sigma=sigma, p_zero=p_zero)


def price_items(items: list[ItemEstimate], cfg: Config) -> list[PricedItem]:
    params = cfg.pricing
    out = []
    for it in items:
        bel = to_belief(it, cfg)
        q: Quote = quote(bel, params)
        out.append(PricedItem(it.item_id, it.description, asdict(bel),
                              q.charge_price, q.acceptance_limit))
    return out


def price_case(case: Case, cfg: Config, use_cache: bool = True,
               budget: float | None = None,
               with_image: bool = True) -> tuple[list[PricedItem], dict]:
    """Returns priced items and timing info.

    Tries the primary model, then the fallback model while time allows. Never raises on a
    model failure: it returns no items and lists every error in timing["model_errors"], and
    the caller submits fallback prices for the invoice lines (see submission_payload).
    budget: seconds left in the game window; defaults to [runtime].deadline_seconds.
    """
    rt, llm = cfg.section("runtime"), cfg.section("llm")
    t0 = time.monotonic()
    deadline = rt.get("deadline_seconds", 50) if budget is None else budget
    errors: list[str] = []
    items: list[ItemEstimate] = []
    used = None
    # with_image=False is the fast safety-net stage: quick model, no photo.
    # with_image=True is the full stage: reasoning model, photo if the case has one.
    if with_image:
        models = [llm.get("model"), llm.get("fallback_model")]
    else:
        models = [llm.get("fast_model") or llm.get("model"), llm.get("fallback_model")]
    # No de-duplication: when both entries name the same model, the second attempt is a
    # retry after a failure (game 21: one dropped connection left us with fixed prices).
    models = [m for m in models if m]
    reference = ""
    if llm.get("use_pricebook", True):
        try:
            from . import pricebook
            reference = pricebook.reference(max_chars=llm.get("pricebook_max_chars", 24000))
        except Exception as e:  # noqa: BLE001  the price book is an extra, never a blocker
            errors.append(f"pricebook: {e!r}"[:200])
    for model in models:
        timeout = min(rt.get("llm_timeout_seconds", 30), deadline - (time.monotonic() - t0) - 5)
        if timeout < 5:
            errors.append(f"{model}: skipped, only {timeout:.0f}s left")
            break
        try:
            items = extract(case, model, timeout, llm.get("temperature", 0.0), use_cache,
                            with_image=with_image, reference=reference,
                            effort=llm.get("reasoning_effort"))
            used = model
            break
        except Exception as e:  # noqa: BLE001  try the next model, then fall back to defaults
            errors.append(f"{model}: {e!r}"[:300])
            print(f"[agent] model {model} failed: {e!r}")
    t1 = time.monotonic()
    priced = price_items(items, cfg)
    t2 = time.monotonic()
    timing = {"model": used, "with_image": with_image and case.image is not None,
              "extract_s": round(t1 - t0, 2),
              "pricing_s": round(t2 - t1, 2), "total_s": round(t2 - t0, 2)}
    if errors:
        timing["model_errors"] = errors
    return priced, timing


def price_case_backup(case: Case, cfg: Config, budget: float) -> tuple[list[PricedItem], dict]:
    """Independent assessment by another provider ([llm].backup, "provider:model@effort"),
    run in parallel to the main stages. It is what ends up submitted only if the full stage
    does not answer in time, so one provider's outage, a dropped connection or a key without
    credit does not leave us on fixed prices. Never raises."""
    import os

    spec = cfg.section("llm").get("backup", "")
    t0 = time.monotonic()
    timing: dict = {"model": spec, "backup": True}
    items: list[ItemEstimate] = []
    try:
        if not spec:
            return [], timing
        provider, _, rest = spec.partition(":")
        model, _, effort = rest.partition("@")
        if provider != "anthropic":
            raise ValueError(f"unsupported backup provider {provider!r}")
        from .config import REPO_ROOT
        from .env import load_env
        load_env(REPO_ROOT / ".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        from . import pricebook
        from .anthropic_extract import extract_claude
        reference = pricebook.reference(max_chars=cfg.section("llm").get("pricebook_max_chars", 24000))
        items, meta = extract_claude(case, model, max(budget - 4, 5), reference, effort or None)
        timing.update(meta)
    except Exception as e:  # noqa: BLE001
        timing["model_errors"] = [f"{spec}: {e!r}"[:300]]
        print(f"[agent] backup {spec} failed: {e!r}")
    priced = price_items(items, cfg)
    timing["total_s"] = round(time.monotonic() - t0, 2)
    return priced, timing


def _index(item_id: str, fallback: int) -> int:
    """Invoice POS number (1-based) as the API line_item index."""
    m = re.match(r"\s*(\d+)", str(item_id))
    return int(m.group(1)) if m else fallback


def _money(x: float) -> float:
    return round(x, 2) if math.isfinite(x) and x > 0 else 0.0


def submission_payload(priced: list[PricedItem], positions: Iterable[int] = (),
                       fallback: tuple[float, float] = (0.0, 0.0)) -> list[dict]:
    """API_HANDBOOK.md format: [{index, charge_price, acceptance_limit}], gross totals.

    positions: line numbers read from the invoice. Any of them the model did not price
    (all of them if the model failed) get fallback = (charge, limit), so no line is left
    at the API default of 0 / 0.
    """
    idx = [_index(p.item_id, i + 1) for i, p in enumerate(priced)]
    if len(set(idx)) != len(idx):          # unparseable / duplicate POS -> invoice order
        idx = list(range(1, len(priced) + 1))
    out = [{"index": i, "charge_price": _money(p.charge_price),
            "acceptance_limit": _money(p.acceptance_limit)} for i, p in zip(idx, priced)]
    have = set(idx)
    a, b = fallback
    out += [{"index": i, "charge_price": _money(a), "acceptance_limit": _money(b)}
            for i in dict.fromkeys(positions) if i not in have]
    return sorted(out, key=lambda o: o["index"])
