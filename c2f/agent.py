"""Pricing agent: case -> line-item estimates -> beliefs -> (a, b) quotes."""

from __future__ import annotations

import time
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


def price_case(case: Case, cfg: Config, use_cache: bool = True) -> tuple[list[PricedItem], dict]:
    """Returns priced items and timing info. Falls back to the cheaper model on failure."""
    rt, llm = cfg.section("runtime"), cfg.section("llm")
    t0 = time.monotonic()
    model = llm["model"]
    try:
        items = extract(case, model, rt.get("llm_timeout_seconds", 30),
                        llm.get("temperature", 0.0), use_cache)
    except Exception as e:  # noqa: BLE001  one retry with the fallback model
        remaining = rt.get("deadline_seconds", 50) - (time.monotonic() - t0) - 5
        if remaining < 5 or not llm.get("fallback_model"):
            raise
        model = llm["fallback_model"]
        items = extract(case, model, remaining, llm.get("temperature", 0.0), use_cache)
        print(f"[agent] primary model failed ({e!r}); used fallback {model}")
    t1 = time.monotonic()
    priced = price_items(items, cfg)
    t2 = time.monotonic()
    return priced, {"model": model, "extract_s": round(t1 - t0, 2),
                    "pricing_s": round(t2 - t1, 2), "total_s": round(t2 - t0, 2)}


def submission_payload(priced: list[PricedItem]) -> list[dict]:
    """Generic payload. Adapt field names to API_HANDBOOK.md in api.py, not here."""
    return [{"item_id": p.item_id, "charge_price": p.charge_price,
             "acceptance_limit": p.acceptance_limit} for p in priced]
