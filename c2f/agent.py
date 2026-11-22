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
from .pricing import Quote, acceptance_limit, quote


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


def price_items(items: list[ItemEstimate], cfg: Config,
                charges: dict[int, float] | None = None) -> list[PricedItem]:
    """charges: POS -> charge a decided by the charge agent (c2f/charge_agent.py). For those
    lines only the limit b is computed here; the search for a under the fixed rule is skipped
    (it takes seconds per line against the learned opponent population)."""
    params = cfg.pricing
    out = []
    for k, it in enumerate(items, 1):
        bel = to_belief(it, cfg)
        a = (charges or {}).get(_index(it.item_id, k))
        if a is not None and a > 0 and math.isfinite(a):
            q = Quote(round(a, 2), round(acceptance_limit(bel, params), 2))
        else:
            q = quote(bel, params)
        out.append(PricedItem(it.item_id, it.description, asdict(bel),
                              q.charge_price, q.acceptance_limit))
    return out


def _agent_charges(items: list[ItemEstimate], case: Case, cfg: Config, timing: dict, wait: bool,
                   time_left: float | None) -> dict[int, float]:
    """Opt-in ([charge_agent] mode = "rule" | "llm"): the charge a per line decided by the
    charge agent (c2f/charge_agent.py), as POS -> a for price_items. The limit b is not
    touched. Never raises: on any error it returns {} and the fixed rule prices every line.

    With [charge_agent] shadow = true the agent's charges are only recorded in the timing
    (shadow_charges) and {} is returned, so the fixed rule's charges are submitted; once the
    agent has won takeover_rounds scored rounds in a row (charge_agent.takeover_status) its
    charges are returned and submitted. A shadow run never waits for the judge."""
    sec = cfg.section("charge_agent")
    if sec.get("mode", "off") not in ("rule", "llm") or not items:
        return {}
    try:
        from . import charge_agent
        shadow = bool(sec.get("shadow", False))
        taken = False
        if shadow and sec.get("takeover", True):
            try:
                taken = bool(charge_agent.shared_takeover(int(sec.get("takeover_rounds", 3)))["active"])
            except Exception as e:  # noqa: BLE001  no decision -> stay in shadow
                print(f"[agent] takeover status unavailable: {e!r}")
        wait = wait and (taken or not shadow)
        est = {_index(it.item_id, k): (it.quantity * it.unit_price_gross_eur, it.price_log_sd)
               for k, it in enumerate(items, 1)}
        dec, info = charge_agent.live_charges(case, est, sec, cfg.section("llm").get("model", "gpt-6.2-sol"),
                                              wait=wait, time_left=time_left)
        out = {pos: d.a for pos, d in dec.items() if d.a > 0 and math.isfinite(d.a)}
        if shadow and not taken:
            info["submitted"] = "legacy"
            info["shadow_charges"] = {str(pos): a for pos, a in sorted(out.items())}
            timing["charge_agent"] = info
            return {}
        info["submitted"] = "agent"
        info["takeover"] = shadow
        info["lines_set"] = len(out)
        timing["charge_agent"] = info
        return out
    except Exception as e:  # noqa: BLE001  an extra, never a blocker
        timing["charge_agent"] = {"error": repr(e)[:200]}
        print(f"[agent] charge agent skipped: {e!r}")
        return {}


def _with_item_prior(reference: str, case: Case, llm: dict, errors: list[str]) -> str:
    """Opt-in ([llm] use_item_prior): append the PRICE PRIORS block for the recurring lines of
    this invoice (c2f/item_prior.py) to the reference text. Never raises."""
    if not llm.get("use_item_prior", False):
        return reference
    try:
        from . import item_prior
        block = item_prior.context_for_case(case)
        return (reference + "\n\n" + block) if (reference and block) else (block or reference)
    except Exception as e:  # noqa: BLE001  an extra, never a blocker
        errors.append(f"item_prior: {e!r}"[:200])
        return reference


def _combine_item_prior(items: list[ItemEstimate], case: Case, cfg: Config) -> None:
    """Opt-in ([belief] item_prior = "fuse" | "replace"): combine the model's estimates with
    the item prior after the assessment (c2f/item_prior.py::combine). Never raises."""
    mode = cfg.section("belief").get("item_prior", "off")
    if mode not in ("fuse", "replace") or not items:
        return
    try:
        from . import item_prior
        item_prior.combine_for_case(items, case, mode)
    except Exception as e:  # noqa: BLE001  an extra, never a blocker
        print(f"[agent] item prior not combined: {e!r}")


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
    try:                                   # mode "llm": the judge runs alongside the assessment
        from . import charge_agent
        charge_agent.begin(case, cfg.section("charge_agent"), cfg.section("llm").get("model", "gpt-6.2-sol"))
    except Exception as e:  # noqa: BLE001
        errors.append(f"charge_agent: {e!r}"[:200])
    reference = ""
    if llm.get("use_pricebook", True):
        try:
            from . import pricebook
            reference = pricebook.reference(max_chars=llm.get("pricebook_max_chars", 24000))
        except Exception as e:  # noqa: BLE001  the price book is an extra, never a blocker
            errors.append(f"pricebook: {e!r}"[:200])
    reference = _with_item_prior(reference, case, llm, errors)
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
    _combine_item_prior(items, case, cfg)
    agent_info: dict = {}
    # The quick safety-net stage never waits for the judge; the full stage waits at most
    # [charge_agent].judge_wait_seconds after the judge started and never into the last 6 s.
    charges = _agent_charges(items, case, cfg, agent_info, wait=with_image,
                             time_left=deadline - (time.monotonic() - t0))
    priced = price_items(items, cfg, charges)
    t2 = time.monotonic()
    timing = {"model": used, "with_image": with_image and case.image is not None,
              "extract_s": round(t1 - t0, 2),
              "pricing_s": round(t2 - t1, 2), "total_s": round(t2 - t0, 2), **agent_info}
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
        from . import charge_agent
        charge_agent.begin(case, cfg.section("charge_agent"), cfg.section("llm").get("model", "gpt-6.2-sol"))
    except Exception as e:  # noqa: BLE001
        print(f"[agent] charge agent not started: {e!r}")
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
        reference = _with_item_prior(reference, case, cfg.section("llm"), [])
        # Large invoices at medium effort came close to the deadline (19 lines: 38 s; 22 lines:
        # 31 s against 19 s at low), so from [llm].low_effort_from_lines on the effort drops to low.
        from .pricebook import invoice_lines
        n_lines = len(invoice_lines(case.invoice_text))
        if effort and n_lines >= cfg.section("llm").get("low_effort_from_lines", 10**6):
            effort = "low"
        timing["effort"], timing["n_lines"] = effort, n_lines
        items, meta = extract_claude(case, model, max(budget - 4, 5), reference, effort or None)
        timing.update(meta)
    except Exception as e:  # noqa: BLE001
        timing["model_errors"] = [f"{spec}: {e!r}"[:300]]
        print(f"[agent] backup {spec} failed: {e!r}")
    _combine_item_prior(items, case, cfg)
    charges = _agent_charges(items, case, cfg, timing, wait=True,
                             time_left=budget - (time.monotonic() - t0))
    priced = price_items(items, cfg, charges)
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
