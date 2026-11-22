"""Case memory: a statistical layer on top of the model's assessment (approach B of the
overhaul, docs/research/overhaul.md). Runs in milliseconds, no model call.

The oracle test says where the money is: with the model's cover decisions but the true price
on every covered line, the replayed net of games 20-46 rises from 119k to 416k. Cover
decisions are not the problem, the price per line is. And the price is not a matter of level
(a uniform +10 % on every estimate loses 57k): it has to be right line by line.

The one source that is right line by line is the invoice line itself when it comes back.
Cases are drawn from a limited set of templates, and the same line of the same case carries
the same threshold again (14 pairs with both thresholds tightly bracketed: median difference
4 %). What changes between two draws of a case is which lines are covered.

So for every invoice line the layer looks up precedents in three classes
    case      the same line (text, quantity, unit) in an earlier game with largely the same invoice
    wording   the same line type under the same policy wording
    type      the same line type anywhere
and learns, on all earlier games, how often a precedent of each class named the later
threshold (share of precedent/line pairs whose revealed ranges are compatible). A class
that has earned trust replaces the model's price; the model keeps the cover decision.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import item_prior as ip

CLASSES = ("case", "wording", "type")


@dataclass(frozen=True)
class Options:
    case_sim: float = 0.5         # share of common line types for "largely the same invoice"
    min_trust: float = 0.6        # a class replaces the model's price from this hit rate on
    min_pairs: int = 8            # ... measured on at least this many earlier pairs
    tol: float = 1.15             # two ranges are compatible if they overlap within this factor
    gap: float = 1.04             # threshold = gap x highest proven-fair charge when open above
    sigma_bracket: float = 0.06   # belief spread for a precedent bracketed on both sides
    sigma_lower: float = 0.10     # ... and for one known from below only
    classes: tuple[str, ...] = ("case",)
    agree: float = 0.0            # > 0: replace only if the model is within e^agree of the precedent
    oracle: str = ""              # diagnostics only: use the true threshold on lines of this class


def options(name: str) -> Options:
    """'mem', 'mem:case+wording', 'mem:case+wording+type', 'mem:case@0.7' (agreement window)."""
    o = {}
    _, _, spec = name.partition(":")
    if spec:
        head, _, agree = spec.partition("@")
        if head.startswith("oracle="):
            o["oracle"] = head[7:]
            o["classes"] = (head[7:],)
        elif head:
            o["classes"] = tuple(head.split("+"))
        if agree:
            o["agree"] = float(agree)
    return Options(**o)


@dataclass(frozen=True)
class Precedent:
    cls: str
    game: int
    kind: str          # bracket | lower | upper | zero
    lo: float          # line total for the current quantity; 0 if none
    hi: float          # inf if none


def _compatible(lo1: float, hi1: float, lo2: float, hi2: float, tol: float) -> bool:
    return lo1 <= hi2 * tol and lo2 <= hi1 * tol


class Memory:
    def __init__(self, obs: list[ip.Obs], opt: Options = Options()):
        self.opt = opt
        self.by_key: dict[str, list[ip.Obs]] = defaultdict(list)
        self.keys_of: dict[int, set[str]] = defaultdict(set)
        for o in obs:
            self.by_key[o.key].append(o)
            self.keys_of[o.game].add(o.key)
        self.trust = self._learn_trust()

    @classmethod
    def build(cls, summary: dict | None = None, casebook: str | Path = ip.CASEBOOK,
              before_game: int | None = None, opt: Options = Options()) -> "Memory":
        return cls(ip.observations(summary, casebook, before_game=before_game), opt)

    # ---- classes -------------------------------------------------------------------
    def _sim(self, keys: set[str], game: int) -> float:
        other = self.keys_of.get(game, set())
        return len(keys & other) / len(keys | other) if keys and other else 0.0

    def _cls(self, o: ip.Obs, qty: float, unit: str, policy: str, sim: float) -> str:
        if sim >= self.opt.case_sim and o.unit == unit and abs(o.qty - qty) <= 0.01 * max(qty, 1.0):
            return "case"
        if policy and o.policy == policy and o.unit == unit:
            return "wording"
        return "type" if o.unit == unit else ""

    def precedents(self, lines: dict, policy: str, before_game: int | None = None) -> dict[int, list[Precedent]]:
        """Invoice position -> precedents (best class first, newest first within a class)."""
        parsed = {}
        for pos, text in lines.items():
            desc, qty, unit = ip.parse_line(text)
            key, _ = ip.line_key(desc, unit)
            parsed[int(pos)] = (key, qty, unit)
        keys = {k for k, _, _ in parsed.values() if k}
        out: dict[int, list[Precedent]] = {}
        for pos, (key, qty, unit) in parsed.items():
            rows = []
            for o in self.by_key.get(key, ()):
                if before_game is not None and o.game >= before_game:
                    continue
                c = self._cls(o, qty, unit, policy, self._sim(keys, o.game))
                if c:
                    rows.append(Precedent(c, o.game, o.kind, o.lo * qty, o.hi * qty))
            rows.sort(key=lambda p: (CLASSES.index(p.cls), -p.game))
            out[pos] = rows
        return out

    # ---- how often did a class name the later threshold? ------------------------------
    def _learn_trust(self) -> dict[str, tuple[float, int]]:
        """class -> (share of earlier precedent/line pairs with compatible positive ranges, pairs).
        Only pairs where both lines were paid count: cover is decided by the model."""
        hit: dict[str, int] = defaultdict(int)
        n: dict[str, int] = defaultdict(int)
        for key, obs in self.by_key.items():
            for later in obs:
                if later.kind not in ("bracket", "lower"):
                    continue
                keys = self.keys_of[later.game]
                best: dict[str, ip.Obs] = {}
                for o in obs:
                    if o.game >= later.game or o.kind not in ("bracket", "lower"):
                        continue
                    c = self._cls(o, later.qty, later.unit, later.policy, self._sim(keys, o.game))
                    if c and (c not in best or o.game > best[c].game):
                        best[c] = o                      # the newest precedent of each class
                for c, o in best.items():
                    # informative pairs only: at least one side must be able to contradict
                    if o.kind == "lower" and later.kind == "lower":
                        continue
                    n[c] += 1
                    hit[c] += _compatible(o.lo, o.hi, later.lo, later.hi, self.opt.tol)
        return {c: ((hit[c] + 1) / (n[c] + 2), n[c]) for c in CLASSES}

    def trusted(self, cls: str) -> bool:
        share, pairs = self.trust.get(cls, (0.0, 0))
        return cls in self.opt.classes and pairs >= self.opt.min_pairs and share >= self.opt.min_trust

    # ---- the value a precedent set names -----------------------------------------------
    def value(self, precs: list[Precedent]) -> tuple[float, float, str] | None:
        """(threshold, belief sigma, class) from the best trusted class, or None."""
        for cls in CLASSES:
            if not self.trusted(cls):
                continue
            rows = [p for p in precs if p.cls == cls and p.kind in ("bracket", "lower")]
            if not rows:
                continue
            newest = rows[0]
            lo, hi = newest.lo, newest.hi
            for p in rows[1:]:                      # older precedents only tighten a compatible range
                if _compatible(lo, hi, p.lo, p.hi, 1.0):
                    lo, hi = max(lo, p.lo), min(hi, p.hi)
            if hi < math.inf and hi >= lo:
                return math.sqrt(lo * hi), max(self.opt.sigma_bracket, math.log(hi / lo) / 3.0), cls
            return lo * self.opt.gap, self.opt.sigma_lower, cls
        return None


def apply(items: list, digest: dict, mem: Memory, opt: Options | None = None,
          truth: dict | None = None) -> list[dict]:
    """Replace the model's price by the precedent's on lines with a trusted precedent.
    items: ItemEstimate list (changed in place). digest: {"lines": {pos: text}, "policy": title}.
    Returns one record per changed line. Lines the model believes not covered keep their price
    as well: the pricing rule charges them as if covered (c2f/pricing.py)."""
    from .agent import _index
    if opt is not None:
        mem.opt = opt
        mem.trust = mem._learn_trust()
    precs = mem.precedents(digest.get("lines", {}), digest.get("policy", ""))
    log = []
    for k, it in enumerate(items, 1):
        pos = _index(it.item_id, k)
        v = mem.value(precs.get(pos, []))
        est = it.quantity * it.unit_price_gross_eur
        if v is None or est <= 0:
            continue
        t, sigma, cls = v
        if mem.opt.oracle:
            if truth is None or truth.get(pos) is None or truth[pos] <= 0:
                continue
            t, sigma = truth[pos], 0.05
        elif mem.opt.agree > 0 and abs(math.log(est / t)) > mem.opt.agree:
            continue
        log.append({"pos": pos, "class": cls, "model": round(est, 2), "memory": round(t, 2)})
        it.unit_price_gross_eur, it.price_log_sd = t / max(it.quantity, 1e-9), sigma
    return log


def apply_for_case(items: list, case, opt: Options | None = None) -> list[dict]:
    """Live entry point (c2f/agent.py). Never raises: on any error the model's answer stands."""
    from . import pricebook
    try:
        mem = Memory.build(opt=opt or Options())
        digest = {"lines": pricebook.invoice_lines(case.invoice_text),
                  "policy": pricebook.policy_title(case.policy)}
        return apply(items, digest, mem)
    except Exception as e:  # noqa: BLE001
        print(f"[memory] skipped: {e!r}")
        return []


def trust_table(mem: Memory) -> str:
    return ", ".join(f"{c}: {s:.0%} of {n}" for c, (s, n) in mem.trust.items())


if __name__ == "__main__":
    m = Memory.build()
    print(trust_table(m))
    print(json.dumps({c: m.trusted(c) for c in CLASSES}))
