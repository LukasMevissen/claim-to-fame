"""Payoff consistency of the public transactions, and the item-level facts they prove.

Every transaction is issuer i (charge a_i) -> reviewer r (limit b_r) on line item j with
secret threshold t_j and cap c_j >= 4 t_j. The payoff table (docs/CASE.md) allows only

    accepted (a <= b):   a <= t -> amount a           a > t -> amount min(a, c)
    rejected (a >  b):   a <= t -> amount 1.5 a       a > t -> amount 0

so, for one issuer on one item, across all its reviewers:
    all accepted amounts are equal (a, or c if the cap bit) and at most a;
    all positive rejected amounts are equal (1.5 a); then a <= t, the cap cannot bite,
    and every accepted amount equals a;
    a positive and a zero rejection cannot both occur (a <= t and a > t).
Per item:
    t_low = max fair a (wrongful rejection)  <=  t  <  min accepted amount m of a fraud
    issuer (uncapped a = m > t; capped m = c >= 4t, so t <= m/4 < m either way);
    accepted at 0 although a > 0 (it was rejected elsewhere, so a > b >= 0) -> c = 0 -> t = 0;
    a revealed cap gives t <= c/4 (not strict).
Per reviewer r:
    b_low = max a_lo of accepted charges <= b_r < b_high = min a of rejected charges.
    a_lo is exact or a lower bound (an accepted amount m <= a). A rejected charge known only
    through an accepted amount m elsewhere may be capped (a > m). If treating it as
    uncapped empties the bracket, the cap bit: c = m is revealed (and t <= c/4).

check() turns every violation into a Finding and never drops a row. Severity:
    error  contradicts the payoff table or our own submission log
    warn   data incomplete or unexplained (missing pairs, a play we have no log for)
    info   a legitimate inference worth knowing (cap hit revealing c)
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .opponents import Tx

US = "GemischtesHack"


def _tol(*xs: float) -> float:
    """Amounts carry float noise (1.5 * a); prices are cents, so 1e-6 relative is strict."""
    return 1e-6 * max([1.0] + [abs(x) for x in xs])


def _eq(x: float, y: float) -> bool:
    return abs(x - y) <= _tol(x, y)


@dataclass(frozen=True)
class Finding:
    kind: str
    severity: str            # error | warn | info
    game: int
    item: int | None
    team: str | None
    detail: str


@dataclass
class IssuerFacts:
    accepted: list[float] = field(default_factory=list)   # amounts paid on accepted rows
    fair: list[float] = field(default_factory=list)       # positive rejected amounts (1.5 a)
    fraud: int = 0                                         # rejections paid 0 (a > t)

    @property
    def a_exact(self) -> float | None:
        """a from a wrongful rejection (amount / 1.5); exact, and then a <= t (no cap)."""
        return self.fair[0] / 1.5 if self.fair else None

    @property
    def a_lo(self) -> float | None:
        """Largest proven lower bound of a (exact when wrongly rejected)."""
        if self.fair:
            return self.a_exact
        return max(self.accepted) if self.accepted else None


@dataclass
class ItemFacts:
    game: int
    item: int
    issuers: dict[str, IssuerFacts]
    t_low: float = 0.0                 # t >= t_low
    t_high: float = math.inf           # t <= t_high
    t_high_strict: bool = False        # t < t_high (fraud charge) rather than t <= (cap / 4)
    cap: float | None = None           # c revealed by a cap hit
    caps_seen: list[float] = field(default_factory=list)   # all cap values implied
    capped: dict[str, float] = field(default_factory=dict)  # issuer -> c (its a > c)

    def a_hi(self, k: str) -> tuple[float, bool] | None:
        """Upper bound of issuer k's a and whether it is exact. An accepted amount m is
        exact when the cap cannot bite: wrongly rejected, or m < 4 t_low <= c."""
        f = self.issuers.get(k)
        if f is None:
            return None
        if f.fair:
            return f.a_exact, True
        if f.accepted:
            m = max(f.accepted)
            return m, m < 4 * self.t_low - _tol(m)
        return None


def item_facts(game: int, item: int, rows: list[Tx]) -> ItemFacts:
    iss: dict[str, IssuerFacts] = defaultdict(IssuerFacts)
    for x in rows:
        f = iss[x.issuer]
        if x.accepted:
            f.accepted.append(x.amount)
        elif x.amount > 0:
            f.fair.append(x.amount)
        else:
            f.fraud += 1
    it = ItemFacts(game, item, dict(iss))
    it.t_low = max((f.a_exact for f in iss.values() if f.fair), default=0.0)

    def upper(v: float, strict: bool) -> None:
        if v < it.t_high - _tol(v):
            it.t_high, it.t_high_strict = v, strict
        elif _eq(v, it.t_high):
            it.t_high_strict |= strict

    for k, f in iss.items():
        if not f.accepted:
            continue
        m = max(f.accepted)
        if f.fraud and m > 0:                         # a > t, paid min(a, c) = m: t < m
            upper(m, True)
        if min(f.accepted) == 0 and (f.fair or f.fraud):   # a > 0 accepted at 0: c = 0
            it.capped[k] = 0.0
    # Reviewer brackets: a rejected charge whose only known value is an accepted amount
    # m <= b_low must be capped (a > b >= b_low >= m), which reveals c = m.
    for r in {x.reviewer for x in rows}:
        lows = [iss[x.issuer].a_lo for x in rows
                if x.reviewer == r and x.accepted and iss[x.issuer].a_lo is not None]
        b_low = max(lows, default=0.0)
        for x in rows:
            if x.reviewer != r or x.accepted:
                continue
            hi = it.a_hi(x.issuer)
            if hi and not hi[1] and hi[0] <= b_low + _tol(b_low) and lows:
                it.capped[x.issuer] = hi[0]
    it.caps_seen = sorted({round(c, 6) for c in it.capped.values()})
    if it.caps_seen:
        it.cap = it.caps_seen[0]
        upper(it.cap / 4, False)                      # c >= 4t
    return it


def our_quotes(path: str | Path) -> dict[tuple[int, int], tuple[float, float]]:
    """(game, line item) -> (a, b) we actually submitted, from runs/submissions.jsonl:
    the last successful, non-dry-run record with a payload per game (later submissions
    overwrite earlier ones on the server). Payload "index" is the 1-based line item."""
    p = Path(path)
    if not p.exists():
        return {}
    last: dict[int, list[dict]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("game_id") is None or not r.get("payload") or r.get("dry_run") or not r.get("ok", True):
            continue
        last[int(r["game_id"])] = r["payload"]
    out = {}
    for g, payload in last.items():
        for pos, q in enumerate(payload, start=1):
            try:
                j = int(q.get("index", q.get("item_id", pos)))
            except (TypeError, ValueError):
                j = pos
            out[(g, j)] = (float(q["charge_price"]), float(q["acceptance_limit"]))
    return out


def check(txs: list[Tx], quotes: dict[tuple[int, int], tuple[float, float]] | None = None,
          us: str = US) -> list[Finding]:
    """All payoff-table violations in `txs` (load with dedup=False to also see repeated
    rows) and mismatches with our own quotes (`our_quotes`)."""
    out: list[Finding] = []

    def add(kind, sev, g, j, team, detail):
        out.append(Finding(kind, sev, g, j, team, detail))

    # ---- structure: one row per ordered pair and item, items 1..K
    by_game: dict[int, list[Tx]] = defaultdict(list)
    for x in txs:
        by_game[x.game].append(x)
    clean: list[Tx] = []
    for g, rows in sorted(by_game.items()):
        seen: dict[tuple, Tx] = {}
        for x in rows:
            k = (x.item, x.issuer, x.reviewer)
            if x.issuer == x.reviewer:
                add("self_transaction", "error", g, x.item, x.issuer, "issuer == reviewer")
            if x.amount < 0 or not math.isfinite(x.amount):
                add("bad_amount", "error", g, x.item, x.issuer, f"amount {x.amount}")
            if k in seen:
                if (seen[k].accepted, seen[k].amount) != (x.accepted, x.amount):
                    add("duplicate_conflict", "error", g, x.item, x.issuer,
                        f"reviewer {x.reviewer}: {seen[k]} vs {x}")
                continue
            seen[k] = x
            clean.append(x)
        teams = {x.issuer for x in rows} | {x.reviewer for x in rows}
        n = len(teams)
        items = sorted({x.item for x in rows})
        if items != list(range(1, len(items) + 1)):
            add("item_gap", "warn", g, None, None, f"line items {items}")
        per_item = Counter(k[0] for k in seen)
        for j in items:
            if per_item[j] != n * (n - 1):
                add("incomplete_matrix", "warn", g, j, None,
                    f"{per_item[j]} rows, expected {n * (n - 1)} for {n} teams")

    # ---- payoff table, per item
    by_item: dict[tuple[int, int], list[Tx]] = defaultdict(list)
    for x in clean:
        by_item[(x.game, x.item)].append(x)
    facts: dict[tuple[int, int], ItemFacts] = {}
    for (g, j), rows in sorted(by_item.items()):
        it = facts[(g, j)] = item_facts(g, j, rows)
        for k, f in sorted(it.issuers.items()):
            if f.accepted and any(not _eq(m, f.accepted[0]) for m in f.accepted):
                add("accepted_amounts_differ", "error", g, j, k,
                    f"accepted amounts {sorted(set(round(m, 2) for m in f.accepted))}")
            if f.fair and any(not _eq(m, f.fair[0]) for m in f.fair):
                add("reject_amounts_differ", "error", g, j, k,
                    f"wrongful-reject amounts {sorted(set(round(m, 2) for m in f.fair))}")
            if f.fair and f.fraud:
                add("fair_and_fraud", "error", g, j, k,
                    f"{len(f.fair)} rejections paid 1.5a and {f.fraud} paid 0")
            if f.fair and any(not _eq(m, f.a_exact) for m in f.accepted):
                add("accept_vs_reject_a", "error", g, j, k,
                    f"a = {f.a_exact:.2f} from 1.5a, but accepted at {max(f.accepted):.2f} "
                    "(a <= t, so no cap)")
        if it.t_low > 0 and it.t_high < math.inf and (
                it.t_low > it.t_high + _tol(it.t_high)
                or (it.t_high_strict and it.t_low >= it.t_high - _tol(it.t_high))):
            add("t_bounds_contradict", "error", g, j, None,
                f"fair a {it.t_low:.2f} <= t but t {'<' if it.t_high_strict else '<='} {it.t_high:.2f}")
        if len(it.caps_seen) > 1:
            add("cap_inconsistent", "error", g, j, None, f"implied caps {it.caps_seen}")
        if it.cap is not None and it.t_low > it.cap / 4 + _tol(it.t_low):
            add("cap_below_4t", "error", g, j, None,
                f"c = {it.cap:.2f} but t >= {it.t_low:.2f} needs c >= {4 * it.t_low:.2f}")
        for k, c in sorted(it.capped.items()):
            add("cap_hit", "info", g, j, k, f"a > c = {c:.2f}, so t <= {c / 4:.2f}")
        if it.cap is not None:
            for k, f in sorted(it.issuers.items()):
                if f.accepted and max(f.accepted) > it.cap + _tol(it.cap):
                    add("paid_above_cap", "error", g, j, k,
                        f"accepted at {max(f.accepted):.2f} > c = {it.cap:.2f}")
        for r in sorted({x.reviewer for x in rows}):
            acc = [(it.issuers[x.issuer].a_lo, x.issuer) for x in rows
                   if x.reviewer == r and x.accepted and it.issuers[x.issuer].a_lo is not None]
            rej = [(it.a_hi(x.issuer), x.issuer) for x in rows if x.reviewer == r and not x.accepted]
            rej = [(h[0], i) for h, i in rej if h and h[1]]
            if acc and rej:
                lo, hi = max(acc), min(rej)
                if lo[0] >= hi[0] - _tol(hi[0]):
                    add("b_bounds_contradict", "error", g, j, r,
                        f"accepted a = {lo[0]:.2f} ({lo[1]}) but rejected a = {hi[0]:.2f} ({hi[1]})")

    # ---- our own log vs what the transactions show for us
    if quotes is None:
        return out
    for g, rows in sorted(by_game.items()):
        items = sorted({x.item for x in rows})
        if not any(x.issuer == us or x.reviewer == us for x in rows):
            add("our_team_missing", "warn", g, None, us, "no transactions for our team")
            continue
        logged = sorted(j for (gg, j) in quotes if gg == g)
        if not logged:
            played = [x for x in rows if
                      (x.issuer == us and x.amount > 0) or (x.reviewer == us and x.accepted and x.amount > 0)]
            if played:
                add("our_unlogged_play", "warn", g, None, us,
                    f"{len(played)} rows show a submission, but no live record in the log")
            continue
        if logged != items:
            add("our_item_mismatch", "error", g, None, us, f"log items {logged}, game items {items}")
        for j in items:
            q = quotes.get((g, j))
            it = facts.get((g, j))
            if q is None or it is None:
                continue
            qa, qb = q
            f = it.issuers.get(us, IssuerFacts())
            mine = [x for x in by_item[(g, j)] if x.issuer == us]
            if qa <= 0:
                if any(not x.accepted or x.amount > 0 for x in mine):
                    add("our_a_mismatch", "error", g, j, us, "a = 0 must be accepted and paid 0")
            else:
                for m in f.fair:
                    if not _eq(m / 1.5, qa):
                        add("our_a_mismatch", "error", g, j, us, f"rejected at {m:.2f} = 1.5 x {m / 1.5:.2f}, we sent a = {qa}")
                for m in sorted(set(f.accepted)):
                    if m > qa + _tol(qa):
                        add("our_a_mismatch", "error", g, j, us, f"paid {m:.2f} > our a = {qa}")
                    elif not _eq(m, qa):
                        add("our_a_capped", "info", g, j, us, f"we sent a = {qa}, paid {m:.2f}: c = {m:.2f}")
            for x in by_item[(g, j)]:
                if x.reviewer != us:
                    continue
                lo = it.issuers[x.issuer].a_lo
                hi = it.a_hi(x.issuer)
                if x.accepted and lo is not None and lo > qb + _tol(qb):
                    add("our_b_mismatch", "error", g, j, x.issuer,
                        f"we accepted a >= {lo:.2f} with b = {qb}")
                if not x.accepted and hi and hi[0] <= qb + _tol(qb):
                    if hi[1]:
                        add("our_b_mismatch", "error", g, j, x.issuer,
                            f"we rejected a = {hi[0]:.2f} with b = {qb}")
                    else:
                        add("cap_hit", "info", g, j, x.issuer,
                            f"we rejected it with b = {qb}, paid {hi[0]:.2f} elsewhere: c = {hi[0]:.2f}")
    return out


def summary(findings: list[Finding]) -> str:
    """One line per (severity, kind) with a count and the first example."""
    groups: dict[tuple[str, str], list[Finding]] = defaultdict(list)
    for f in findings:
        groups[(f.severity, f.kind)].append(f)
    order = {"error": 0, "warn": 1, "info": 2}
    lines = []
    for (sev, kind), fs in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 3), kv[0][1])):
        e = fs[0]
        lines.append(f"{sev:5s} {kind:24s} {len(fs):4d}  e.g. game {e.game} item {e.item} "
                     f"{e.team or ''}: {e.detail}")
    return "\n".join(lines) or "no findings"


def write_csv(findings: list[Finding], path: str | Path) -> None:
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["severity", "kind", "game", "item", "team", "detail"])
        for f in findings:
            w.writerow([f.severity, f.kind, f.game, "" if f.item is None else f.item, f.team or "", f.detail])
