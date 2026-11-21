"""Belief about how the other teams play, learned from the public transactions.

What the leaderboard reveals (per game, line item, issuer, reviewer):
    accepted (a <= b) and the amount paid. From the payoff table:
      accepted, amount m       -> issuer's a = m (ignoring the cap), reviewer's b >= a
      rejected, amount 1.5a    -> a = m / 1.5, a <= t (fair), reviewer's b < a
      rejected, amount 0       -> a > t (fraudulent), reviewer's b < a, a itself hidden
    b is never shown. It is interval censored: reviewer r accepted the known charges up
    to L_r and rejected those from U_r up, so L_r <= b_r < U_r.

Latent variable model (per line item j, per team k)
    z_j in {covered, uncovered}, P(uncovered) = pi
    v_j  = the item's apparent value; t_j = v_j if covered, else t_j = 0
    log a_kj = log v_j + alpha_k + sigma_k * eps        (charge rule)
    log b_kj = log v_j + beta_k  + tau_k   * eta        (acceptance rule)
    uncovered items: team k misses the exclusion with prob m_k (b as above),
                     otherwise it spots it and sets b = 0
    absent in a game: a = b = 0 on every item (detected from the data)

    The payoff signals (fair vs fraudulent rejections) pin t on an absolute scale, which
    breaks the shift invariance between v and the team biases.

Fit: EM. E-step = exact posterior of (z_j, v_j) on a log grid per item. M-step = shrunk
weighted moments for (alpha, sigma) and an interval-censored likelihood for
(beta, tau, m) per team. Older games are down-weighted by `decay` per game, so the
belief follows teams that change their strategy during the tournament.

Use for the best response: `opponent_types()` turns the fitted teams into the
`OpponentType` population that `pricing.charge_price` optimises a against.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ._numerics import erfc, expit, minimize_nm, norm_interval

from .pricing import OpponentType

# ---------------------------------------------------------------- data

@dataclass(frozen=True)
class Tx:
    game: int
    item: int
    issuer: str
    reviewer: str
    accepted: bool
    amount: float


def load_transactions(path: str | Path) -> list[Tx]:
    """All data/transactions/game*.json files (or one file) -> transactions, deduplicated."""
    p = Path(path)
    files = sorted(p.glob("game*.json")) if p.is_dir() else [p]
    seen, out = set(), []
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        game = int(doc["game_id"])
        for r in doc["items"]:
            key = (game, int(r["line_item_index"]), r["issuer"], r["reviewer"])
            if key in seen:
                continue
            seen.add(key)
            out.append(Tx(game, key[1], key[2], key[3], bool(r["accepted"]), float(r["amount"])))
    return out


@dataclass
class _Item:
    game: int
    item: int
    charges: dict[str, float] = field(default_factory=dict)   # issuer -> known a > 0
    hidden: dict[str, float] = field(default_factory=dict)    # issuer -> lower bound of a hidden a
    fair: list[float] = field(default_factory=list)           # a with a <= t
    fraud: list[float] = field(default_factory=list)          # a with a > t
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)  # reviewer -> (L, U)
    prior: tuple[float, float] | None = None                  # (log median, log sd) of t from us


def _absent(txs: list[Tx]) -> set[tuple[int, str]]:
    """(game, team) pairs where the team did not submit: it charged 0 on every item and,
    as reviewer, never accepted a positive charge (b = 0 rejects everything above 0).
    A team that charged 0 but accepted positive charges did submit a b."""
    active = defaultdict(bool)
    for x in txs:
        active[(x.game, x.issuer)] |= x.amount > 0
        active[(x.game, x.reviewer)] |= x.accepted and x.amount > 0
    return {k for k, v in active.items() if not v}


def _items(txs: list[Tx], absent: set[tuple[int, str]]) -> list[_Item]:
    by_item: dict[tuple[int, int], list[Tx]] = defaultdict(list)
    for x in txs:
        by_item[(x.game, x.item)].append(x)
    items = []
    for (g, j), rows in sorted(by_item.items()):
        it = _Item(g, j)
        for x in rows:          # recover each issuer's a
            if x.amount > 0:
                it.charges[x.issuer] = x.amount if x.accepted else x.amount / 1.5
        fair, fraud = {}, {}
        for x in rows:          # truth signals from rejections of known charges (one per issuer)
            a = it.charges.get(x.issuer)
            if x.accepted or a is None:
                continue
            (fair if x.amount > 0 else fraud)[x.issuer] = a
        it.fair, it.fraud = list(fair.values()), list(fraud.values())
        lo: dict[str, float] = defaultdict(float)
        hi: dict[str, float] = defaultdict(lambda: math.inf)
        for x in rows:          # interval for each reviewer's b
            if (g, x.reviewer) in absent:
                continue
            a = it.charges.get(x.issuer)
            if a is None:
                continue
            if x.accepted:
                lo[x.reviewer] = max(lo[x.reviewer], a)
            else:
                hi[x.reviewer] = min(hi[x.reviewer], a)
            lo[x.reviewer] += 0.0   # register reviewers that only rejected
        it.intervals = {r: (lo[r], hi[r]) for r in lo}
        it.charges = {k: a for k, a in it.charges.items() if (g, k) not in absent}
        if any(x.amount > 0 for x in rows):      # t > 0 proven, so a hidden charge is real
            for k in {x.issuer for x in rows} - set(it.charges):
                if (g, k) in absent:
                    continue
                mine = [x for x in rows if x.issuer == k]
                if mine and all(not x.accepted and x.amount == 0 for x in mine):
                    # rejected by everyone as fraudulent: a > t and a > b_r >= L_r
                    it.hidden[k] = max([lo[x.reviewer] for x in mine if x.reviewer in lo] + [0.0])
        if it.charges or it.fair or it.fraud:
            items.append(it)
    return items


# ---------------------------------------------------------------- model

@dataclass
class TeamBelief:
    alpha: float = 0.0     # log-bias of charges vs the apparent value
    sigma: float = 0.3     # log-noise of charges
    beta: float = 0.0      # log-bias of acceptance limits
    tau: float = 0.3       # log-noise of acceptance limits
    miss: float = 0.5      # P(misses an exclusion on an uncovered item)
    absent_rate: float = 0.0
    n_games: float = 0.0
    n_charges: float = 0.0
    n_limits: float = 0.0


@dataclass
class OpponentBelief:
    teams: dict[str, TeamBelief]
    pi_uncovered: float
    games: list[int]
    items: list[dict]      # posterior of t per item: game, item, p_zero, t_median, t_q10, t_q90

    def to_json(self) -> str:
        return json.dumps({"teams": {k: asdict(v) for k, v in self.teams.items()},
                           "pi_uncovered": self.pi_uncovered, "games": self.games,
                           "items": self.items}, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "OpponentBelief":
        d = json.loads(s)
        return cls({k: TeamBelief(**v) for k, v in d["teams"].items()},
                   d["pi_uncovered"], d["games"], d["items"])


_EPS = 0.02      # probability a hard truth signal is wrong (rounding, cap, data noise)
_SOFT = 0.02     # log width of the soft step for truth signals
_LOGMIN = -50.0


def _soft_ge(x: np.ndarray, thr: float) -> np.ndarray:
    """log P(signal 'x >= thr' is consistent), soft and floored."""
    return np.log(_EPS + (1 - 2 * _EPS) * expit((x - thr) / _SOFT))


def _sf(x):
    """Standard normal survival function."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / math.sqrt(2.0))


def _interval_logp(v: np.ndarray, L: float, U: float, beta: float, tau: float) -> np.ndarray:
    """log P(L <= v * exp(beta + tau Z) < U) on the grid v (log scale)."""
    hi = (math.log(U) - v - beta) / tau if U < math.inf else np.full_like(v, np.inf)
    lo = (math.log(L) - v - beta) / tau if L > 0 else np.full_like(v, -np.inf)
    p = norm_interval(lo, hi)
    return np.log(np.maximum(p, 1e-300))


def _item_posterior(it: _Item, teams: dict[str, TeamBelief], pi: float, n_grid: int):
    """Grid and posterior weights w[z, g] (z=0 covered, z=1 uncovered) for one item."""
    anchors = [math.log(a) for a in it.charges.values()] + [math.log(a) for a in it.fair + it.fraud]
    if it.prior:
        anchors.append(it.prior[0])
    c = float(np.median(anchors)) if anchors else 0.0
    v = np.linspace(c - 3.5, c + 3.5, n_grid)
    lp = np.zeros((2, n_grid))
    if it.prior:
        lp += -0.5 * ((v - it.prior[0]) / max(it.prior[1], 0.1)) ** 2
    lp[0] += math.log(1 - pi)
    lp[1] += math.log(pi)
    for k, a in it.charges.items():        # same charge rule whether covered or not
        tb = teams[k]
        lp += -0.5 * ((math.log(a) - v - tb.alpha) / tb.sigma) ** 2 - math.log(tb.sigma)
    for k, low in it.hidden.items():       # a > max(low, t): left-censored charge
        tb = teams[k]
        for z, floor in ((0, v), (1, np.full_like(v, -np.inf))):
            thr = np.maximum(math.log(low) if low > 0 else -np.inf, floor)
            lp[z] += np.log(np.maximum(_sf((thr - v - tb.alpha) / tb.sigma), 1e-300))
    for a in it.fair:                      # a <= t: covered and v >= log a
        lp[0] += _soft_ge(v, math.log(a))
        lp[1] += math.log(_EPS)
    for a in it.fraud:                     # a > t: uncovered, or v < log a
        lp[0] += _soft_ge(-v, -math.log(a))
    for r, (L, U) in it.intervals.items():
        tb = teams[r]
        li = _interval_logp(v, L, U, tb.beta, tb.tau)
        lp[0] += li
        spotted = 0.0 if L == 0 else _EPS     # spotted exclusion -> b = 0 -> accepts nothing positive
        lp[1] += np.log(tb.miss * np.exp(li) + (1 - tb.miss) * spotted + 1e-300)
    lp -= lp.max()
    w = np.exp(lp)
    return v, w / w.sum()


def _summary(it: _Item, v: np.ndarray, w: np.ndarray) -> dict:
    p_zero = float(w[1].sum())
    cov = w[0] / max(w[0].sum(), 1e-300)
    cdf = np.cumsum(cov)
    q = lambda p: float(math.exp(v[min(np.searchsorted(cdf, p), len(v) - 1)]))  # noqa: E731
    return {"game": it.game, "item": it.item, "p_zero": round(p_zero, 4),
            "t_median": round(q(0.5), 2), "t_q10": round(q(0.1), 2), "t_q90": round(q(0.9), 2)}


def fit(txs: list[Tx], decay: float = 0.85, n_iter: int = 40, n_grid: int = 141,
        own_estimates: dict[tuple[int, int], tuple[float, float]] | None = None,
        prior_strength: float = 2.0) -> OpponentBelief:
    """EM fit of every team's behaviour. `own_estimates` maps (game, item) to our own
    (median, log sd) of t, used as an extra prior. Games are weighted decay**age."""
    absent = _absent(txs)
    items = _items(txs, absent)
    for it in items:
        if own_estimates and (it.game, it.item) in own_estimates:
            med, sd = own_estimates[(it.game, it.item)]
            it.prior = (math.log(max(med, 1e-6)), sd)
    games = sorted({x.game for x in txs})
    last = max(games) if games else 0
    gw = {g: decay ** (last - g) for g in games}
    names = sorted({x.issuer for x in txs} | {x.reviewer for x in txs})
    teams = {k: TeamBelief() for k in names}
    for k, tb in teams.items():
        played = [g for g in games if any(x.game == g and (x.issuer == k or x.reviewer == k) for x in txs)]
        tb.n_games = len(played)
        wsum = sum(gw[g] for g in played)
        tb.absent_rate = sum(gw[g] for g in played if (g, k) in absent) / wsum if wsum else 0.0
    pi = 0.2

    post = []
    for _ in range(n_iter):
        post = [(it, *_item_posterior(it, teams, pi, n_grid)) for it in items]
        # pi: share of uncovered items
        pz = np.array([w[1].sum() for _, _, w in post])
        wts = np.array([gw[it.game] for it, _, _ in post])
        pi = float(np.clip((wts @ pz + 1) / (wts.sum() + 5), 0.02, 0.6))
        for k, tb in teams.items():
            _m_step_charge(k, tb, post, gw, prior_strength)
            _m_step_limit(k, tb, post, gw, prior_strength)

    summaries = [_summary(it, v, w) for it, v, w in post]
    return OpponentBelief(teams, pi, games, summaries)


def _m_step_charge(k, tb: TeamBelief, post, gw, s0: float) -> None:
    """Weighted moments of r = log a - log v. Hidden charges (a > threshold) enter
    through the truncated-normal conditional moments (EM for censored data)."""
    sw = swr = swr2 = 0.0
    for it, v, w in post:
        g = gw[it.game]
        a = it.charges.get(k)
        if a is not None:
            r = math.log(a) - v
            marg = w.sum(axis=0)
            sw += g
            swr += g * float(marg @ r)
            swr2 += g * float(marg @ r ** 2)
            continue
        if k not in it.hidden:
            continue
        low = it.hidden[k]
        for z, floor in ((0, v), (1, np.full_like(v, -np.inf))):
            thr = np.maximum(math.log(low) if low > 0 else -np.inf, floor) - v   # r > thr
            c = np.clip((thr - tb.alpha) / tb.sigma, -30, 30)
            lam = np.exp(-0.5 * c * c) / math.sqrt(2 * math.pi) / np.maximum(_sf(c), 1e-300)
            er = tb.alpha + tb.sigma * lam
            var = tb.sigma ** 2 * np.maximum(1 + c * lam - lam ** 2, 1e-6)
            sw += g * float(w[z].sum())
            swr += g * float(w[z] @ er)
            swr2 += g * float(w[z] @ (var + er ** 2))
    tb.n_charges = sw
    if sw == 0:
        return
    alpha = swr / (sw + s0)                       # shrink towards 0
    var = (swr2 - 2 * alpha * swr + alpha ** 2 * sw + s0 * 0.3 ** 2) / (sw + s0)
    tb.alpha, tb.sigma = alpha, float(np.clip(math.sqrt(max(var, 1e-6)), 0.05, 2.0))


def _m_step_limit(k, tb: TeamBelief, post, gw, s0: float) -> None:
    data = [(gw[it.game], v, w, *it.intervals[k]) for it, v, w in post if k in it.intervals]
    tb.n_limits = float(sum(d[0] for d in data))
    if not data:
        return
    G = np.array([d[0] for d in data])[:, None]                  # (n, 1) game weights
    V = np.stack([d[1] for d in data])                           # (n, grid)
    W0 = np.stack([d[2][0] for d in data]) * G
    W1 = np.stack([d[2][1] for d in data]) * G
    logL = np.array([math.log(d[3]) if d[3] > 0 else -np.inf for d in data])[:, None]
    logU = np.array([math.log(d[4]) if d[4] < math.inf else np.inf for d in data])[:, None]
    spotted = np.array([0.0 if d[3] == 0 else _EPS for d in data])[:, None]

    def nll(x):
        beta, tau, miss = x[0], math.exp(x[1]), float(expit(x[2]))
        li = norm_interval((logL - V - beta) / tau, (logU - V - beta) / tau)
        tot = float((W0 * np.log(li + 1e-300)).sum()
                    + (W1 * np.log(miss * li + (1 - miss) * spotted + 1e-300)).sum())
        prior = (-0.5 * s0 * (beta / 0.5) ** 2 - 0.5 * s0 * ((x[1] - math.log(0.3)) / 0.7) ** 2
                 + s0 * (math.log(miss) + math.log(1 - miss)))
        return -(tot + prior)

    x0 = [tb.beta, math.log(tb.tau), math.log(tb.miss / (1 - tb.miss))]
    x = minimize_nm(nll, x0, bounds=[(-3, 3), (math.log(0.03), math.log(3.0)), (-6, 6)])
    tb.beta, tb.tau, tb.miss = float(x[0]), float(math.exp(x[1])), float(expit(x[2]))


# ---------------------------------------------------------------- best response input

def opponent_types(belief: OpponentBelief, us: str, min_limits: float = 3.0,
                   fallback: OpponentType | None = None) -> tuple[OpponentType, ...]:
    """Fitted teams -> opponent population for pricing.charge_price (one entry per team,
    equal weight per team, split into its absent share and its learned acceptance rule).
    Teams with too little evidence on b keep the fallback rule."""
    fallback = fallback or OpponentType(1.0, "noisy", mu=0.0, s=0.35, miss_zero=0.5)
    out = []
    for name, tb in sorted(belief.teams.items()):
        if name == us:
            continue
        if tb.absent_rate > 0:
            out.append(OpponentType(tb.absent_rate, "absent"))
        if tb.absent_rate < 1:
            if tb.n_limits >= min_limits:
                out.append(OpponentType(1 - tb.absent_rate, "noisy", mu=tb.beta, s=tb.tau,
                                        miss_zero=tb.miss))
            else:
                out.append(OpponentType(1 - tb.absent_rate, "noisy", mu=fallback.mu,
                                        s=fallback.s, miss_zero=fallback.miss_zero))
    return tuple(out)


def save_belief(belief: OpponentBelief, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(belief.to_json(), encoding="utf-8")


def load_belief(path: str | Path) -> OpponentBelief:
    return OpponentBelief.from_json(Path(path).read_text(encoding="utf-8"))
