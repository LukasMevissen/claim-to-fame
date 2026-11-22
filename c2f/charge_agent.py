"""Charge agent: chooses the charge a for every invoice line from all earlier rounds.

The fixed rule (c2f/pricing.py) charges max(optimiser, 0.87 x estimate) on every line; the
0.87 comes from a replay of games 1-12 and is the same whether the line is new or has been
priced five times before. This agent replaces that share by a decision per line:

1. Belief of the case assessment. log t ~ log m + bias + sd, where bias and sd are fitted
   on every earlier round (how far the model's estimates really were from the revealed
   thresholds, given the uncertainty it stated), not taken from the model's own statement.
2. Evidence of earlier rounds. Every earlier line of the same type is a precedent with what
   its transactions proved (bracket, lower bound). Each precedent gets a relevance weight
   w in [0, 1]: how likely its threshold per unit is the same quantity as the one asked for
   now. A precedent enters as  (1 - w) x "tells nothing" + w x "t within tau of this bound",
   so a tight bracket from a near-identical case moves the belief a lot, a loose lower bound
   from another policy wording hardly at all, and a precedent that contradicts the
   assessment (other indemnity basis) is ignored instead of averaged in.
3. Decision. a maximises E[revenue(a, t) ** gamma] over the resulting distribution of t,
   against the acceptance behaviour of the other teams measured on earlier rounds.
   gamma = 1 is risk neutral, gamma > 1 risk loving (more weight on the outcome in which the
   high charge is still fair).

Two modes decide the relevance weights:

    rule   text similarity of the line, same policy wording, similarity of the claim, age.
           No model call; a whole invoice takes milliseconds.
    llm    a model reads the current case next to all earlier cases with their revealed
           thresholds (the price book) and names, per line, the earlier lines that are
           informative, with a weight and a scale factor. It runs in parallel to the case
           assessment and has a hard time limit; without an answer in time the rule
           weights are used, so it can never delay a submission beyond that limit.

Shadow run ([charge_agent] shadow = true): the agent's charges are recorded next to the
submitted ones and scored after every round; after `takeover_rounds` winning rounds in a
row they are submitted instead (see "shadow run and takeover" at the end of this file).

Config: [charge_agent] in configs/default.toml. Backtest: scripts/charge_agent_backtest.py.
Design and results: docs/research/charge_agent.md.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import item_prior, novelty
from ._numerics import ndtr
from .config import REPO_ROOT

CASEBOOK = REPO_ROOT / "results" / "casebook"
SUMMARY = REPO_ROOT / "results" / "market" / "summary.json"
EXTRACTIONS = REPO_ROOT / "results" / "extractions"
US = "GemischtesHack"

# log(t / m) grid for the belief and log(a / m) grid for the candidates
_X = np.arange(-2.5, 2.5001, 0.01)
_Y = np.arange(-1.0, 0.7001, 0.005)
# log(a / t) grid for the acceptance curve of the other teams
_R = np.arange(-1.5, 1.5001, 0.02)


@dataclass(frozen=True)
class Params:
    gamma: float = 1.0           # utility = revenue ** gamma; 1 = risk neutral, > 1 = risk loving
    tau: float = 0.08            # log-sd of the same line type's t per unit between two cases
    gap_rate: float = 8.0        # lower bound only: ln t - ln lo ~ Exp(rate) (item_prior: same idea)
    min_sim: float = 0.6         # line text similarity below this is no precedent (rule mode)
    decay: float = 0.99          # relevance factor per game of age
    w_max: float = 0.9           # no precedent is trusted more than this
    temper: float = 1.0          # exponent on the precedent evidence (< 1: the assessment saw it too)
    broad: float = 1.0           # log-sd of "tells nothing": where an unrelated t could lie around m
    # calibrated belief of the assessment per class of stated log-sd: (upper edge, bias, sd).
    # Fitted on earlier rounds by calibrate(); these defaults are the fit on games 10-48.
    # The defaults are the errors measured earlier (c2f/agent.py::to_belief) and act as the
    # prior of the fit.
    cal: tuple = ((0.12, 0.0, 0.075), (0.2, 0.0, 0.2), (0.4, 0.0, 0.4), (math.inf, 0.0, 0.7))
    max_bias: float = 0.0        # calibrate() fits a bias of the estimates within +- this (0: sd only)
    tail: float = 0.1            # share of estimates that miss by a wide margin (policy arithmetic)
    tail_sd: float = 1.0
    cap: float = 4.0             # accepted inflated charges are paid up to cap x t (rules: c >= 4t)
    # guard rails as shares of the estimate m
    min_ratio: float = 0.5
    max_ratio: float = 1.6


NEUTRAL = Params(gamma=1.0)
LOVING = Params(gamma=2.0)
SETTINGS = {"neutral": NEUTRAL, "loving": LOVING}


# ------------------------------------------------------------------ history of earlier rounds

@dataclass(frozen=True)
class Precedent:
    game: int
    line: int
    key: str
    desc: str
    qty: float
    unit: str
    kind: str            # bracket | lower   (zero / upper lines carry no price level)
    lo: float            # per unit
    hi: float            # per unit; inf if none
    policy: str
    tokens: frozenset
    claim: frozenset


@dataclass
class History:
    """Everything earlier rounds revealed, up to (not including) `before_game`."""
    before_game: int | None
    precedents: list[Precedent]
    digests: dict[int, dict]          # game -> casebook digest
    summary: dict                     # game (str) -> line (str) -> market summary
    accept: np.ndarray                # P(an opponent's limit >= a) on the _R grid of log(a / t)
    relevance: dict = field(default_factory=dict)     # match class -> P(same t per unit), see below
    by_ref: dict = field(default_factory=dict)
    util: dict = field(default_factory=dict)          # cached utility matrices, see decide()

    def __post_init__(self) -> None:
        self.by_ref = {(p.game, p.line): p for p in self.precedents}


def _claim_tokens(text: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z]{4,}", text.lower()))


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def t_point(s: dict, up: float = 1.15) -> float:
    """Same reading of the revealed bounds as c2f/replay.py."""
    if not any(v > 0 for v in s["charges"].values()):
        return 0.0
    lo, hi = s["t_low"], s["t_high"]
    if lo > 0 and hi:
        return (lo + hi) / 2
    if lo > 0:
        return lo * up
    return hi * 0.85 if hi else 0.0


def acceptance_curve(summary: dict, before_game: int | None = None, us: str = US,
                     decay: float = 0.97) -> np.ndarray:
    """P(opponent accepts a charge a) as a function of log(a / t), measured on the limit
    bounds of every opponent on every earlier covered line (newer games count more).
    Interpolation inside a bound as in c2f/replay.py. Falls back to a logistic curve
    around t when there is no history."""
    num = np.zeros_like(_R)
    den = 0.0
    last = max((int(g) for g in summary if before_game is None or int(g) < before_game), default=0)
    for g, items in summary.items():
        if before_game is not None and int(g) >= before_game:
            continue
        wg = decay ** (last - int(g))
        for s in items.values():
            t = t_point(s)
            if t <= 0:
                continue
            a = t * np.exp(_R)
            for k, (lo, hi) in s["b_bounds"].items():
                if k == us:
                    continue
                if hi is None:
                    p = np.where(a <= lo, 1.0, np.where(a <= 2 * lo, 0.5, 0.0))
                elif hi <= lo:
                    p = (a <= lo).astype(float)
                else:
                    p = np.clip((hi - a) / (hi - lo), 0.0, 1.0)
                num += wg * p
                den += wg
    if den < 30:
        return 1.0 / (1.0 + np.exp(_R / 0.25))
    # a looser charge is never accepted more often: enforce a non-increasing curve
    return np.minimum.accumulate(num / den)


def load_history(before_game: int | None = None, summary: dict | None = None,
                 casebook: str | Path | None = None) -> History:
    casebook = CASEBOOK if casebook is None else casebook
    if summary is None:
        summary = _load_json(SUMMARY) if SUMMARY.exists() else {}
    digests: dict[int, dict] = {}
    for g in summary:
        if before_game is not None and int(g) >= before_game:
            continue
        f = Path(casebook) / f"game{g}.json"
        if f.exists():
            digests[int(g)] = _load_json(f)
    precs = []
    for o in item_prior.observations(summary, casebook, before_game):
        if o.kind not in ("bracket", "lower"):
            continue
        d = digests.get(o.game, {})
        precs.append(Precedent(o.game, o.line, o.key, o.text, o.qty, o.unit, o.kind, o.lo, o.hi,
                               o.policy, novelty.tokens(o.text),
                               _claim_tokens(d.get("description", ""))))
    return History(before_game, precs, digests, summary,
                   acceptance_curve(summary, before_game), learn_relevance(precs))


_hist_lock = threading.Lock()
_hist_cache: dict = {}


def shared_history() -> History:
    """History from the files on disk, reloaded when the market summary changes."""
    stamp = SUMMARY.stat().st_mtime_ns if SUMMARY.exists() else 0
    with _hist_lock:
        if _hist_cache.get("stamp") != stamp:
            _hist_cache.update(stamp=stamp, hist=load_history())
        return _hist_cache["hist"]


# ------------------------------------------------------------------ relevance of earlier rounds

@dataclass(frozen=True)
class Weight:
    game: int
    line: int
    w: float
    factor: float = 1.0      # expected t per unit now / t per unit then
    why: str = ""


# Prior of P(an earlier line of this match class has the same t per unit): Beta with this mean
# and PRIOR_N pseudo-pairs. Updated with every pair of earlier lines, see learn_relevance().
PRIOR_W = {("same", "same"): 0.6, ("same", "other"): 0.25, ("fuzzy", "same"): 0.3, ("fuzzy", "other"): 0.2}
PRIOR_N = 6.0
SAME_TOL = 0.15          # two thresholds per unit within this log distance count as the same
CLAIM_SIM = 0.15         # claim descriptions with at least this word overlap count as similar


def _match(key: str, tok: frozenset, unit: str, policy: str, claim: frozenset, q: Precedent,
           min_sim: float) -> tuple | None:
    """Match class of an earlier line q for a line (key, tok, unit) of a case (policy, claim):
    (same | fuzzy line type, same | other policy wording, similar claim or not); None = no match."""
    if q.unit != unit:
        return None
    if q.key == key:
        kind = "same"
    else:
        if key.startswith("labour:") or q.key.startswith("labour:"):
            return None                       # labour grades have their own rates
        if novelty.similarity(tok, q.tokens) < min_sim:
            return None
        kind = "fuzzy"
    jac = (len(claim & q.claim) / len(claim | q.claim)) if claim and q.claim else 0.0
    return kind, "same" if (policy and q.policy == policy) else "other", jac >= CLAIM_SIM


def _point(q: Precedent) -> float:
    return math.sqrt(q.lo * q.hi) if math.isfinite(q.hi) else q.lo * 1.1


def learn_relevance(precs: list[Precedent], min_sim: float = 0.6) -> dict:
    """How informative is an earlier round? For every pair (earlier line, later line) of the
    same match class, count how often the later threshold per unit turned out within SAME_TOL
    of the earlier one. The share, shrunk towards PRIOR_W, is the relevance weight of that
    class: it is measured on the rounds played so far, not assumed."""
    hits: dict = {}
    for new in precs:
        for old in precs:
            if old.game >= new.game:
                continue
            cls = _match(new.key, new.tokens, new.unit, new.policy, new.claim, old, min_sim)
            if cls is None:
                continue
            h = hits.setdefault(cls, [0.0, 0.0])
            h[0] += abs(math.log(_point(new) / _point(old))) < SAME_TOL
            h[1] += 1
    out = {}
    for kind in ("same", "fuzzy"):
        for pol in ("same", "other"):
            n2 = sum(hits.get((kind, pol, c), [0, 0])[1] for c in (True, False))
            k2 = sum(hits.get((kind, pol, c), [0, 0])[0] for c in (True, False))
            base = (k2 + PRIOR_W[kind, pol] * PRIOR_N) / (n2 + PRIOR_N)
            for c in (True, False):                     # the claim split is pulled to its parent
                k, n = hits.get((kind, pol, c), [0.0, 0.0])
                out[kind, pol, c] = round((k + base * PRIOR_N) / (n + PRIOR_N), 3)
    return out


def rule_weights(lines: dict[int, str], policy: str, claim: str, hist: History,
                 p: Params = NEUTRAL, game: int | None = None) -> dict[int, list[Weight]]:
    """Mode "rule": the relevance of an earlier line is the measured share of its match class
    (same line type or similar text, same policy wording or not, similar claim or not) whose
    threshold later proved the same, times a small discount per game of age."""
    claim_tok = _claim_tokens(claim)
    now = game if game is not None else (hist.before_game or
                                         max((q.game for q in hist.precedents), default=0) + 1)
    out: dict[int, list[Weight]] = {}
    for pos, text in lines.items():
        desc, _qty, unit = item_prior.parse_line(text)
        key, _fam = item_prior.line_key(desc, unit)
        tok = novelty.tokens(desc)
        ws = []
        for q in hist.precedents:
            cls = _match(key, tok, unit, policy, claim_tok, q, p.min_sim)
            if cls is None:
                continue
            w = hist.relevance.get(cls, PRIOR_W[cls[:2]]) * p.decay ** max(0, now - q.game)
            if w >= 0.05:
                ws.append(Weight(q.game, q.line, min(w, p.w_max), 1.0,
                                 f"{cls[0]} line type, {cls[1]} wording, "
                                 f"{'similar' if cls[2] else 'other'} claim"))
        out[pos] = sorted(ws, key=lambda x: -x.w)[:12]
    return out


# ------------------------------------------------------------------ belief about t

def _norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def prior_density(sigma: float, p: Params) -> np.ndarray:
    """Calibrated belief of the assessment about log(t / m) on the _X grid."""
    bias, sd = next(((b, s) for edge, b, s in p.cal if sigma < edge), p.cal[-1][1:])
    d = (1 - p.tail) * _norm_pdf(_X, bias, sd) + p.tail * _norm_pdf(_X, bias, max(p.tail_sd, sd))
    return d / d.sum()


def _evidence(kind: str, lo: float, hi: float, level: np.ndarray, p: Params) -> np.ndarray:
    """Likelihood of what a precedent proved, given the log level of t per unit now.
    lo, hi: the precedent's bounds per unit (already scaled by the factor)."""
    if kind == "bracket":
        return ndtr((math.log(hi) - level) / p.tau) - ndtr((math.log(lo) - level) / p.tau)
    # lower bound: t = lo * exp(gap), gap ~ Exp(rate); smoothed by tau
    g = level - math.log(lo)
    lam = p.gap_rate
    # density of gap + noise at g: exponential convolved with a normal
    z = g / p.tau - lam * p.tau
    return lam * np.exp(np.clip(-lam * g + 0.5 * (lam * p.tau) ** 2, -60, 60)) * ndtr(z)


def posterior(m: float, sigma: float, qty: float, weights: list[Weight], hist: History,
              p: Params) -> tuple[np.ndarray, list[dict]]:
    """Distribution of log(t / m) on _X and, per precedent, how much it moved the belief."""
    prior = prior_density(sigma, p)
    if m <= 0 or not weights:
        return prior, []
    level = _X + math.log(m / max(qty, 1e-9))            # log t per unit now
    broad = _norm_pdf(_X, 0.0, p.broad)
    broad /= broad.sum()
    lik = np.ones_like(_X)
    used = []
    for w in weights:
        q = hist.by_ref.get((w.game, w.line))
        if q is None or w.w <= 0 or w.factor <= 0:
            continue
        ww = min(max(w.w, 0.0), p.w_max)
        hi = q.hi * w.factor if math.isfinite(q.hi) else math.inf
        ev = _evidence(q.kind, q.lo * w.factor, hi, level, p)
        base = float((broad * ev).sum())                 # same evidence if t were unrelated
        if base <= 0:
            continue
        term = (1 - ww) + ww * ev / base
        lik *= term ** p.temper
        used.append({"game": q.game, "line": q.line, "w": round(ww, 2), "kind": q.kind,
                     "lo": round(q.lo * w.factor * qty, 2),
                     "hi": round(hi * qty, 2) if math.isfinite(hi) else None,
                     "pull": round(float((prior * ev).sum() / base), 2), "why": w.why})
    # the level is shared by the precedents; the current case deviates from it by tau
    k = _norm_pdf(np.arange(-4 * p.tau, 4 * p.tau + 1e-9, 0.01), 0.0, p.tau)
    lik = np.convolve(lik, k / k.sum(), mode="same")
    post = prior * lik
    s = post.sum()
    return (post / s if s > 0 else prior), used


# ------------------------------------------------------------------ decision

def _revenue(a: np.ndarray, t: np.ndarray, accept: np.ndarray, cap: float) -> np.ndarray:
    """Mean payment per opponent for charges a (rows) and thresholds t (columns)."""
    A, T = a[:, None], t[None, :]
    pacc = np.interp(np.log(A / T), _R, accept, left=1.0, right=0.0)
    fair = A <= T
    return np.where(fair, A * (1.5 - 0.5 * pacc), np.minimum(A, cap * T) * pacc)


@dataclass
class Decision:
    a: float
    ratio: float                 # a / m
    t_q10: float
    t_q50: float
    t_q90: float
    p_fair: float                # P(a <= t) under the belief
    n_precedents: int
    precedents: list[dict]


def decide(m: float, sigma: float, qty: float, weights: list[Weight], hist: History,
           p: Params = NEUTRAL) -> Decision:
    if m <= 0:
        return Decision(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, [])
    post, used = posterior(m, sigma, qty, weights, hist, p)
    y = _Y[(_Y >= math.log(p.min_ratio)) & (_Y <= math.log(p.max_ratio))]
    key = (p.cap, p.gamma, p.min_ratio, p.max_ratio)
    if key not in hist.util:                                    # the same for every line: in units of m
        hist.util[key] = _revenue(np.exp(y), np.exp(_X), hist.accept, p.cap) ** p.gamma
    util = hist.util[key] @ post
    yb = float(y[int(np.argmax(util))])
    cdf = np.cumsum(post)
    q = [float(m * math.exp(_X[min(int(np.searchsorted(cdf, c)), len(_X) - 1)])) for c in (0.1, 0.5, 0.9)]
    p_fair = float(post[_X >= yb - 1e-9].sum())
    used.sort(key=lambda u: -abs(math.log(max(u["pull"], 1e-6))) * u["w"])
    return Decision(round(m * math.exp(yb), 2), round(math.exp(yb), 3), *[round(v, 2) for v in q],
                    round(p_fair, 3), len(used), used[:6])


# ------------------------------------------------------------------ calibration on earlier rounds

def estimate_history(summary: dict, extractions: str | Path | None = None,
                     model: str = "gpt-6.2-sol", before_game: int | None = None) -> list[tuple]:
    """(game, m, stated sigma, t_low, t_high or inf) for every earlier line the assessment
    believed indemnified and on which the transactions bound t from below."""
    rows = []
    extractions = EXTRACTIONS if extractions is None else extractions
    for g, items in summary.items():
        if before_game is not None and int(g) >= before_game:
            continue
        fs = sorted(Path(extractions).glob(f"game{g}__x*__{model}.json"))
        if not fs:
            continue
        try:
            est = {str(it["item_id"]).split()[0].strip("."): it for it in _load_json(fs[-1])["items"]}
        except Exception:  # noqa: BLE001
            continue
        for i, s in items.items():
            it = est.get(str(i))
            if not it or s["t_low"] <= 0:
                continue
            m = float(it["quantity"]) * float(it["unit_price_gross_eur"])
            if m <= 0 or it["covered_prob"] * it["related_prob"] < 0.5:
                continue
            rows.append((int(g), m, float(it["price_log_sd"]), s["t_low"], s["t_high"] or math.inf))
    return rows


def _fit_class(like: np.ndarray, p: Params, prior_fit: tuple | None = None,
               prior_n: float = 0.0) -> tuple[float, float]:
    """(bias, sd) of one class by censored maximum likelihood on a grid. prior_fit with
    prior_n pseudo-lines pulls a small class towards the fit of all lines."""
    best, best_ll = (0.0, 0.4), -math.inf
    for bias in (np.arange(-p.max_bias, p.max_bias + 1e-9, 0.05) if p.max_bias > 0 else (0.0,)):
        for sd in (0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            d = (1 - p.tail) * _norm_pdf(_X, bias, sd) + p.tail * _norm_pdf(_X, bias, max(p.tail_sd, sd))
            d /= d.sum()
            ll = float(np.log(np.maximum(like @ d, 1e-300)).sum())
            if prior_fit is not None:
                ll -= prior_n * (0.5 * ((bias - prior_fit[0]) / 0.15) ** 2
                                 + 0.5 * (math.log(sd / prior_fit[1]) / 0.5) ** 2)
            if ll > best_ll:
                best, best_ll = (float(bias), float(sd)), ll
    return best


def calibrate(rows: list[tuple], p: Params = NEUTRAL, min_rows: int = 25) -> Params:
    """Fit bias and sd of the assessment's error log(t / m) per class of stated log-sd, by
    censored maximum likelihood (bracket: probability of the interval; lower bound only:
    gap ~ Exp). Every class is pulled towards its default (3 pseudo-lines). Keeps the
    defaults with fewer than min_rows lines."""
    if len(rows) < min_rows:
        return p
    likes, sgs = [], []
    for _g, m, sg, lo, hi in rows:
        if math.isfinite(hi) and hi > lo:
            like = ((_X >= math.log(lo / m)) & (_X < math.log(hi / m))).astype(float)
            if like.sum() == 0:
                like = _norm_pdf(_X, math.log(math.sqrt(lo * hi) / m), 0.02)
        else:
            gap = _X - math.log(lo / m)
            like = np.where(gap >= 0, p.gap_rate * np.exp(-p.gap_rate * np.clip(gap, 0, 50)), 0.0)
        if like.sum() <= 0:
            continue                                   # outside the grid: no information
        likes.append(like / like.sum())
        sgs.append(sg)
    like, sg = np.stack(likes), np.array(sgs)
    cal, lo_edge = [], -math.inf
    for edge, b0, s0 in p.cal:
        sel = (sg >= lo_edge) & (sg < edge)
        cal.append((edge, *(_fit_class(like[sel], p, (b0, s0), 3.0) if sel.sum() >= 4 else (b0, s0))))
        lo_edge = edge
    return replace(p, cal=tuple(cal))


# ------------------------------------------------------------------ mode "llm": relevance judge

JUDGE_SYSTEM = """You support a pricing agent in an insurance claims game. Every round has one claim:
a policy wording, a damage description and a contractor invoice. Each invoice line has a secret
threshold t (the amount the insurer owes for that line, gross, for the whole line; 0 if not
indemnified). After a round the public transactions reveal bounds on t for every line.

You receive EARLIER CASES with those revealed bounds, then the CURRENT CASE. For every line of
the current invoice, name the earlier lines whose revealed threshold is informative for the
threshold of the current line, and how informative.

Rules:
- An earlier line is informative when it is the same work or item at the same indemnity basis:
  same kind of object and specification, same trade and grade for labour, same policy treatment
  (cover applies in full in both, or the same cap / percentage applies in both).
- Compare per unit: the amounts shown are per unit (per hour, per piece, per m, per flat rate).
- w between 0 and 1: 1 = near-identical case and line (expect the same t per unit within a few
  percent), 0.5 = same line type but another kind of claim or wording, below 0.3 = loosely
  related. Leave out lines that are not informative.
- factor: expected t per unit of the current line divided by t per unit of the earlier line.
  1.0 unless the cases state a concrete difference (e.g. the earlier indemnity was halved for a
  breached duty and the current is not: 2.0; the current item is a stated share of the earlier
  one). Do not guess market differences; use 1.0 then.
- Earlier lines with "t = 0" tell nothing about the price level: leave them out.
- Several earlier rounds of the same line may all be listed; rounds whose claim and wording
  match the current case best get the highest w.

Answer with JSON only:
{"lines": [{"pos": <current line number>, "refs": [{"game": <int>, "line": <int>, "w": <0..1>,
"factor": <number>}]}]}
Include every current line; "refs" may be empty."""


def _eur(x: float) -> str:
    return f"{x:.2f}" if x < 100 else f"{x:.0f}"


def _prec_line(text: str, s: dict) -> str:
    desc, qty, unit = item_prior.parse_line(text)
    k = item_prior._kind(s)
    per = f" per {unit}" if unit and qty != 1 else ""
    if k is None:
        return f"{text} => t unknown"
    kind, lo, hi = k
    if kind == "zero":
        return f"{text} => t = 0"
    if kind == "bracket":
        return f"{text} => t{per} between {_eur(lo / qty)} and {_eur(hi / qty)}"
    if kind == "lower":
        return f"{text} => t{per} >= {_eur(lo / qty)} (highest charge proven fair)"
    return f"{text} => t{per} below {_eur(hi / qty)}, probably 0"


def judge_prompt(lines: dict[int, str], policy: str, claim: str, hist: History,
                 max_chars: int = 60000) -> str:
    """All earlier cases (most similar first, so a cut drops the least similar) + current case."""
    cur_tok = frozenset().union(*[novelty.tokens(t) for t in lines.values()]) if lines else frozenset()
    claim_tok = _claim_tokens(claim)
    blocks = []
    for g, d in hist.digests.items():
        items = hist.summary.get(str(g), {})
        rows = [f"  L{i}: {_prec_line(d['lines'][str(i)], s)}"
                for i, s in sorted((int(k), v) for k, v in items.items()) if str(i) in d["lines"]]
        if not rows:
            continue
        old_tok = frozenset().union(*[novelty.tokens(t) for t in d["lines"].values()])
        old_claim = _claim_tokens(d.get("description", ""))
        jac = lambda x, y: len(x & y) / len(x | y) if x and y else 0.0  # noqa: E731
        rel = (1.0 if policy and d.get("policy") == policy else 0.0) + 2 * jac(cur_tok, old_tok) \
            + jac(claim_tok, old_claim)
        blocks.append((rel, g, f"Case {g} [{d.get('policy', '')}]\n  Claim: {d.get('description', '')[:500]}\n"
                       + "\n".join(rows)))
    blocks.sort(key=lambda b: (-b[0], -b[1]))
    ref = ""
    for _rel, _g, b in blocks:
        if len(ref) + len(b) <= max_chars:
            ref += b + "\n\n"
    cur = "\n".join(f"  {pos}: {text}" for pos, text in sorted(lines.items()))
    return (f"EARLIER CASES (revealed thresholds):\n{ref}\nCURRENT CASE [{policy}]\n  Claim: {claim}\n"
            f"  Invoice lines:\n{cur}")


def parse_judgement(payload: str, hist: History) -> dict[int, list[Weight]]:
    data = json.loads(payload)
    out: dict[int, list[Weight]] = {}
    for row in data.get("lines", []):
        try:
            pos = int(row["pos"])
        except (KeyError, TypeError, ValueError):
            continue
        ws = []
        for r in row.get("refs") or []:
            try:
                g, ln = int(r["game"]), int(r["line"])
                w, f = float(r.get("w", 0)), float(r.get("factor", 1) or 1)
            except (KeyError, TypeError, ValueError):
                continue
            if (g, ln) in hist.by_ref and 0 < w <= 1 and 0.1 <= f <= 10:
                ws.append(Weight(g, ln, w, f, "llm"))
        out[pos] = sorted(ws, key=lambda x: -x.w)[:12]
    return out


def judge(lines: dict[int, str], policy: str, claim: str, hist: History, model: str = "gpt-4.1",
          timeout: float = 20.0, effort: str | None = None,
          max_chars: int = 60000) -> tuple[dict[int, list[Weight]], dict]:
    """One model call. Raises on any failure (the caller falls back to rule weights)."""
    from .extract import _create_with_failover, _model_args
    t0 = time.monotonic()
    resp = _create_with_failover(
        timeout, model=model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": judge_prompt(lines, policy, claim, hist, max_chars)}],
        **_model_args(model, 0.0, effort))
    ws = parse_judgement(resp.choices[0].message.content, hist)
    u = getattr(resp, "usage", None)
    meta = {"model": model, "seconds": round(time.monotonic() - t0, 2),
            "input_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None)}
    return ws, meta


# ---- calibrated use of the judgement
#
# Measured on games 13-47 (both judge models): an earlier line the judge names with w >= 0.9
# has the same threshold per unit in about 70 % of cases, with w 0.7-0.9 in 40-60 %, below
# that in 5-20 %; a line it names that is no text match at all (other unit, other line type)
# in under 10 %. So the judge's weight is not used as a probability. Like the rule weights,
# the relevance is the measured share per class, here (rule match class, judge's weight
# class), learned from the stored judgements of earlier games.

JUDGE_DIR = REPO_ROOT / "results" / "charge_agent" / "judge"


def _coarse(cls: tuple | None) -> str:
    if cls is None:
        return "none"
    if cls[0] == "fuzzy":
        return "fuzzy"
    return "same/same" if cls[1] == "same" else ("same/other+claim" if cls[2] else "same/other")


def _bin(w: float | None) -> str:
    return "not" if w is None else "high" if w >= 0.9 else "mid" if w >= 0.7 else "low"


def _judge_prior(coarse: str, b: str, hist: History) -> float:
    r = {"none": 0.05, "fuzzy": hist.relevance.get(("fuzzy", "same", False), 0.3),
         "same/same": hist.relevance.get(("same", "same", False), 0.6),
         "same/other": hist.relevance.get(("same", "other", False), 0.25),
         "same/other+claim": hist.relevance.get(("same", "other", True), 0.25)}[coarse]
    return {"high": min(0.9, 0.5 * r + 0.35), "mid": 0.5 * r + 0.2, "low": 0.4 * r, "not": 0.7 * r}[b]


def _candidates(key: str, tok: frozenset, unit: str, policy: str, claim: frozenset, hist: History,
                cited: dict, before: int, min_sim: float) -> list[tuple]:
    """(precedent, coarse rule class, judge bin, factor) for every earlier line that the rule
    matches or the judge named; lines in another unit cannot be compared per unit."""
    out = []
    for q in hist.precedents:
        if q.game >= before or q.unit != unit:
            continue
        cls = _match(key, tok, unit, policy, claim, q, min_sim)
        c = cited.get((q.game, q.line))
        if cls is None and c is None:
            continue
        out.append((q, _coarse(cls), _bin(c.w if c else None), c.factor if c else 1.0))
    return out


def load_judgements(directory: str | Path | None = None, pattern: str = "game{g}.json",
                    before_game: int | None = None) -> dict[int, dict[int, list[Weight]]]:
    out = {}
    d = Path(JUDGE_DIR if directory is None else directory)
    if not d.exists():
        return out
    rx = re.compile("^" + re.escape(pattern).replace(r"\{g\}", r"(\d+)") + "$")
    for f in d.iterdir():
        m = rx.match(f.name)
        if not m or (before_game is not None and int(m.group(1)) >= before_game):
            continue
        try:
            j = _load_json(f)
            out[int(m.group(1))] = {int(k): [Weight(**w) for w in v] for k, v in j["weights"].items()}
        except Exception:  # noqa: BLE001
            continue
    return out


def save_judgement(game: int, weights: dict[int, list[Weight]], meta: dict,
                   directory: str | Path | None = None, name: str = "game{g}.json") -> None:
    from dataclasses import asdict
    directory = JUDGE_DIR if directory is None else directory
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / name.format(g=game)).write_text(json.dumps(
        {"meta": meta, "weights": {k: [asdict(w) for w in v] for k, v in weights.items()}}, indent=1))


def learn_judge(hist: History, judgements: dict[int, dict[int, list[Weight]]],
                min_sim: float = 0.6) -> dict:
    """(rule class, judge class) -> share of earlier lines whose threshold per unit later
    proved the same, shrunk towards a prior. Uses only finished games with a stored judgement."""
    hits: dict = {}
    for g, jw in judgements.items():
        for pos, refs in jw.items():
            new = hist.by_ref.get((g, pos))
            if new is None:
                continue
            cited = {(r.game, r.line): r for r in refs}
            for q, coarse, b, factor in _candidates(new.key, new.tokens, new.unit, new.policy, new.claim,
                                                    hist, cited, g, min_sim):
                h = hits.setdefault((coarse, b), [0.0, 0.0])
                h[0] += abs(math.log(_point(new) / (_point(q) * factor))) < SAME_TOL
                h[1] += 1
    out = {}
    for coarse in ("same/same", "same/other+claim", "same/other", "fuzzy", "none"):
        for b in ("high", "mid", "low", "not"):
            k, n = hits.get((coarse, b), [0.0, 0.0])
            out[coarse, b] = round((k + _judge_prior(coarse, b, hist) * PRIOR_N) / (n + PRIOR_N), 3)
    return out


def llm_weights(lines: dict[int, str], policy: str, claim: str, hist: History,
                judgement: dict[int, list[Weight]], table: dict | None = None,
                p: Params = NEUTRAL, game: int | None = None) -> dict[int, list[Weight]]:
    """Mode "llm": relevance from the rule match class and the judge's weight class together.
    table: learn_judge(); None -> the priors."""
    claim_tok = _claim_tokens(claim)
    now = game if game is not None else (hist.before_game or
                                         max((q.game for q in hist.precedents), default=0) + 1)
    out: dict[int, list[Weight]] = {}
    for pos, text in lines.items():
        desc, _qty, unit = item_prior.parse_line(text)
        key, _fam = item_prior.line_key(desc, unit)
        cited = {(r.game, r.line): r for r in judgement.get(pos, [])}
        ws = []
        for q, coarse, b, factor in _candidates(key, novelty.tokens(desc), unit, policy, claim_tok, hist,
                                                cited, now, p.min_sim):
            rate = table.get((coarse, b)) if table else None
            w = (rate if rate is not None else _judge_prior(coarse, b, hist)) * p.decay ** max(0, now - q.game)
            if w >= 0.05:
                ws.append(Weight(q.game, q.line, min(w, p.w_max), factor, f"rule {coarse}, judge {b}"))
        out[pos] = sorted(ws, key=lambda x: -x.w)[:12]
    return out


# One judgement per case, shared by the parallel stages of a game (fast, full, backup).
_judge_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="charge-judge")
_judge_lock = threading.Lock()
_judge_jobs: dict[str, tuple[float, Future]] = {}


def _judge_job(lines: dict[int, str], policy: str, claim: str, model: str, timeout: float,
               effort: str | None):
    """Runs in the background: load the history (cached), then one judge call."""
    return judge(lines, policy, claim, shared_history(), model, timeout, effort)


def start_judge(case_id: str, lines: dict[int, str], policy: str, claim: str, model: str,
                timeout: float, effort: str | None = None) -> tuple[float, Future]:
    """Start (or join) the judgement of this case in the background; returns (start, future).
    Returns at once: nothing is loaded or called on the caller's thread."""
    with _judge_lock:
        job = _judge_jobs.get(case_id)
        if job is None:
            if len(_judge_jobs) > 8:
                _judge_jobs.clear()
            job = (time.monotonic(), _judge_pool.submit(_judge_job, lines, policy, claim, model,
                                                        timeout, effort))
            _judge_jobs[case_id] = job
        return job


# ------------------------------------------------------------------ entry point for the agent

@dataclass
class Line:
    pos: int
    text: str
    m: float            # the assessment's estimate of t for the whole line
    sigma: float        # its stated log-sd


def charges(lines: list[Line], policy: str, claim: str, hist: History, p: Params,
            weights: dict[int, list[Weight]] | None = None,
            game: int | None = None) -> dict[int, Decision]:
    """Charge per line. weights: from llm_weights() (mode "llm"); None or a missing line ->
    rule weights for that line."""
    texts = {ln.pos: ln.text for ln in lines}
    rule = rule_weights(texts, policy, claim, hist, p, game)
    out = {}
    for ln in lines:
        _desc, qty, _unit = item_prior.parse_line(ln.text)
        ws = weights[ln.pos] if (weights is not None and ln.pos in weights) else rule.get(ln.pos, [])
        out[ln.pos] = decide(ln.m, ln.sigma, qty, ws, hist, p)
    return out


_cal_lock = threading.Lock()
_cal_cache: dict = {}


def shared_params(setting: str, model: str = "gpt-6.2-sol", gamma_loving: float | None = None) -> Params:
    """Params of a risk setting with the belief calibration fitted on all rounds on disk."""
    hist = shared_history()
    key = (setting, model, gamma_loving, id(hist))
    with _cal_lock:
        if key not in _cal_cache:
            base = SETTINGS.get(setting, NEUTRAL)
            if setting == "loving" and gamma_loving:
                base = replace(base, gamma=float(gamma_loving))
            try:
                base = calibrate(estimate_history(hist.summary, model=model), base)
            except Exception as e:  # noqa: BLE001
                print(f"[charge_agent] calibration skipped: {e!r}")
            for k in [k for k in _cal_cache if k[-1] != id(hist)]:
                del _cal_cache[k]
            _cal_cache[key] = base
        return _cal_cache[key]


def _game_number(case_id: str) -> int | None:
    m = re.match(r"game(\d+)", str(case_id))
    return int(m.group(1)) if m else None


def warm(cfg_section: dict, model: str = "gpt-6.2-sol") -> None:
    """Load history and calibration ahead of the first pricing call (about 1 s once)."""
    try:
        shared_params(cfg_section.get("risk", "neutral"), model, cfg_section.get("gamma_loving"))
    except Exception as e:  # noqa: BLE001
        print(f"[charge_agent] warm-up failed: {e!r}")


def begin(case, cfg_section: dict, model: str = "gpt-6.2-sol") -> None:
    """Called when the pricing of a case starts. Returns at once and never raises: history
    and calibration are loaded in the background (they are ready long before the assessment
    answers), and in mode "llm" the judge is started, once per case, shared by the stages."""
    try:
        mode = cfg_section.get("mode", "off")
        if mode not in ("rule", "llm"):
            return
        _judge_pool.submit(warm, cfg_section, model)
        if cfg_section.get("shadow", False):
            _judge_pool.submit(shared_takeover, int(cfg_section.get("takeover_rounds", 3)))
        if mode == "llm":
            from .pricebook import invoice_lines, policy_title
            jm, _, effort = str(cfg_section.get("judge_model", "gpt-4.1")).partition("@")
            start_judge(case.case_id, invoice_lines(case.invoice_text), policy_title(case.policy),
                        " ".join(case.description.split()), jm,
                        float(cfg_section.get("judge_timeout_seconds", 15)), effort or None)
    except Exception as e:  # noqa: BLE001
        print(f"[charge_agent] not started: {e!r}")


def live_charges(case, estimates: dict[int, tuple[float, float]], cfg_section: dict,
                 model: str = "gpt-6.2-sol", wait: bool = True,
                 time_left: float | None = None) -> tuple[dict[int, Decision], dict]:
    """Charges for the lines of a live case. estimates: POS -> (m, stated log-sd).

    Time: mode "rule" needs no model call (milliseconds once the history is loaded). Mode
    "llm" uses the judge's answer if it is there; with wait=True it waits for it, but only
    until judge_wait_seconds after the judge was started and never longer than time_left - 6 s.
    After that, or when the judge failed, the rule weights are used. Raises on internal
    errors; the caller keeps the fixed-rule charges then."""
    from .pricebook import invoice_lines, policy_title
    t0 = time.monotonic()
    mode = cfg_section.get("mode", "off")
    hist = shared_history()
    p = shared_params(cfg_section.get("risk", "neutral"), model, cfg_section.get("gamma_loving"))
    texts = invoice_lines(case.invoice_text)
    policy, claim = policy_title(case.policy), " ".join(case.description.split())
    game = _game_number(case.case_id) or None      # game 0 is the test game: no position in time
    lines = [Line(pos, texts[pos], m, sg) for pos, (m, sg) in estimates.items() if pos in texts and m > 0]
    info: dict = {"mode": mode, "risk": cfg_section.get("risk", "neutral"), "gamma": p.gamma, "weights": "rule"}
    weights = None
    if mode == "llm":
        with _judge_lock:
            job = _judge_jobs.get(case.case_id)
        if job is not None:
            started, fut = job
            budget = 0.0
            if wait:
                budget = float(cfg_section.get("judge_wait_seconds", 20)) - (time.monotonic() - started)
                if time_left is not None:
                    budget = min(budget, time_left - 6.0)
            try:
                judgement, meta = fut.result(timeout=max(budget, 0.0))
                table = learn_judge(hist, load_judgements(before_game=game))
                weights = llm_weights(texts, policy, claim, hist, judgement, table, p, game)
                info.update(weights="llm", judge=meta)
                if game is not None and not (JUDGE_DIR / f"game{game}.json").exists():
                    try:
                        save_judgement(game, judgement, meta)
                    except OSError:
                        pass
            except Exception as e:  # noqa: BLE001  timeout or model failure -> rule weights
                info["judge_error"] = (type(e).__name__ + ": " + str(e))[:160] or "not ready"
    dec = charges(lines, policy, claim, hist, p, weights=weights, game=game)
    info["lines_with_precedents"] = sum(d.n_precedents > 0 for d in dec.values())
    info["seconds"] = round(time.monotonic() - t0, 3)
    return dec, info


# ------------------------------------------------------------------ shadow run and takeover
#
# [charge_agent] shadow = true: the agent prices every live game, but its charges are only
# recorded (runs/submissions.jsonl, timing.charge_agent.shadow_charges) while the fixed rule's
# charges are submitted. After a game's transactions are public, both sets of charges are
# scored against the real opponents of that game. Once the agent has earned more than the
# fixed rule in `takeover_rounds` scored rounds in a row, its charges are submitted from the
# next game on. The decision is recomputed from the run log and the market summary, so every
# machine that has both files reaches the same answer; nothing else is stored.

QUOTES = REPO_ROOT / "runs" / "submissions.jsonl"
SHADOW_REPORT = REPO_ROOT / "results" / "charge_agent" / "shadow.json"


def line_revenue(a: float, s: dict, up: float = 1.15, cap: float | None = 4.0) -> float:
    """Handyman revenue of charge a on a finished line against the real opponents
    (c2f/replay.py rules). Where t is bracketed, the mean over t spread evenly across the
    bracket (a single midpoint makes charges inside the bracket a coin flip); where only a
    lower bound is known, t = up x that bound."""
    from . import replay
    lo, hi = s["t_low"], s["t_high"]
    if lo > 0 and hi and hi > lo and any(v > 0 for v in s["charges"].values()):
        ts = [lo + (hi - lo) * (i + 0.5) / 9 for i in range(9)]
    else:
        ts = [t_point(s, up)]
    return sum(replay.item_net(a, 0.0, s, t, cap=cap) - replay.item_net(0.0, 0.0, s, t, cap=cap)
               for t in ts) / len(ts)


def _run_records(quotes: str | Path) -> dict[int, dict]:
    """Game -> last successful live record with a payload (what stood on the server)."""
    out: dict[int, dict] = {}
    p = Path(quotes)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("game_id") is None or not r.get("payload") or r.get("dry_run") or not r.get("ok", True):
            continue
        out[int(r["game_id"])] = r
    return out


def takeover_status(quotes: str | Path | None = None, summary: dict | None = None, rounds: int = 3,
                    min_diff: float = 1.0) -> dict:
    """Score every finished shadow round and decide whether the agent has taken over.

    A round counts when the agent ran in shadow and the game's transactions are public. It
    is a win when the agent's charges would have earned more than the submitted ones by more
    than min_diff, a loss when less; a round without a difference (nothing covered, or the
    same charges) is left out and does not break a streak. `rounds` wins in a row switch the
    agent live from the next game; the switch is not taken back automatically.
    Returns {"active", "since_game", "streak", "needed", "rounds": [{game, legacy, agent, diff}]}."""
    if summary is None:
        summary = _load_json(SUMMARY) if SUMMARY.exists() else {}
    records = _run_records(QUOTES if quotes is None else quotes)
    out: dict = {"active": False, "since_game": None, "streak": 0, "needed": int(rounds), "rounds": []}
    for g in sorted(records):
        info = (records[g].get("timing") or {}).get("charge_agent") or {}
        if info.get("submitted") == "agent" and info.get("takeover") and not out["active"]:
            out.update(active=True, since_game=g)        # switched on a machine with more history
        shadow = info.get("shadow_charges")
        items = summary.get(str(g))
        if out["active"] or not shadow or not items or g == 0:
            continue
        legacy = agent = 0.0
        for q in records[g]["payload"]:
            s = items.get(str(q.get("index")))
            if s is None:
                continue
            a_old = float(q.get("charge_price") or 0.0)
            a_new = float(shadow.get(str(q.get("index")), a_old))
            legacy += line_revenue(a_old, s)
            agent += line_revenue(a_new, s)
        diff = agent - legacy
        out["rounds"].append({"game": g, "legacy": round(legacy, 1), "agent": round(agent, 1),
                              "diff": round(diff, 1)})
        if diff > min_diff:
            out["streak"] += 1
        elif diff < -min_diff:
            out["streak"] = 0
        if rounds > 0 and out["streak"] >= rounds:
            later = [h for h in records if h > g]
            out.update(active=True, since_game=min(later) if later else g + 1)
    return out


_take_lock = threading.Lock()
_take_cache: dict = {}


def shared_takeover(rounds: int = 3) -> dict:
    """takeover_status() on the files on disk, recomputed when one of them changes."""
    stamp = (rounds, *(f.stat().st_mtime_ns if f.exists() else 0 for f in (QUOTES, SUMMARY)))
    with _take_lock:
        if _take_cache.get("stamp") != stamp:
            _take_cache.update(stamp=stamp, status=takeover_status(rounds=rounds))
        return _take_cache["status"]


def write_shadow_report(rounds: int = 3, path: str | Path | None = None) -> dict:
    """Score the shadow rounds and save the result for the team (results/charge_agent/shadow.json)."""
    st = takeover_status(rounds=rounds)
    path = Path(SHADOW_REPORT if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(st, indent=1)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return st
