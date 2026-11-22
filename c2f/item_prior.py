"""Item prior: the distribution of the secret threshold t for recurring invoice line types.

Invoice lines repeat across games ("Service technician hours", "Vehicle costs",
"Construction waste disposal", "Profipress elbow 90 copper 15mm", "Processing flat fee") and
their threshold per unit is stable. After every game the public transactions bound t for each
line (c2f/market.py). This module pools those bounds per line type and fits, per type,

    t / quantity = 0                        with probability p0   (not indemnified)
                 = exp(mu + sigma Z)        otherwise             (Z standard normal)

The data are censored, never exact, so each earlier line enters with what it proves:

    nobody was paid                 t = 0                         likelihood p0
    bracket   lo <= t < hi          (1 - p0) [F(hi) - F(lo)]
    lower     t >= lo, no hi        (1 - p0) E[ lam exp(-lam (ln t - ln lo)) ; t >= lo ]
    upper     t < hi, lo = 0        p0 + (1 - p0) kappa F(hi)     (see below)

"upper" means somebody was paid on the line but no charge was proven fair. With t > 0 and
a dozen teams charging, at least one fair charge is almost certain, so such a line is nearly
always an uncovered one on which a lenient reviewer accepted a small charge; kappa is the
small probability that a covered line shows no fair charge.

A lower bound is the highest charge that was proven fair. Several teams price close to t, so
t is usually just above it: the gap ln t - ln lo is modelled as exponential with rate
`gap_rate` (mean gap 1 / 12, about 8 %; sensitivity: scripts/item_prior_backtest.py --sweep).
Without this term a line type seen only with lower bounds would have no upper end at all.

Policy arithmetic (aggregate caps, percentage reductions) sometimes pushes t far below the
ordinary price of the work. So that one such case does not drag the price level down, every
positive observation comes from the main lognormal with probability 1 - eps and from a wide
component centred lower with probability eps. A game in which the recurring lines sit
jointly far below their usual level (an aggregate cap shared across the invoice, game 41) is
detected from the data and its amounts are left out of the price level altogether.

The level of a line type also depends on the kind of case: "Vehicle costs" is about 80 under
the plain dwelling wording (pipe leaks) and about 40 under the frost edition. When the line
type has history under the policy wording of the current case, lines from other wordings
count with weight `other_policy` only (a tempered likelihood).

The posterior over (p0, mu, sigma) is computed on a grid: flat prior on mu (for hourly labour
a normal prior around the pooled level of all grades), a lognormal prior on sigma and a Beta
prior on p0. `context()` turns the fitted priors for the lines of the current invoice into
the text block that the case assessment model receives next to the reference cases.
"""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._numerics import erfc, ndtr
from .config import REPO_ROOT

CASEBOOK = REPO_ROOT / "results" / "casebook"
SUMMARY = REPO_ROOT / "results" / "market" / "summary.json"


@dataclass(frozen=True)
class Params:
    gap_rate: float = 12.0        # lam: mean gap between the best fair charge and t is 1 / lam
    sigma_center: float = 0.12    # prior median of the log-sd of t per unit within a line type
    sigma_width: float = 0.8      # log-sd of that prior
    eps: float = 0.10             # share of positive observations distorted by policy arithmetic
    wide_shift: float = -0.7      # the distorted component: N(mu + wide_shift, wide_sigma)
    wide_sigma: float = 0.8
    other_policy: float = 0.3     # weight of other policy wordings once the same wording has history
    upper_pos: float = 0.05       # kappa: P(no charge proven fair | line indemnified)
    deflated_shift: float = -0.3  # a game whose recurring lines sit this far (log) below their
    deflated_min_lines: int = 3   # level in other games, on at least this many lines, is capped
    p0_a: float = 0.5             # Beta prior on p0
    p0_b: float = 0.5
    family_spread: float = 1.5    # prior sd of a grade's level = this x the family's sd
    fuzzy: float = 0.0            # word overlap needed to borrow from a similar line type (0 = off;
                                  # 0.5 was worse in the backtest: "freight shipping" is not "shipping")
    min_obs: int = 2              # earlier lines needed before a line type is reported...
    min_priced: int = 1           # ...or this many with a positive lower bound


DEFAULT = Params()


# ---------------------------------------------------------------- line -> (type, quantity)

_QTY = re.compile(r"^(?P<desc>.*?)[\s|]+(?P<qty>\d[\d.,]*)\s+(?P<unit>flat rate|linear m|[^\s\d]\S*)\s*$")
_GRADE = re.compile(r"(technician|installer|helper|worker|apprentice|painter|electrician|"
                    r"fitter|plumber|labourer|locksmith|joiner|tiler|roofer|glazier)$")
_ADMIN = re.compile(r"\b(administrati\w*|processing|handling)\b.*\b(fee|flat rate|flat fee|charge)\b")
_UNITS = {"hrs": "hr", "hours": "hr", "h": "hr", "pcs": "pc", "days": "day", "lines": "line",
          "flat rate": "flat rate", "linear m": "m"}


def parse_line(text: str) -> tuple[str, float, str]:
    """'Service technician hours 6.75 hrs' -> ('Service technician hours', 6.75, 'hr')."""
    text = " ".join(text.split())
    m = _QTY.match(text)
    if not m:
        return text.strip(" |"), 1.0, ""
    try:
        qty = float(m.group("qty").replace(",", ""))
    except ValueError:
        qty = 1.0
    unit = m.group("unit").lower()
    unit = _UNITS.get(unit, unit)
    if unit == "flat rate" or qty <= 0:
        qty = 1.0
    return m.group("desc").strip(" |"), qty, unit


def normalize(desc: str) -> str:
    s = desc.lower()
    s = re.sub(r"\([^)]*\)?", " ", s)                 # qualifiers in brackets
    s = re.split(r"\s[-–—]\s", s)[0]        # "X - return visit", "X - living room"
    s = re.sub(r"\bmodel \d+\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def line_key(desc: str, unit: str = "") -> tuple[str, str | None]:
    """(line type, family). Family 'labour' pools the hourly rates of all grades."""
    low = desc.lower()
    n = normalize(desc)
    if "return visit" in low and "vehicle" in low:
        return "vehicle costs return visit", None
    if _ADMIN.search(re.sub(r"[^a-z]+", " ", low)):
        return "administrative or processing fee", None
    if n.endswith(" hours") and len(n) > 6:
        return "labour: " + n[:-6], "labour"
    if unit == "hr" and _GRADE.search(n):
        return "labour: " + n, "labour"
    return n, None


# ---------------------------------------------------------------- observations

@dataclass(frozen=True)
class Obs:
    game: int
    line: int
    key: str
    family: str | None
    text: str
    qty: float
    unit: str
    kind: str            # zero | bracket | lower | upper
    lo: float            # per unit; 0 if none
    hi: float            # per unit; inf if none
    policy: str = ""     # policy title of the case (c2f/pricebook.py::policy_title)


def _kind(it: dict) -> tuple[str, float, float] | None:
    lo, hi = float(it.get("t_low") or 0.0), it.get("t_high")
    if not any(v > 0 for v in it.get("charges", {}).values()):
        return "zero", 0.0, math.inf
    if lo > 0 and hi and hi > lo:
        return "bracket", lo, float(hi)
    if lo > 0:
        return "lower", lo, math.inf
    if hi:
        return "upper", 0.0, float(hi)
    return None


def observations(summary: dict | None = None, casebook: str | Path = CASEBOOK,
                 before_game: int | None = None) -> list[Obs]:
    """Every earlier invoice line with what the transactions revealed about its t.

    summary: output of c2f.market.write_summary ({game: {line: {...}}}); default: the file."""
    if summary is None:
        if not SUMMARY.exists():
            return []
        summary = json.loads(SUMMARY.read_text())
    out = []
    for g in sorted(int(k) for k in summary):
        if before_game is not None and g >= before_game:
            continue
        f = Path(casebook) / f"game{g}.json"
        if not f.exists():
            continue
        digest = json.loads(f.read_text(encoding="utf-8"))
        lines, policy = digest["lines"], digest.get("policy", "")
        items = summary.get(str(g), summary.get(g, {}))
        for i, it in sorted((int(k), v) for k, v in items.items()):
            text = lines.get(str(i))
            k = _kind(it)
            if not text or k is None:
                continue
            desc, qty, unit = parse_line(text)
            key, fam = line_key(desc, unit)
            if not key:
                continue
            kind, lo, hi = k
            out.append(Obs(g, i, key, fam, desc, qty, unit, kind, lo / qty, hi / qty, policy))
    return out


# ---------------------------------------------------------------- likelihood on the grid

def _log_sf(z):
    """log of the standard normal survival function, stable far into the upper tail."""
    z = np.asarray(z, dtype=float)
    zc = np.minimum(z, 30.0)
    near = np.log(np.maximum(0.5 * erfc(zc / math.sqrt(2.0)), 1e-300))
    zf = np.maximum(z, 30.0)
    far = -0.5 * zf * zf - np.log(zf * math.sqrt(2.0 * math.pi))
    return np.where(z < 30.0, near, far)


def _component(kind: str, lo: float, hi: float, mu, sigma, lam: float):
    """Likelihood of one positive-t observation under ln t ~ N(mu, sigma) (arrays broadcast)."""
    if kind == "bracket":
        return np.maximum(ndtr((math.log(hi) - mu) / sigma) - ndtr((math.log(lo) - mu) / sigma), 0.0)
    if kind == "upper":
        return ndtr((math.log(hi) - mu) / sigma)
    ell = math.log(lo)                                  # lower bound with an exponential gap
    return lam * np.exp(lam * (ell - mu) + 0.5 * (lam * sigma) ** 2
                        + _log_sf((ell - mu + lam * sigma ** 2) / sigma))


def _positive_lik(o_kind: str, lo: float, hi: float, mu, sigma, p: Params):
    main = _component(o_kind, lo, hi, mu, sigma, p.gap_rate)
    if p.eps <= 0:
        return main
    wide = _component(o_kind, lo, hi, mu + p.wide_shift, p.wide_sigma, p.gap_rate)
    return (1.0 - p.eps) * main + p.eps * wide


def event_lik(kind: str, lo: float, hi: float, p0, mu, sigma, p: Params):
    """Likelihood of one observation under (p0, mu, sigma); also scores any other belief."""
    if kind == "zero":
        return p0 + 0 * mu * sigma
    g = _positive_lik(kind, lo, hi, mu, sigma, p)
    if kind == "upper":
        return p0 + (1.0 - p0) * p.upper_pos * g
    return (1.0 - p0) * g


_SIG = np.geomspace(0.03, 1.0, 22)
_P0 = np.linspace(0.01, 0.99, 25)
_MU_STEP = 0.02


@dataclass
class Fit:
    """Posterior of one line type and what it predicts for the next line of that type."""
    key: str
    obs: list[Obs]
    params: Params
    borrowed_from: str | None = None       # set when the evidence is a similar line type's
    policy: str = ""                       # wording the fit was made for ("" = all pooled)
    n_same: int = 0                        # observations under that wording
    mu: np.ndarray = field(default=None, repr=False)
    post: np.ndarray = field(default=None, repr=False)       # [p0, mu, sigma], sums to 1
    resp_zero: list[float] = field(default_factory=list)     # P(t = 0 | data) per observation
    _cdf: tuple | None = field(default=None, repr=False)

    # --- summaries
    @property
    def n(self) -> int:
        return len(self.obs)

    @property
    def n_priced(self) -> int:
        return sum(o.lo > 0 for o in self.obs)

    @property
    def expected_zeros(self) -> float:
        return float(sum(self.resp_zero))

    @property
    def p_zero(self) -> float:
        return float((self.post.sum(axis=(1, 2)) * _P0).sum())

    def _w(self) -> np.ndarray:
        return self.post.sum(axis=0)                           # [mu, sigma]

    def quantiles(self, qs=(0.1, 0.5, 0.9)) -> list[float]:
        """Quantiles of t per unit given that the line is indemnified (main component)."""
        if self._cdf is None:
            w = self._w()
            wm = np.cumsum(w.sum(axis=1))
            i0, i1 = np.searchsorted(wm, 1e-5), min(np.searchsorted(wm, 1 - 1e-5) + 1, len(wm))
            mu, w = self.mu[i0:i1], w[i0:i1]
            smax = _SIG[np.nonzero(w.sum(axis=0) > 1e-6)[0][-1]]
            x = np.linspace(mu[0] - 3.5 * smax, mu[-1] + 3.5 * smax, 240)
            cdf = np.zeros_like(x)
            for j, sg in enumerate(_SIG):
                if w[:, j].sum() > 1e-6:
                    cdf += w[:, j] @ ndtr((x[None, :] - mu[:, None]) / sg)
            self._cdf = (x, cdf / max(cdf[-1], 1e-12))
        x, cdf = self._cdf
        return [float(math.exp(np.interp(q, cdf, x))) for q in qs]

    def level(self) -> tuple[float, float]:
        """(mean, sd) of ln t per unit under the predictive main component."""
        w = self._w()
        m = float((w.sum(axis=1) * self.mu).sum())
        var = float((w * ((self.mu[:, None] - m) ** 2 + _SIG[None, :] ** 2)).sum())
        return m, math.sqrt(var)

    def prob(self, kind: str, lo: float, hi: float) -> float:
        """Predictive probability of a new observation (per unit bounds), for scoring."""
        return float((self.post * event_lik(kind, lo, hi, _P0[:, None, None], self.mu[:, None],
                                            _SIG[None, :], self.params)).sum())


def fit(key: str, obs: list[Obs], params: Params = DEFAULT,
        mu_prior: tuple[float, float] | None = None, policy: str = "") -> Fit:
    """Grid posterior for one line type. mu_prior: (mean, sd) of a normal prior on mu.
    policy: wording of the case being priced; see `other_policy`."""
    n_same = sum(o.policy == policy for o in obs) if policy else 0
    wts = [1.0 if (not n_same or o.policy == policy) else params.other_policy for o in obs]
    pos = [o for o in obs if o.kind != "zero"]
    bounds = [b for o in pos for b in (o.lo, o.hi) if 0 < b < math.inf]
    if mu_prior is not None:
        bounds += [math.exp(mu_prior[0] - 2 * mu_prior[1]), math.exp(mu_prior[0] + 2 * mu_prior[1])]
    if not bounds:
        bounds = [1.0, 1000.0]
    lo_mu, hi_mu = math.log(min(bounds)) - 1.5, math.log(max(bounds)) + 1.5
    mu = np.arange(lo_mu, hi_mu + _MU_STEP, _MU_STEP)
    M, S = mu[:, None], _SIG[None, :]
    logprior_ms = -0.5 * ((np.log(S) - math.log(params.sigma_center)) / params.sigma_width) ** 2 + 0 * M
    if mu_prior is not None:
        logprior_ms = logprior_ms - 0.5 * ((M - mu_prior[0]) / mu_prior[1]) ** 2
    # Zeros and priced lines factorise in p0 and (mu, sigma); only "upper" lines couple them.
    n_zero = sum(w for o, w in zip(obs, wts) if o.kind == "zero")
    n_pos, uppers = 0.0, {}
    for i, (o, w) in enumerate(zip(obs, wts)):
        if o.kind == "zero":
            continue
        g = _positive_lik(o.kind, o.lo, o.hi, M, S, params)
        if o.kind == "upper":
            uppers[i] = params.upper_pos * g
        else:
            logprior_ms = logprior_ms + w * np.log(np.maximum(g, 1e-300))
            n_pos += w
    p0 = _P0[:, None, None]
    logpost = ((params.p0_a - 1 + n_zero) * np.log(p0) + (params.p0_b - 1 + n_pos) * np.log1p(-p0)
               + logprior_ms[None])
    for i, g in uppers.items():
        logpost = logpost + wts[i] * np.log(p0 + (1.0 - p0) * g[None])
    post = np.exp(logpost - logpost.max())
    post /= post.sum()
    resp = [1.0 if o.kind == "zero" else 0.0 if i not in uppers
            else float((post * p0 / (p0 + (1.0 - p0) * uppers[i][None])).sum())
            for i, o in enumerate(obs)]
    return Fit(key, list(obs), params, policy=policy if n_same else "", n_same=n_same,
               mu=mu, post=post, resp_zero=resp)


# ---------------------------------------------------------------- the model

def _point(o: Obs) -> float:
    return math.log(math.sqrt(o.lo * o.hi) if o.hi < math.inf else o.lo)


def deflated_games(obs: list[Obs], params: Params = DEFAULT) -> dict[int, float]:
    """Games whose recurring lines sit jointly far below the level the same line types have
    in other games: {game: median log deviation}. Sign of a cap shared across the invoice."""
    priced: dict[str, list[Obs]] = {}
    for o in obs:
        if o.lo > 0:
            priced.setdefault(o.key, []).append(o)
    devs: dict[int, list[float]] = {}
    for os_ in priced.values():
        for o in os_:
            others = [_point(x) for x in os_ if x.game != o.game]
            if others:
                devs.setdefault(o.game, []).append(_point(o) - float(np.median(others)))
    return {g: float(np.median(d)) for g, d in devs.items()
            if len(d) >= params.deflated_min_lines and np.median(d) < params.deflated_shift}


def _words(key: str) -> tuple[frozenset, frozenset]:
    toks = key.replace("labour: ", "").split()
    return (frozenset(t for t in toks if not any(c.isdigit() for c in t)),
            frozenset(t for t in toks if any(c.isdigit() for c in t)))


class ItemPrior:
    """All earlier observations grouped by line type; fits are made on demand and cached."""

    def __init__(self, obs: list[Obs], params: Params = DEFAULT):
        self.params = params
        self.by_key: dict[str, list[Obs]] = {}
        self.by_family: dict[str, list[Obs]] = {}
        self.deflated = deflated_games(obs, params)
        for o in obs:
            if o.lo > 0 and o.game in self.deflated:
                continue                        # indemnified, but the amount says nothing
            self.by_key.setdefault(o.key, []).append(o)
            if o.family:
                self.by_family.setdefault(o.family, []).append(o)
        self._fits: dict[str, Fit | None] = {}
        self._family: dict[str, tuple[float, float] | None] = {}

    def family_level(self, family: str) -> tuple[float, float] | None:
        if family not in self._family:
            obs = [o for o in self.by_family.get(family, []) if o.lo > 0]
            if len(obs) < 3:
                self._family[family] = None
            else:
                # pooled over grades: a wide sigma prior, so the spread between grades is kept
                p = Params(**{**self.params.__dict__, "sigma_center": 0.3})
                m, s = fit(family, obs, p).level()
                self._family[family] = (m, max(s, 0.2) * self.params.family_spread)
        return self._family[family]

    def _similar(self, key: str) -> str | None:
        if self.params.fuzzy <= 0:
            return None
        words, nums = _words(key)
        best, best_j = None, 0.0
        for k, obs in self.by_key.items():
            if k == key or k.startswith("labour: ") != key.startswith("labour: "):
                continue
            w, n = _words(k)
            if n != nums or not words or not w:
                continue
            j = len(words & w) / len(words | w)
            if j > best_j or (j == best_j and best and len(obs) > len(self.by_key[best])):
                best, best_j = k, j
        return best if best_j >= self.params.fuzzy and not key.startswith("labour: ") else None

    def lookup(self, desc: str, unit: str = "", policy: str = "") -> Fit | None:
        """Prior for a line, or None if the line type has too little history.
        policy: policy title of the case being priced ('' pools all wordings)."""
        key, fam = line_key(desc, unit)
        ck = f"{key}|{fam}|{policy}"
        if ck in self._fits:
            return self._fits[ck]
        obs, borrowed = self.by_key.get(key, []), None
        if not obs:
            borrowed = self._similar(key)
            obs = self.by_key.get(borrowed, []) if borrowed else []
        mu_prior = self.family_level(fam) if fam else None
        priced = sum(o.lo > 0 for o in obs)
        enough = len(obs) >= self.params.min_obs or priced >= self.params.min_priced
        if fam and mu_prior is not None and not obs:
            enough = True                       # an unseen grade still gets the family level
        f = None
        if enough:
            f = fit(key, obs, self.params, mu_prior, policy)
            f.borrowed_from = borrowed
        self._fits[ck] = f
        return f


def load(before_game: int | None = None, params: Params = DEFAULT, summary: dict | None = None,
         casebook: str | Path = CASEBOOK) -> ItemPrior:
    return ItemPrior(observations(summary, casebook, before_game), params)


# ---------------------------------------------------------------- prompt text

def _eur(x: float) -> str:
    return f"{x:.0f}" if x >= 20 else f"{x:.1f}" if x >= 2 else f"{x:.2f}"


def describe(f: Fit, qty: float, unit: str) -> str:
    """One prompt row for a line with quantity qty: zero history and price range."""
    n, ez = f.n, f.expected_zeros
    zeros = round(ez)
    src = f" (history of the similar line type '{f.borrowed_from}')" if f.borrowed_from else ""
    if n == 0:
        hist = "no earlier line of this grade; level of the other hourly grades"
    elif zeros == 0:
        hist = f"indemnified in all {n} earlier cases" if n > 1 else "indemnified in the one earlier case"
    elif zeros >= n:
        hist = (f"NOT indemnified (t = 0) in all {n} earlier cases" if n > 1
                else "NOT indemnified (t = 0) in the one earlier case")
    else:
        hist = f"indemnified in {n - zeros} of {n} earlier cases, t = 0 in {zeros}"
    if f.n_priced == 0 and n > 0:
        return f"{hist}{src}; no price evidence"
    q10, q50, q90 = f.quantiles()
    per = unit if unit and unit != "flat rate" else "unit"
    row = f"{hist}{src}; when indemnified t per {per} about {_eur(q50)} (80% range {_eur(q10)} to {_eur(q90)})"
    seen = sorted((o for o in f.obs if o.lo > 0), key=lambda o: (o.policy == f.policy, o.game))[-6:]
    if seen:
        other = [o for o in seen if f.policy and o.policy != f.policy]
        row += f"; revealed per {per}: " + ", ".join(
            (f"{_eur(o.lo)}-{_eur(o.hi)}" if o.hi < math.inf else f">={_eur(o.lo)}")
            + f" (case {o.game}{'*' if o in other else ''})" for o in seen)
        if other:
            row += " (* = other policy wording)"
    if qty != 1:
        row += f", so this line ({qty:g} {unit}) about {_eur(q50 * qty)} ({_eur(q10 * qty)} to {_eur(q90 * qty)})"
    return row


HEADER = """PRICE PRIORS for recurring line types (statistical model fitted on the thresholds t revealed
in earlier games; gross amounts). How to use them:
- The amounts are final thresholds as they were revealed: they already contain whatever
  deductible, VAT and customary-price judgement applied in those cases, and they were computed
  from the full invoiced quantity (the whole quantity was indemnified there). For an
  indemnified line of this type take the stated amount for the line as t as it stands and give
  a price_log_sd that matches the stated range. Do not deduct a deductible from it and do not
  cut the invoiced quantity.
- Deviate only for something this case has and those cases did not: an aggregate cap or
  sub-limit that the lines of this invoice must share, a percentage reduction for a breached
  duty, or a purchase price or sum insured stated in the claim that limits the line.
- Whether a line is indemnified in THIS case is decided by the policy and the claim, not by
  the history. The history only says how often this line type was paid at all. A line type
  that was never indemnified before is usually an ancillary charge the policies do not carry;
  treat it as indemnified only if this policy clearly says so."""


def context(lines: dict[int, str], prior: ItemPrior | None = None, before_game: int | None = None,
            policy: str = "") -> str:
    """Prompt block for the invoice lines {POS: 'description | quantity unit'} ('' if nothing
    is known about any of them). policy: policy title of the case (pricebook.policy_title)."""
    prior = prior if prior is not None else load(before_game)
    rows = []
    for pos, text in sorted(lines.items()):
        desc, qty, unit = parse_line(text)
        f = prior.lookup(desc, unit, policy)
        if f is None:
            continue
        rows.append(f"- line {pos} \"{desc[:70]}\": {describe(f, qty, unit)}")
    return HEADER + "\n" + "\n".join(rows) if rows else ""


# ---------------------------------------------------------------- combining with an estimate

def combine(median: float, sigma: float, f: Fit, qty: float, mode: str = "fuse",
            guard: float = 1.0) -> tuple[float, float] | None:
    """An estimate of the line total (median, log-sd) combined with the prior of its line
    type; None leaves the estimate alone.

    mode "fuse": precision-weighted mean in logs (the estimate and the prior as two
    independent measurements); "replace": the prior's level and spread.
    guard: the estimate is only touched when it is within exp(guard) of the prior. A larger
    gap usually means the assessor applied a cap or reduction the prior knows nothing about
    (game 41), or found the line type priced differently in this kind of case."""
    if f is None or median <= 0 or not (f.n_priced or f.n == 0):
        return None
    mu, sd = f.level()
    mu, sd = mu + math.log(qty), max(sd, 0.1)
    if abs(math.log(median) - mu) > guard:
        return None
    if mode == "replace":
        return math.exp(mu), sd
    wl, wp = 1.0 / max(sigma, 0.05) ** 2, 1.0 / sd ** 2
    return math.exp((wl * math.log(median) + wp * mu) / (wl + wp)), max(math.sqrt(1.0 / (wl + wp)), 0.1)


_lock = threading.Lock()
_shared: dict = {}


def shared() -> ItemPrior:
    """The prior on everything revealed so far, kept until the transaction summary changes.
    The stages of one game (fast, full, backup) run in parallel and ask for the same line
    types, so they share one instance and its fits."""
    stamp = SUMMARY.stat().st_mtime if SUMMARY.exists() else 0.0
    with _lock:
        if _shared.get("stamp") != stamp:
            _shared.update(stamp=stamp, prior=load(), ctx={})
        return _shared["prior"]


def context_for_case(case) -> str:
    """PRICE PRIORS block for a case ('' if no line has history); built once per case."""
    from . import pricebook
    lines = pricebook.invoice_lines(case.invoice_text)
    policy = pricebook.policy_title(case.policy)
    prior = shared()
    key = (tuple(sorted(lines.items())), policy)
    with _lock:
        ctx = _shared["ctx"]
        if key not in ctx:
            ctx[key] = context(lines, prior, policy=policy)
        return ctx[key]


def combine_for_case(items: list, case, mode: str) -> int:
    """Belief-level use of the prior: combine every estimate the assessor believes indemnified
    with the prior of its line type (see `combine`). items: c2f.extract.ItemEstimate, changed
    in place. Returns the number of lines changed."""
    from . import pricebook
    lines = pricebook.invoice_lines(case.invoice_text)
    policy = pricebook.policy_title(case.policy)
    prior, changed = shared(), 0
    for k, it in enumerate(items, 1):
        m = re.match(r"\s*(\d+)", str(it.item_id))
        text = lines.get(int(m.group(1)) if m else k)
        if not text or it.covered_prob * it.related_prob < 0.5 or it.quantity <= 0:
            continue
        desc, qty, unit = parse_line(text)
        with _lock:
            f = prior.lookup(desc, unit, policy)
        out = combine(it.quantity * it.unit_price_gross_eur, it.price_log_sd, f, qty, mode)
        if out is not None:
            it.unit_price_gross_eur, it.price_log_sd = out[0] / it.quantity, out[1]
            changed += 1
    return changed
