"""Confidence-dependent pricing: a = alpha(sd) * m, b = beta(sd) * m, with alpha, beta chosen
per bucket of the model's stated uncertainty sd. Fitted on half of the games, scored on the
other half (and vice versa), against the live pricing rule. No API calls."""
import sys, json, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.config import load_config
from c2f.extract import ItemEstimate
from c2f.agent import to_belief
from c2f.pricing import quote
from c2f.replay import load_summary, item_net, t_point
cfg = load_config(); S = load_summary(); P = cfg.pricing
BUCKETS = [(0, 0.15), (0.15, 0.35), (0.35, 9)]
A = [0.6, 0.7, 0.8, 0.87, 0.92, 0.96, 1.0, 1.05]; B = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35]
lines = []   # (run, game, bucket, m, p, s, live_a, live_b)
qc = {}
for d in sorted(Path('/tmp/c2f_work/prompt_exp').glob('v0/run*')):
    for f in d.glob('game*.json'):
        r = json.load(open(f)); g = r['game']
        for i in r['items']:
            s = S.get(str(g), {}).get(str(i['item_id']).strip())
            if not s: continue
            it = ItemEstimate(**i); m = it.quantity * it.unit_price_gross_eur; p = it.covered_prob * it.related_prob
            k = (round(m, 2), it.price_log_sd, round(p, 3))
            if k not in qc: q = quote(to_belief(it, cfg), P); qc[k] = (q.charge_price, q.acceptance_limit)
            bk = next(n for n, (lo, hi) in enumerate(BUCKETS) if lo <= it.price_log_sd < hi)
            lines.append((d.name, g, bk, m, p, s) + qc[k])
print('line evaluations', len(lines), 'per bucket', [sum(l[2] == b for l in lines) for b in range(3)], flush=True)
def net(a, b, s): return (item_net(a, b, s, t_point(s, 1.15)) + item_net(a, b, s, t_point(s, 1.4))) / 2
def val(l, al, be):
    _, g, bk, m, p, s, la, lb = l
    if p < 0.5: return net(la, 0.0, s)                    # uncovered call: keep live behaviour
    return net(al * m, be * m, s)
def fit(sub):
    out = {}
    for bk in range(3):
        ls = [l for l in sub if l[2] == bk]
        out[bk] = max(itertools.product(A, B), key=lambda ab: sum(val(l, *ab) for l in ls)) if ls else (0.87, 0.9)
    return out
def score(sub, pol): return sum(val(l, *pol[l[2]]) for l in sub)
live = lambda sub: sum(net(l[6], l[7], l[5]) for l in sub)
even = [l for l in lines if l[1] % 2 == 0]; odd = [l for l in lines if l[1] % 2 == 1]
pe, po, pa = fit(even), fit(odd), fit(lines)
print('fit on even games:', pe); print('fit on odd games :', po); print('fit on all       :', pa)
print(f'out of sample: odd games  live {live(odd):9.0f}  policy(fit even) {score(odd, pe):9.0f}')
print(f'out of sample: even games live {live(even):9.0f}  policy(fit odd)  {score(even, po):9.0f}')
print(f'in sample all: live {live(lines):9.0f} policy {score(lines, pa):9.0f}')
for bk in range(3):
    ls = [l for l in lines if l[2] == bk]
    print(f'bucket sd {BUCKETS[bk]}: live {live(ls):9.0f}  policy(all) {score(ls, pa):9.0f}  n={len(ls)}')
    row = {al: round(sum(val(l, al, pa[bk][1]) for l in ls)) for al in A}; print('    alpha sweep (beta fixed):', row)
    row = {be: round(sum(val(l, pa[bk][0], be) for l in ls)) for be in B}; print('    beta sweep (alpha fixed):', row)
