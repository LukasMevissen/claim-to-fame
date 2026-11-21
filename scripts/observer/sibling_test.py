"""Sibling-line rule, tested on stored single-call runs (no API calls).

A line whose text and quantity (almost) equal a line of an earlier game gets that line's
revealed bounds: both bounds known -> estimate = window midpoint, treated as confident;
only a proven-fair level known -> charge at least 0.95 x and accept up to about that level.
Compared with the live pricing of the same stored model estimates."""
import sys, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.config import load_config
from c2f.extract import ItemEstimate
from c2f.agent import to_belief
from c2f.pricing import quote
from c2f.replay import load_summary, item_net, t_point
cfg = load_config(); S = load_summary(); P = cfg.pricing
CB = {int(f.stem[4:]): json.load(open(f)) for f in (ROOT / 'results/casebook').glob('game*.json') if f.stem[4:].isdigit()}
STOP = {'the', 'and', 'for', 'from', 'with', 'incl', 'of', 'to', 'in', 'on', 'a', 'an'}
def key(text):
    m = re.search(r'\|?\s*([\d.,]+)\s+(pcs|hrs|flat rate|m2|m|lines|days|kg|l)\s*$', text.strip())
    qty = (m.group(1), m.group(2)) if m else None
    body = text[:m.start()] if m else text
    return frozenset(w for w in re.findall(r'[a-z]{3,}', body.lower()) if w not in STOP), qty
def siblings(g, line_text, min_sim):
    tok, qty = key(line_text); lo = 0.0; hi = None; src = []
    for h, d in CB.items():
        if h >= g or str(h) not in S: continue
        for idx, t2 in d['lines'].items():
            tk, q2 = key(t2); s = S[str(h)].get(str(idx))
            if not s or not tok or not tk or q2 != qty: continue
            if len(tok & tk) / len(tok | tk) < min_sim: continue
            if not any(v > 0 for v in s['charges'].values()): continue
            src.append(h)
            if s['t_low'] > lo: lo = s['t_low']
            if s['t_high'] and (hi is None or s['t_high'] < hi): hi = s['t_high']
    return lo, hi, src
def net(a, b, s): return (item_net(a, b, s, t_point(s, 1.15)) + item_net(a, b, s, t_point(s, 1.4))) / 2
runs = sorted(Path('/tmp/c2f_work/prompt_exp').glob('v0/run*'))
for min_sim in (1.0, 0.8, 0.6):
    tot_live = tot_new = 0.0; n = 0; per = {}; detail = []
    for d in runs:
        for f in d.glob('game*.json'):
            r = json.load(open(f)); g = r['game']
            if g not in CB: continue
            for i in r['items']:
                pos = str(i['item_id']).strip(); s = S.get(str(g), {}).get(pos); text = CB[g]['lines'].get(pos)
                if not s or not text: continue
                it = ItemEstimate(**i); m = it.quantity * it.unit_price_gross_eur; p = it.covered_prob * it.related_prob
                q = quote(to_belief(it, cfg), P); live = net(q.charge_price, q.acceptance_limit, s)
                lo, hi, src = siblings(g, text, min_sim)
                if not src or p < 0.5 or lo <= 0:
                    continue
                if hi and lo < hi: a, b = 0.95 * (lo + hi) / 2, 0.96 * (lo + hi) / 2 if False else (lo + hi) / 2 * 0.97
                elif hi: a, b = 0.95 * min(lo, hi), min(lo, hi)
                else: a, b = max(q.charge_price, 0.95 * lo), max(q.acceptance_limit, lo)
                new = net(a, b, s); n += 1; tot_live += live; tot_new += new; per[g] = per.get(g, 0) + new - live
                if d.name == 'run1': detail.append((g, pos, round(m), (lo, hi), (s['t_low'], s['t_high']), round(live), round(new)))
    print(f'min similarity {min_sim}: line evaluations {n}  live {tot_live:9.0f}  sibling rule {tot_new:9.0f}  change {tot_new - tot_live:+9.0f}  games up/down {sum(v > 1 for v in per.values())}/{sum(v < -1 for v in per.values())}  worst game {min(per.values()) if per else 0:8.0f}', flush=True)
    if min_sim == 0.8:
        for x in detail: print('   game %2d line %2s  model est %5d  sibling window %s  truth %s  live %6d -> %6d' % x)
