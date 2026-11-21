"""Confident lines only (stated sd < 0.12): live rule vs trusting the stated sd for the limit
and charging a fixed high share of the estimate. All stored runs, no API calls."""
import sys, json, dataclasses
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.config import load_config
from c2f.belief import TBelief
from c2f.pricing import quote
from c2f.replay import load_summary, item_net, t_point
cfg = load_config(); S = load_summary(); P = cfg.pricing
def net(a, b, s): return (item_net(a, b, s, t_point(s, 1.15)) + item_net(a, b, s, t_point(s, 1.4))) / 2
rows = []
for d in sorted(Path('/tmp/c2f_work/prompt_exp').glob('*/run*')):
    for f in d.glob('game*.json'):
        r = json.load(open(f)); g = r['game']
        for i in r['items']:
            s = S.get(str(g), {}).get(str(i['item_id']).strip())
            if s and i['price_log_sd'] < 0.12 and i['covered_prob'] * i['related_prob'] >= 0.5: rows.append((g, i, s))
print('confident line evaluations:', len(rows), 'games', sorted({r[0] for r in rows}))
for g, i, s in rows[:0]: pass
def run(sig_floor, ratio):
    per = {}
    for g, i, s in rows:
        m = i['quantity'] * i['unit_price_gross_eur']; p = i['covered_prob'] * i['related_prob']
        q = quote(TBelief(m, max(i['price_log_sd'], sig_floor), 1 - p), P)
        a = max(q.charge_price, ratio * m); per[g] = per.get(g, 0) + net(a, q.acceptance_limit, s)
    return per
base = run(0.2, 0.0)
print(f"{'variant':34s} total   games up/down vs live   worst game change")
for sf in (0.2, 0.1, 0.06):
    for ratio in (0.0, 0.90, 0.93, 0.95, 0.97, 1.0):
        per = run(sf, ratio); d = {g: per[g] - base[g] for g in per}
        print(f'sigma floor {sf:4} charge >= {ratio:4} x m: {sum(per.values()):8.0f}   {sum(v > 1 for v in d.values())}/{sum(v < -1 for v in d.values())}   {min(d.values()):8.0f}', flush=True)
