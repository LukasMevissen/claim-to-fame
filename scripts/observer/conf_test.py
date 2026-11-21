"""Live pricing rule vs the confidence-aware one (min_sigma lowered, charge_conf_k) on stored
single-call runs. Only lines whose stated sd is below the old min_sigma can change."""
import sys, json, dataclasses
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.config import load_config
from c2f.extract import ItemEstimate
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
            if s and i['price_log_sd'] < 0.2 and i['covered_prob'] * i['related_prob'] >= 0.5:
                rows.append((d.parent.name + '/' + d.name, g, i, s))
print('lines with stated sd < 0.2:', len(rows), 'in games', sorted({r[1] for r in rows}))
def q(i, min_sigma, k):
    m = i['quantity'] * i['unit_price_gross_eur']; p = i['covered_prob'] * i['related_prob']
    r = quote(TBelief(m, min(max(i['price_log_sd'], min_sigma), 1.2), 1 - p), dataclasses.replace(P, charge_conf_k=k)); return r.charge_price, r.acceptance_limit
for ms, k in ((0.2, 0.0), (0.05, 0.0), (0.05, 0.6), (0.05, 0.8), (0.05, 1.0), (0.05, 1.3), (0.1, 0.8)):
    tot = 0; worst = 0; per = {}
    for name, g, i, s in rows:
        v = net(*q(i, ms, k), s); tot += v; per[g] = per.get(g, 0) + v
    print(f'min_sigma {ms} conf_k {k}: total {tot:9.0f}   per game {dict(sorted((g, round(v)) for g, v in per.items()))}', flush=True)
