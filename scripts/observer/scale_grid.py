"""Scale our charge a and limit b by constant factors on stored estimates; no API calls.
Rows: every stored run (prompt experiments on Opus, live model). Shows whether a systematic
shift of a or b would have paid, and whether runs agree."""
import sys, json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.config import load_config
from c2f.extract import ItemEstimate
from c2f.agent import price_items
from c2f.replay import load_summary, game_net
cfg = load_config(); S = load_summary()
runs = {}
for d in sorted(Path('/tmp/c2f_work/prompt_exp').glob('*/run*')):
    runs[f'opus {d.parent.name}/{d.name}'] = {json.load(open(f))['game']: json.load(open(f))['items'] for f in d.glob('game*.json')}
sol = {}
for f in glob.glob(str(ROOT / 'results/shadow/game*__gpt-6.2-sol__*.json')):
    r = json.load(open(f)); sol[r['game']] = r['items']
for f in glob.glob(str(ROOT / 'results/extractions/game*__x3__gpt-6.2-sol.json')):
    r = json.load(open(f)); g = r['case_id'].replace('game', '')
    if g.isdigit() and int(g) not in sol: sol[int(g)] = r['items']
runs['sol live'] = sol
KA = (0.9, 1.0, 1.05, 1.1, 1.2); KB = (0.9, 1.0, 1.1, 1.2, 1.35, 1.5)
agg = {}
for name, by in runs.items():
    Q = {}
    for g, items in by.items():
        if str(g) in S: Q[g] = {int(p.item_id): (p.charge_price, p.acceptance_limit) for p in price_items([ItemEstimate(**i) for i in items], cfg)}
    def tot(ka, kb): return sum((game_net({i: (a * ka, b * kb) for i, (a, b) in q.items()}, g, S, 1.15) + game_net({i: (a * ka, b * kb) for i, (a, b) in q.items()}, g, S, 1.4)) / 2 for g, q in Q.items())
    base = tot(1, 1)
    print(f'{name:18s} n={len(Q)} base {base:7.0f} | a x{KA}: {[round(tot(k, 1) - base) for k in KA]} | b x{KB}: {[round(tot(1, k) - base) for k in KB]}', flush=True)
