"""Are our estimates systematically low on expensive lines? Stored estimates only, no API calls.
Part 1: estimate / proven-fair level by size of the estimate.
Part 2: what scaling charge and limit up on large lines would have earned (replay)."""
import sys, json, glob, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.config import load_config
from c2f.extract import ItemEstimate
from c2f.agent import to_belief
from c2f.pricing import quote
from c2f.replay import load_summary, item_net, t_point
cfg = load_config(); S = load_summary(); P = cfg.pricing
def load_sol():
    by = {}
    for f in glob.glob(str(ROOT / 'results/shadow/game*__gpt-6.2-sol__*.json')):
        r = json.load(open(f)); by[r['game']] = r['items']
    for f in glob.glob(str(ROOT / 'results/extractions/game*__x3__gpt-6.2-sol.json')):
        r = json.load(open(f)); g = r['case_id'].replace('game', '')
        if g.isdigit() and int(g) > 0 and int(g) not in by: by[int(g)] = r['items']
    return by
sources = {'sol games 1-36': [load_sol()],
           'opus games 1-30 (3 runs)': [{json.load(open(f))['game']: json.load(open(f))['items'] for f in d.glob('game*.json')} for d in sorted(Path('/tmp/c2f_work/prompt_exp').glob('v0/run*'))]}
BK = [(0, 100), (100, 300), (300, 1000), (1000, 1e9)]
for name, runs in sources.items():
    rows = []
    for by in runs:
        for g, items in by.items():
            for i in items:
                s = S.get(str(g), {}).get(str(i['item_id']).strip())
                if not s or s['t_low'] <= 0 or i['covered_prob'] * i['related_prob'] < 0.5: continue
                it = ItemEstimate(**i); m = it.quantity * it.unit_price_gross_eur; q = quote(to_belief(it, cfg), P)
                rows.append((m, s, q.charge_price, q.acceptance_limit))
    print(f'== {name}: {len(rows)} covered lines with a proven-fair level')
    for lo, hi in BK:
        x = [r for r in rows if lo <= r[0] < hi]
        if not x: continue
        two = [r for r in x if r[1]['t_high']]
        print(f"  estimate {lo:>4}-{hi if hi < 1e8 else 'up':<5}: n={len(x):3d}  median estimate/proven-fair level {st.median(r[0] / r[1]['t_low'] for r in x):.2f}  "
              f"estimate below that level: {sum(r[0] < r[1]['t_low'] for r in x)}  above the proven-inflated level: {sum(r[0] > r[1]['t_high'] for r in two)} of {len(two)} with an upper bound")
    def net(a, b, s, up): return item_net(a, b, s, t_point(s, up))
    for thr in (300, 1000):
        big = [r for r in rows if r[0] >= thr]
        print(f'  lines with estimate >= {thr}: n={len(big)}; replay net if charge and limit are both scaled (t assumed 1.15x / 1.4x the lower bound where only that is known):')
        for k in (1.0, 1.05, 1.1, 1.15, 1.25):
            print(f"     x{k:<4}: {sum(net(a * k, b * k, s, 1.15) for _, s, a, b in big):9.0f} / {sum(net(a * k, b * k, s, 1.4) for _, s, a, b in big):9.0f}   limit only: {sum(net(a, b * k, s, 1.15) for _, s, a, b in big):9.0f} / {sum(net(a, b * k, s, 1.4) for _, s, a, b in big):9.0f}")
