"""Compare charge floors on every stored run (no API calls): fixed share of the estimate
versus a quantile of the belief, which lowers the floor when the model is unsure."""
import sys, json, glob, dataclasses
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.config import load_config
from c2f.extract import ItemEstimate
from c2f.agent import to_belief
from c2f.pricing import quote
from c2f.replay import load_summary, game_net
cfg = load_config(); S = load_summary(); base = cfg.pricing
runs = {}
for d in sorted(Path('/tmp/c2f_work/prompt_exp').glob('*/run*')):
    runs[f'opus {d.parent.name}/{d.name}'] = {json.load(open(f))['game']: json.load(open(f))['items'] for f in d.glob('game*.json')}
for model in ('claude-fable-5-2', 'claude-opus-5-5', 'claude-sonnet-5-5', 'gpt-6.2-sol'):
    by = {}
    for f in glob.glob(str(ROOT / f'results/shadow/game*__{model}__*.json')):
        r = json.load(open(f))
        if r['ok']: by[r['game']] = r['items']
    runs[f'shadow {model}'] = by
VARS = [('ratio .87 (live)', dict(charge_min_ratio=0.87, charge_min_quantile=0.0)), ('none', dict(charge_min_ratio=0.0, charge_min_quantile=0.0)),
        ('q .20', dict(charge_min_ratio=0.0, charge_min_quantile=0.20)), ('q .25', dict(charge_min_ratio=0.0, charge_min_quantile=0.25)),
        ('q .30', dict(charge_min_ratio=0.0, charge_min_quantile=0.30)), ('q .35', dict(charge_min_ratio=0.0, charge_min_quantile=0.35))]
tot = {n: 0.0 for n, _ in VARS}; worst = {n: 0.0 for n, _ in VARS}
print(f"{'run':28s} n  " + '  '.join(f'{n:>15s}' for n, _ in VARS))
for name, by in runs.items():
    row = []
    for n, kw in VARS:
        P = dataclasses.replace(base, **kw); t = 0.0
        for g, items in by.items():
            if str(g) not in S: continue
            q = {}
            for i in items:
                it = ItemEstimate(**i); r = quote(to_belief(it, cfg), P); q[int(it.item_id)] = (r.charge_price, r.acceptance_limit)
            v = (game_net(q, g, S, 1.15) + game_net(q, g, S, 1.4)) / 2; t += v; worst[n] = min(worst[n], v)
        tot[n] += t; row.append(t)
    print(f"{name:28s} {len(by):2d} " + '  '.join(f'{x:15.0f}' for x in row), flush=True)
print(f"{'SUM':28s}    " + '  '.join(f'{tot[n]:15.0f}' for n, _ in VARS))
print(f"{'worst single game':28s}    " + '  '.join(f'{worst[n]:15.0f}' for n, _ in VARS))
