"""Counterfactual charge rules on the live games 10-24 (game 21 excluded: model failed, a = 20).

Our limit b stays as submitted; only the charge a changes. Uses c2f/replay.py.
    python scripts/observer/charge_counterfactual.py
"""
import json, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.replay import load_summary, t_point, item_net
S = load_summary()
subs = {}
for l in open(ROOT/'runs/submissions.jsonl'):
    r = json.loads(l)
    if not r.get('dry_run') and r.get('ok') and r.get('payload'): subs[r['game_id']] = r
games = [g for g in range(10, 25) if g != 21 and g in subs and str(g) in S]
est = {}
for g in games:
    p = ROOT/f'results/extractions/game{g}__x3__gpt-6.2-sol.json'
    est[g] = {int(i['item_id']): i for i in json.load(open(p))['items'] if str(i['item_id']).isdigit()}
def run(rule, up):
    tot = 0.0
    for g in games:
        pay = {p['index']: (p['charge_price'], p['acceptance_limit']) for p in subs[g]['payload']}
        for i, s in S[str(g)].items():
            i = int(i); a, b = pay.get(i, (0.0, 0.0)); t = t_point(s, up)
            it = est[g].get(i); m = it['quantity']*it['unit_price_gross_eur'] if it else a/0.87
            tot += item_net(rule(a, m, t), b, s, t)
    return tot
print('games', games)
rules = {'actual (live a)': lambda a,m,t: a, 'a = 0.87 m (floor everywhere)': lambda a,m,t: 0.87*m,
         'a = 1.0 m (no shade)': lambda a,m,t: m, 'a = 1.15 m': lambda a,m,t: 1.15*m, 'a = 1.3 m': lambda a,m,t: 1.3*m,
         'a = t (oracle, "Nash")': lambda a,m,t: t if t > 0 else a, 'a = 0.95 t (oracle)': lambda a,m,t: 0.95*t if t>0 else a}
for up in (1.15, 1.4):
    base = run(rules['actual (live a)'], up)
    print(f'\nt reading up={up}:')
    for k, f in rules.items():
        v = run(f, up); print(f'  {k:32s} {v:10,.0f}   ({v-base:+,.0f} vs actual)')
