"""Score the subscription test: arm A (one pricing step) vs arm B (pricing step fed with
research notes on all earlier cases and stated amounts). Same eight games, same pricing rules."""
import sys, json, math, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.config import load_config
from c2f.extract import _parse
from c2f.agent import price_items
from c2f.replay import load_summary, game_net
cfg = load_config(); S = load_summary(); OUT = Path('/tmp/c2f_work/subtest/out')
res = {}
for arm in ('A', 'B'):
    for f in sorted((OUT / arm).glob('game*.json')):
        g = int(f.stem[4:])
        try: items = _parse(f.read_text())
        except Exception as e: print('unreadable', arm, g, repr(e)[:100]); continue
        q = {int(p.item_id): (p.charge_price, p.acceptance_limit) for p in price_items(items, cfg)}
        errs = []; cov_ok = cov_n = 0
        for it in items:
            s = S.get(str(g), {}).get(str(it.item_id).strip())
            if not s: continue
            paid = any(v > 0 for v in s['charges'].values()); p = it.covered_prob * it.related_prob
            if s['t_low'] > 0 or not paid:
                cov_n += 1; cov_ok += (p >= 0.5) == (s['t_low'] > 0)
            if s['t_low'] > 0 and p >= 0.5:
                m = it.quantity * it.unit_price_gross_eur; lo, hi = s['t_low'], s['t_high']
                errs.append(math.log(m / lo) if m < lo else (math.log(m / hi) if hi and m > hi else 0.0))
        res[(arm, g)] = (game_net(q, g, S, 1.15), game_net(q, g, S, 1.4), errs, cov_ok, cov_n)
games = sorted({g for _, g in res})
print('game      net A (1.15 / 1.4)      net B (1.15 / 1.4)    mean |price error| A / B   coverage right A / B')
tot = {'A': [0, 0], 'B': [0, 0]}; allerr = {'A': [], 'B': []}
for g in games:
    if ('A', g) not in res or ('B', g) not in res: print(g, 'missing an arm'); continue
    a, b = res[('A', g)], res[('B', g)]
    for arm, r in (('A', a), ('B', b)): tot[arm][0] += r[0]; tot[arm][1] += r[1]; allerr[arm] += r[2]
    ea = st.mean(abs(e) for e in a[2]) if a[2] else float('nan'); eb = st.mean(abs(e) for e in b[2]) if b[2] else float('nan')
    print(f'{g:4d}   {a[0]:9.0f} / {a[1]:9.0f}   {b[0]:9.0f} / {b[1]:9.0f}        {ea:5.2f} / {eb:5.2f}              {a[3]}/{a[4]}  {b[3]}/{b[4]}')
print(f"TOTAL  {tot['A'][0]:9.0f} / {tot['A'][1]:9.0f}   {tot['B'][0]:9.0f} / {tot['B'][1]:9.0f}")
for arm in 'AB':
    e = allerr[arm]
    if e: print(f'arm {arm}: {len(e)} priced covered lines, mean |error| {st.mean(abs(x) for x in e):.3f}, too low {sum(x < 0 for x in e)}, too high {sum(x > 0 for x in e)}, inside {sum(x == 0 for x in e)}')
