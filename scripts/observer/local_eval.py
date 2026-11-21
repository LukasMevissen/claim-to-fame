"""Replay the no-model fallbacks on past games (price book of earlier games only)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.config import load_config
from c2f.agent import price_items
from c2f.case_io import load_case
from c2f.local_prices import estimate_lines
from c2f.pricebook import invoice_lines
from c2f.replay import load_summary, game_net
cfg = load_config(); S = load_summary(); CASES = Path('/tmp/c2f_work/cases')
G = [g for g in range(1, 60) if str(g) in S and (CASES / f'game{g}').exists()]
texts = {g: load_case(CASES / f'game{g}', f'game{g}').invoice_text for g in G}
def total(a0, b0, use_local, min_sim=0.5):
    tot = 0.0; matched = n = 0
    for g in G:
        q = {i: (a0, b0) for i in invoice_lines(texts[g])}; n += len(q)
        if use_local:
            for p in price_items(estimate_lines(texts[g], before_game=g, min_sim=min_sim), cfg):
                q[int(p.item_id)] = (p.charge_price if p.charge_price > 0 else a0, p.acceptance_limit); matched += 1
        tot += (game_net(q, g, S, 1.15) + game_net(q, g, S, 1.4)) / 2
    return round(tot), matched, n
print('games', G)
print('absent 0/0          ', total(0, 0, False))
print('fixed 20/20 (live)  ', total(20, 20, False))
for a0 in (20, 40, 60, 100):
    print(f'fixed a={a0}: b=(0,20,50,100,200,400)', [total(a0, b, False)[0] for b in (0, 20, 50, 100, 200, 400)])
for ms in (0.35, 0.5, 0.65):
    print(f'local match sim>={ms}, rest 20/20:', total(20, 20, True, ms))
