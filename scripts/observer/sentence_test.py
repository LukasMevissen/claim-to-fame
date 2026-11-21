"""Did the two extra instruction sentences (x4, live in game 39 only) cause the low prices
there? Game 39 priced N times with them and N times without, Claude Opus at medium effort as
live, price book of games before 39. usage: sentence_test.py [N]"""
import sys, json, time, importlib.util, math, statistics as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.env import load_env
from c2f.config import load_config
from c2f.case_io import load_case
from c2f import pricebook
from c2f.extract import SYSTEM as WITHOUT
from c2f.anthropic_extract import extract_claude
from c2f.agent import price_items
from c2f.replay import load_summary, game_net
load_env(ROOT / '.env')
spec = importlib.util.spec_from_file_location('x4', '/tmp/c2f_work/extract_x4.py'); x4 = importlib.util.module_from_spec(spec)
src = open('/tmp/c2f_work/extract_x4.py').read(); a = src.index('SYSTEM = """') + len('SYSTEM = """'); WITH = src[a:src.index('"""', a)]
assert 'multiply it by 1.19' in WITH and 'multiply it by 1.19' not in WITHOUT
G = 39; N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = Path('/tmp/c2f_work/sentence_test'); OUT.mkdir(exist_ok=True)
case = load_case(Path(f'/tmp/c2f_work/cases/game{G}'), f'game{G}'); ref = pricebook.reference(before_game=G)
def wait():
    while not 60 <= (time.time() - 1795352400 + 0) % 800 <= 775: time.sleep(5)
def one(a):
    arm, k = a; f = OUT / f'{arm}_{k}.json'
    if f.exists(): return
    wait()
    try:
        items, meta = extract_claude(case, 'claude-opus-5-5', 120, ref, 'medium', system=WITH if arm == 'with' else WITHOUT)
        f.write_text(json.dumps({'items': [i.__dict__ for i in items], **meta}))
    except Exception as e: print('fail', arm, k, repr(e)[:200], flush=True)
with ThreadPoolExecutor(5) as p: list(p.map(one, [(arm, k) for arm in ('with', 'without') for k in range(N)]))
from c2f.extract import ItemEstimate
cfg = load_config(); S = load_summary(); s39 = S[str(G)]
print(f'game {G}: lines with a proven-fair level:', {k: v['t_low'] for k, v in s39.items() if v['t_low'] > 0})
for arm in ('with', 'without'):
    rows = []
    for f in sorted(OUT.glob(f'{arm}_*.json')):
        items = [ItemEstimate(**i) for i in json.load(open(f))['items']]
        est = {str(i.item_id).strip(): i.quantity * i.unit_price_gross_eur for i in items}
        ratios = [est[k] / v['t_low'] for k, v in s39.items() if v['t_low'] > 0 and k in est and est[k] > 0]
        q = {int(p.item_id): (p.charge_price, p.acceptance_limit) for p in price_items(items, cfg)}
        rows.append((st.median(ratios), est.get('8', 0), est.get('15', 0), game_net(q, G, S, 1.15), game_net(q, G, S, 1.4)))
    print(f'-- {arm} the two sentences ({len(rows)} runs)')
    for r in rows: print(f'   median estimate / proven-fair level {r[0]:.2f}   technician 9 hrs lines: {r[1]:6.0f} and {r[2]:6.0f} (fair level 732)   replay net {r[3]:7.0f} / {r[4]:7.0f}')
    if rows: print(f'   AVERAGE: ratio {st.mean(r[0] for r in rows):.2f}   technician {st.mean(r[1] for r in rows):.0f} / {st.mean(r[2] for r in rows):.0f}   net {st.mean(r[3] for r in rows):.0f} / {st.mean(r[4] for r in rows):.0f}')
