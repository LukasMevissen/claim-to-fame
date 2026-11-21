"""Prompt experiments on Claude (Anthropic credits), prior-games-only price book.
usage: prompt_exp.py VARIANT RUN [model]   -> /tmp/c2f_work/prompt_exp/<variant>/run<RUN>/game<g>.json
       prompt_exp.py score
Variants: v0 live prompt; v1 neutral price instruction; v2 v1 + price built from components;
v3 v1 + price book lines annotated with our own earlier estimate."""
import sys, json, glob, math, os, statistics as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from c2f.env import load_env
from c2f.config import load_config
from c2f.case_io import load_case
from c2f import pricebook
from c2f.extract import SYSTEM, ItemEstimate
from c2f.replay import load_summary, game_net, t_point
load_env(ROOT / '.env'); os.environ.pop('ANTHROPIC_BASE_URL', None)
OUT = Path('/tmp/c2f_work/prompt_exp'); CASES = Path('/tmp/c2f_work/cases')
MODERATE = ("Without a close reference stay moderate:\n   small service and disposal lines tend to be cheaper than first guessed.")
assert MODERATE in SYSTEM
NEUTRAL = ("Without a close reference use ordinary market\n   prices for the trade and region; do not shade the estimate up or down.")
V1 = SYSTEM.replace(MODERATE, NEUTRAL)
V2 = V1.replace("Return JSON", "Build every price from its components before you commit to it: labour hours x hourly\nrate, material, plant, travel, plus VAT. Name the components and their amounts in `reasoning`.\n\nReturn JSON")
V3 = V1.replace("They are the best calibration available:", "Where a reference line shows \"we estimated X\", that was our own estimate before t was\nrevealed: learn from the direction and size of those misses for similar work.\nThey are the best calibration available:")
PROMPTS = {'v0': SYSTEM, 'v1': V1, 'v2': V2, 'v3': V3}

def sol_estimates():
    est = {}
    for f in glob.glob(str(ROOT / 'results/shadow/game*__gpt-6.2-sol__*.json')):
        r = json.load(open(f)); est[r['game']] = {str(i['item_id']).strip(): i['quantity'] * i['unit_price_gross_eur'] for i in r['items']}
    for f in glob.glob(str(ROOT / 'results/extractions/game*__x3__gpt-6.2-sol.json')):
        r = json.load(open(f)); g = int(r['case_id'].replace('game', '')) if r['case_id'].replace('game', '').isdigit() else None
        if g and g not in est: est[g] = {str(i['item_id']).strip(): i['quantity'] * i['unit_price_gross_eur'] for i in r['items']}
    return est

def reference(g, annotate):
    ref = pricebook.reference(before_game=g)
    if not annotate: return ref
    est = sol_estimates(); summary = load_summary(); out = []; cur = None; idx = 0
    for line in ref.splitlines():
        if line.startswith('Case '):
            cur = int(line.split()[1]); idx = 0
        elif line.startswith('  - ') and cur in est:
            keys = sorted(int(k) for k in summary[str(cur)]); k = str(keys[idx]) if idx < len(keys) else None; idx += 1
            if k in est[cur] and 't = 0' not in line: line += f" (we estimated {est[cur][k]:.0f})"
        out.append(line)
    return "\n".join(out)

def run(variant, r, model):
    from c2f.anthropic_extract import extract_claude
    S = load_summary(); d = OUT / variant / f'run{r}'; d.mkdir(parents=True, exist_ok=True)
    def one(g):
        f = d / f'game{g}.json'
        if f.exists(): return
        case = load_case(CASES / f'game{g}', f'game{g}')
        try:
            items, meta = extract_claude(case, model, 120, reference(g, variant == 'v3'), 'low', system=PROMPTS[variant])
            f.write_text(json.dumps({'game': g, 'items': [i.__dict__ for i in items], **meta}))
        except Exception as e: print('fail', variant, g, repr(e)[:200])
    gs = [g for g in range(1, 60) if str(g) in S and (CASES / f'game{g}').exists()]
    with ThreadPoolExecutor(5) as p: list(p.map(one, gs))
    print(variant, r, 'done', len(list(d.glob('game*.json'))), 'of', len(gs))

def score():
    from c2f.agent import price_items
    cfg = load_config(); S = load_summary()
    for d in sorted(OUT.glob('*/run*')):
        files = sorted(d.glob('game*.json')); net = {1.15: 0.0, 1.4: 0.0}; errs = []
        for f in files:
            r = json.load(open(f)); items = [ItemEstimate(**i) for i in r['items']]; g = r['game']
            q = {int(p.item_id): (p.charge_price, p.acceptance_limit) for p in price_items(items, cfg)}
            for up in net: net[up] += game_net(q, g, S, up)
            for it in items:
                s = S[str(g)].get(str(it.item_id).strip())
                if not s or s['t_low'] <= 0: continue
                m = it.quantity * it.unit_price_gross_eur; lo, hi = s['t_low'], s['t_high']
                errs.append(math.log(m / lo) if m < lo else (math.log(m / hi) if hi and m > hi else 0.0))
        print(f"{d.parent.name}/{d.name}: games {len(files)} net {net[1.15]:8.0f} / {net[1.4]:8.0f} | lines {len(errs)} "
              f"mean|e| {st.mean(abs(e) for e in errs):.3f} bias {st.mean(errs):+.3f} below {sum(e < 0 for e in errs)} above {sum(e > 0 for e in errs)}", flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'score': score()
    else: run(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'claude-opus-5-5')
