"""Replay games against the real opponents. usage: bench.py MODEL [effort] [--noref] [--games 1,2]"""
import sys, time, json, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/home/diego/Documents/claim-to-fame')
from c2f.env import load_env
from c2f.config import REPO_ROOT, load_config
from c2f.case_io import load_case
from c2f import pricebook, extract as ex
load_env(REPO_ROOT/'.env')
OUT=Path('/tmp/c2f_work/bench'); OUT.mkdir(exist_ok=True)

def run_model(g, model, effort, ref, img=True):
    f=OUT/f'g{g}__{model}__{effort}__{"ref" if ref else "noref"}__{ex.EXTRACTOR_VERSION}.json'
    if f.exists(): return json.loads(f.read_text())
    case=load_case(Path(f'/tmp/c2f_work/cases/game{g}'), f'game{g}')
    t0=time.time()
    try:
        items=ex.extract(case, model, 90, 0.0, False, with_image=img,
                         reference=pricebook.reference(before_game=g) if ref else "", effort=effort, tag='-bench')
        d={'secs':round(time.time()-t0,1),'items':[i.__dict__ for i in items]}
    except Exception as e:
        return {'secs':round(time.time()-t0,1),'error':repr(e)[:300],'items':[]}
    f.write_text(json.dumps(d,indent=1)); return d

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('effort',nargs='?',default=None)
    ap.add_argument('--noref',action='store_true'); ap.add_argument('--games',default='1,2,3,4,5,6,7,8'); ap.add_argument('-v',action='store_true')
    a=ap.parse_args(); gs=[int(x) for x in a.games.split(',')]
    with ThreadPoolExecutor(4) as p: res=list(p.map(lambda g: run_model(g,a.model,a.effort,not a.noref), gs))
    S=json.loads((REPO_ROOT/'results/market/summary.json').read_text())
    for g,r in zip(gs,res):
        print(f"game {g}: {r['secs']}s {r.get('error','')}")
        for it in r['items']:
            s=S[str(g)].get(str(it['item_id']),{})
            paid=any(v>0 for v in s.get('charges',{}).values())
            truth='ZERO' if not paid else f"[{s['t_low']}, {s['t_high']}]"
            print(f"   {it['item_id']:>2} p_ind={it['covered_prob']*it['related_prob']:.2f} est={it['quantity']*it['unit_price_gross_eur']:8.1f} sd={it['price_log_sd']:.2f} truth={truth:22s} {it['rationale'][:230] if a.v else ''}")
