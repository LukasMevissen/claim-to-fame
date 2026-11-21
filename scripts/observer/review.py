import sys, json, collections
sys.path.insert(0,'/home/diego/Documents/claim-to-fame')
from c2f.config import REPO_ROOT
US='GemischtesHack'
S=json.loads((REPO_ROOT/'results/market/summary.json').read_text())
sub={}
for l in open(REPO_ROOT/'runs/submissions.jsonl'):
    r=json.loads(l)
    if r.get('ok') and not r.get('dry_run'): sub[r['game_id']]={p['index']:(p['charge_price'],p['acceptance_limit']) for p in r['payload']}
for g in map(int,sys.argv[1:]):
    f=REPO_ROOT/f'data/transactions/game{g}.json'
    if not f.exists(): print('game',g,'not published yet'); continue
    d=json.loads(f.read_text())['items']
    tot=collections.defaultdict(float)
    print(f'== game {g}')
    for j in sorted({x['line_item_index'] for x in d}):
        rec=collections.defaultdict(float); paid=collections.defaultdict(float)
        for x in d:
            if x['line_item_index']==j: rec[x['issuer']]+=x['amount']; paid[x['reviewer']]+=x['amount']
        teams=set(rec)|set(paid)
        for k in teams: tot[k]+=rec[k]-paid[k]
        s=S[str(g)][str(j)]; best=max(teams,key=lambda k:rec[k]-paid[k])
        ch=sorted(v for v in s['charges'].values())
        print(f"  {j}: t[{s['t_low']},{s['t_high']}] us a,b={sub.get(g,{}).get(j)} fair={s['fair'].get(US)} rec {rec[US]:.0f} paid {paid[US]:.0f} net {rec[US]-paid[US]:.0f} | best {best[:10]} {rec[best]-paid[best]:.0f} a={s['charges'].get(best)} | charges {ch}")
    rk=sorted(tot,key=lambda k:-tot[k])
    print(f"  TOTAL us {tot[US]:.0f} rank {rk.index(US)+1}/{len(rk)}; top: "+', '.join(f'{k[:10]} {tot[k]:.0f}' for k in rk[:4]))
