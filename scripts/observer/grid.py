import sys, json
sys.path.insert(0,'/tmp/c2f_work/scripts'); sys.path.insert(0,'/home/diego/Documents/claim-to-fame')
from replay import *
from pathlib import Path
def est(model,effort):
    out={}
    for g in range(1,13):
        d=json.loads(Path(f'/tmp/c2f_work/bench/g{g}__{model}__{effort}__ref__x3.json').read_text())
        out[g]={int(i['item_id']):(i['quantity']*i['unit_price_gross_eur'], i['covered_prob']*i['related_prob']) for i in d['items']}
    return out
for model,eff in [('gpt-6.2-sol','low')]:
    E=est(model,eff); print('==',model)
    for pz_cut in (0.5,):
        best=[]
        for k in (0.7,0.8,0.85,0.9,0.95,1.0,1.1):
            row=[]
            for m in (0.7,0.8,0.9,1.0,1.1,1.25):
                q={g:{i:(k*v, m*v if p>=pz_cut else 0.0) for i,(v,p) in it.items()} for g,it in E.items()}
                row.append((score(q,1.15)[0]+score(q,1.4)[0])//2)
            print(f'  a=est*{k:4}: b=est*(.7,.8,.9,1,1.1,1.25) ->',row)
