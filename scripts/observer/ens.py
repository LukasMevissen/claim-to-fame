import json,glob,sys,math,statistics as st,itertools
sys.path.insert(0,'/home/diego/Documents/claim-to-fame')
from c2f.replay import game_net, load_summary
from c2f.config import load_config
from c2f.extract import ItemEstimate
from c2f.agent import price_items
S=load_summary(); cfg=load_config()
def load(d,model):
    R={}
    for f in glob.glob(f'{d}/game*__{model}__*.json'):
        r=json.load(open(f))
        if r['ok']: R[r['game']]=[ItemEstimate(**i) for i in r['items']]
    return R
src={'fable1':load('/tmp/c2f_work/run1','claude-fable-5-2'),'fable2':load('/tmp/c2f_work/run2','claude-fable-5-2'),
     'opus1':load('/tmp/c2f_work/run1','claude-opus-5-5'),'opus2':load('/tmp/c2f_work/run2','claude-opus-5-5'),
     'sonnet':load('/tmp/c2f_work/run1','claude-sonnet-5-5'),'sol':load('results/shadow','gpt-6.2-sol')}
for g in (15,16,17):
    f=f'results/extractions/game{g}__x3__gpt-6.2-sol.json'
    try: src['sol'][g]=[ItemEstimate(**i) for i in json.load(open(f))['items']]
    except Exception as e: print('no sol',g,e)
G=[g for g in range(1,17) if str(g) in S and all(g in src[k] for k in src)]
def combine(lists):
    by={}
    for L in lists:
        for it in L: by.setdefault(str(it.item_id).strip(),[]).append(it)
    out=[]
    for k,its in by.items():
        ms=[i.quantity*i.unit_price_gross_eur for i in its if i.quantity*i.unit_price_gross_eur>0]
        m=math.exp(st.mean(math.log(x) for x in ms)) if ms else 0.0
        p=st.mean(i.covered_prob*i.related_prob for i in its)
        sd=st.mean(i.price_log_sd for i in its)
        out.append(ItemEstimate(k,its[0].description,1.0,'',p,1.0,m,sd))
    return out
def score(names):
    res=[]
    for up in (1.15,1.4):
        tot=0; per=[]
        for g in G:
            items=combine([src[n][g] for n in names])
            q={int(p.item_id):(p.charge_price,p.acceptance_limit) for p in price_items(items,cfg)}
            v=game_net(q,g,S,up); tot+=v; per.append(v)
        res.append((round(tot),round(min(per))))
    return res
print('games',G)
for names in [('sol',),('fable1',),('fable2',),('opus1',),('opus2',),('sonnet',),('fable1','fable2'),('opus1','opus2'),('sol','fable1'),('sol','opus1'),('fable1','opus1'),('fable2','opus2'),('sol','fable1','opus1'),('sol','fable2','opus2'),('sol','fable1','opus2'),('sol','fable2','opus1'),('sol','fable1','opus1','sonnet'),('sol','fable1','fable2','opus1','opus2')]:
    r=score(names); print(f"{'+'.join(names):32s} total {r[0][0]:6d} / {r[1][0]:6d}   worst game {r[0][1]:6d} / {r[1][1]:6d}")
