import json,glob,sys,math,dataclasses
sys.path.insert(0,'/home/diego/Documents/claim-to-fame')
from c2f.replay import game_net, load_summary, t_point, item_net
from c2f.config import load_config
from c2f.extract import ItemEstimate
from c2f.agent import to_belief
from c2f.pricing import quote
S=load_summary(); cfg=load_config(); P=cfg.pricing
est={}
for f in glob.glob('results/shadow/game*__gpt-6.2-sol__*.json'):
    r=json.load(open(f)); est[r['game']]=[ItemEstimate(**i) for i in r['items']]
for g in range(15,30):
    try: est[g]=[ItemEstimate(**i) for i in json.load(open(f'results/extractions/game{g}__x3__gpt-6.2-sol.json'))['items']]
    except Exception: pass
G=sorted(g for g in est if str(g) in S)
print('games',G, flush=True)
cache={}
def q(m,sd,p):
    k=(round(m,2),round(sd,3),round(p,3))
    if k not in cache:
        it=ItemEstimate('x','',1.0,'',p,1.0,m,sd); r=quote(to_belief(it,cfg),P); cache[k]=(r.charge_price,r.acceptance_limit)
    return cache[k]
for up in (1.15,1.4):
    tot=dict(oracle=0,perfect_est=0,perfect_cov=0,perfect_price=0,llm=0,absent=0)
    cls=dict(cov_wrong=0,in_interval=0,above=0,below=0); loss=dict(cov_wrong=0,in_interval=0,above=0,below=0,zero_lines=0)
    for g in G:
        for it in est[g]:
            s=S[str(g)].get(str(it.item_id).strip())
            if s is None: continue
            t=t_point(s,up); m=it.quantity*it.unit_price_gross_eur; p=it.covered_prob*it.related_prob; sd=it.price_log_sd
            o=item_net(t,t,s,t); tot['oracle']+=o; tot['absent']+=item_net(0,0,s,t)
            pe=item_net(*q(t if t>0 else max(m,1), 0.0, 1.0 if t>0 else 0.0),s,t); tot['perfect_est']+=pe
            tot['perfect_cov']+=item_net(*q(max(m,1e-9),sd,1.0 if t>0 else 0.0),s,t)
            tot['perfect_price']+=item_net(*q(t if t>0 else max(m,1),sd,p),s,t)
            l=item_net(*q(m,sd,p),s,t); tot['llm']+=l
            if t>0:
                lo,hi=s['t_low'],s['t_high']
                c='cov_wrong' if p<0.5 else ('below' if (lo>0 and m<lo) else ('above' if (hi and m>hi) or (not hi and m>t) else 'in_interval'))
                cls[c]+=1; loss[c]+=pe-l
            else: loss['zero_lines']+=pe-l
    print(f'--- t assumption {up}'); print({k:round(v) for k,v in tot.items()})
    print('lines with t>0 by LLM outcome:',cls); print('loss vs perfect-estimate pricing, by outcome:',{k:round(v) for k,v in loss.items()}, flush=True)
