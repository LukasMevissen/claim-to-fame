"""Experiment: one model call per invoice line (each call sees the whole case but prices one
line) versus the live single call for all lines. Claude Opus, low effort, prior-only price book.
usage: perline_exp.py RUN   -> /tmp/c2f_work/prompt_exp/perline/run<RUN>/game<g>.json"""
import sys, json, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from c2f.env import load_env
from c2f.case_io import load_case
from c2f import pricebook
from c2f.extract import SYSTEM, _parse
from c2f.anthropic_extract import _SCHEMA, API_BASE
from c2f.replay import load_summary
load_env(ROOT / '.env')
import anthropic
client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'], base_url=API_BASE, timeout=120, max_retries=2)
CASES = Path('/tmp/c2f_work/cases'); S = load_summary()

def call(shared, pos):
    resp = client.messages.create(
        model='claude-opus-5-5', max_tokens=8000, system=SYSTEM,
        output_config={'effort': os.environ.get('EFFORT', 'low'), 'format': {'type': 'json_schema', 'schema': _SCHEMA}},
        messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': shared, 'cache_control': {'type': 'ephemeral'}},
            {'type': 'text', 'text': f'Assess ONLY invoice line {pos} now. The other lines are context: use them where '
                                     f'the policy makes lines interact (shared sub-limits, one attendance per invoice, '
                                     f'combined work), but return exactly one item, for line {pos}.'}]}])
    if resp.stop_reason in ('refusal', 'max_tokens'): raise RuntimeError(resp.stop_reason)
    items = _parse(next(b.text for b in resp.content if b.type == 'text'))
    it = next((i for i in items if str(i.item_id).strip() == str(pos)), items[0]); it.item_id = str(pos)
    return it, resp.usage

def game(g, d):
    f = d / f'game{g}.json'
    if f.exists(): return
    case = load_case(CASES / f'game{g}', f'game{g}')
    ref = pricebook.reference(before_game=g)
    shared = (f"REFERENCE CASES (earlier games, revealed thresholds):\n{ref}\n\n" if ref else "") + \
        f"POLICY:\n{case.policy}\n\nDAMAGE DESCRIPTION:\n{case.description}\n\nINVOICE (text extracted from PDF):\n{case.invoice_text}"
    pos = sorted(pricebook.invoice_lines(case.invoice_text))
    t0 = time.time()
    try:
        first = call(shared, pos[0])                      # writes the prompt cache for the rest
        with ThreadPoolExecutor(8) as p: rest = list(p.map(lambda x: call(shared, x), pos[1:]))
        res = [first] + rest
        f.write_text(json.dumps({'game': g, 'items': [i.__dict__ for i, _ in res], 'secs': round(time.time() - t0, 1),
                                 'cache_read': sum(u.cache_read_input_tokens or 0 for _, u in res),
                                 'input': sum(u.input_tokens for _, u in res), 'output': sum(u.output_tokens for _, u in res)}))
    except Exception as e: print('fail', g, repr(e)[:200], flush=True)

if __name__ == '__main__':
    d = Path('/tmp/c2f_work/prompt_exp/perline' + ('_' + os.environ['EFFORT'] if os.environ.get('EFFORT') else '')) / f'run{sys.argv[1]}'; d.mkdir(parents=True, exist_ok=True)
    gs = [g for g in range(1, 60) if str(g) in S and (CASES / f'game{g}').exists()]
    with ThreadPoolExecutor(3) as p: list(p.map(lambda g: game(g, d), gs))
    fs = [json.load(open(x)) for x in d.glob('game*.json')]
    print('run', sys.argv[1], 'games', len(fs), 'of', len(gs), '| uncached input', sum(x['input'] for x in fs), 'cache read', sum(x['cache_read'] for x in fs), 'output', sum(x['output'] for x in fs))
