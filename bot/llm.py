"""Claude juzga cada línea de la factura: cubierta, relacionada y precio unitario neto.
El total bruto (cantidad x unitario x IVA) se calcula en código, nunca en el LLM."""
import base64
import json
import re
import urllib.request

import config
from belief import line_sd

PROMPT = """You are a senior German home-insurance claims adjuster.
You get a damage report, the policy (coverage + exclusions), the handyman's invoice line
items (quantities and units, NO prices) and maybe photos.

For EVERY invoice line, judge:
- covered_p: probability the POLICY covers this item (explicitly excluded -> ~0)
- related_p: probability the item is explained by THIS damage (report + photos)
- unit_net: reasonable NET unit price in EUR (Germany, today) that an insurer would accept,
  median estimate; unit_net_low / unit_net_high = 10% / 90% range
- quantity and unit exactly as on the invoice

Return ONLY JSON:
{"lines": [{"index": <invoice index>, "item": "...", "quantity": <number>, "unit": "...",
  "covered_p": 0-1, "related_p": 0-1, "unit_net": <number>, "unit_net_low": <number>,
  "unit_net_high": <number>, "why": "<= 12 words"}]}
Use the invoice's own index numbers. Do not skip any line.

=== DAMAGE REPORT ===
{description}

=== POLICY ===
{policy}

=== INVOICE ===
{invoice}
"""


def _img(path):
    media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return {"type": "image", "source": {"type": "base64", "media_type": media,
                                            "data": base64.b64encode(f.read()).decode()}}


def call(case, model, with_images, timeout):
    text = PROMPT.replace("{description}", case["description"]) \
                 .replace("{policy}", case["policy"]).replace("{invoice}", case["invoice_text"])
    content = [_img(p) for p in case["images"][:4]] if with_images else []
    content.append({"type": "text", "text": text})
    body = json.dumps({"model": model, "max_tokens": 4000, "temperature": 0.2,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"x-api-key": config.ANTHROPIC_API_KEY, "anthropic-version": config.ANTHROPIC_VERSION,
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return parse("".join(b.get("text", "") for b in out.get("content", [])))


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    lines = json.loads(m.group(0))["lines"]
    out = {}
    for ln in lines:
        q = float(ln.get("quantity") or 1)
        u = float(ln["unit_net"])
        lo = float(ln.get("unit_net_low") or u * 0.7)
        hi = float(ln.get("unit_net_high") or u * 1.4)
        out[int(ln["index"])] = {
            "item": ln.get("item", ""), "quantity": q, "unit": ln.get("unit", ""),
            "mid": q * u * config.VAT,                        # total BRUTO de la línea
            "sd": line_sd(lo, hi),
            "p_bad": 1 - float(ln.get("covered_p", 1)) * float(ln.get("related_p", 1)),
            "covered_p": float(ln.get("covered_p", 1)), "related_p": float(ln.get("related_p", 1)),
            "why": ln.get("why", ""),
        }
    return out


def combine(fast, full):
    """Si llegan las dos pasadas: media geométrica del precio, la del modelo grande pesa 2/3."""
    if not fast:
        return full
    if not full:
        return fast
    out = {}
    for i in set(fast) | set(full):
        f, g = fast.get(i), full.get(i)
        if not f or not g:
            out[i] = f or g
            continue
        o = dict(g)
        o["mid"] = f["mid"] ** (1 / 3) * g["mid"] ** (2 / 3)
        o["p_bad"] = f["p_bad"] / 3 + 2 * g["p_bad"] / 3
        out[i] = o
    return out
