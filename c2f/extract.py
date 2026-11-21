"""One LLM call per case: list line items and estimate coverage, relatedness and fair gross price.

Outputs are cached in results/extractions/ keyed by case, EXTRACTOR_VERSION and model,
so evaluation and CI can replay the pricing step without calling the API.
Bump EXTRACTOR_VERSION whenever the prompt or schema changes.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .case_io import Case
from .config import REPO_ROOT

EXTRACTOR_VERSION = "x3"   # x3: policy-first prompt, line totals, price book of revealed thresholds (x4 with two extra sentences ran in game 39 only and was withdrawn)
IMAGE_MAX_SIDE = 1024
RAW_IMAGE_MAX_BYTES = 2_000_000   # without Pillow, larger photos are left out (upload too slow)


def image_for_model(path: Path) -> tuple[str, bytes] | None:
    """Photo as (mime, bytes), shrunk so the upload stays small.

    Case photos can be 10 MB+ PNGs; sent as-is the model call timed out (game 5).
    """
    try:
        from PIL import Image
    except ImportError:
        if path.stat().st_size > RAW_IMAGE_MAX_BYTES:
            return None
        return ("image/png" if path.suffix.lower() == ".png" else "image/jpeg"), path.read_bytes()
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return "image/jpeg", buf.getvalue()
CACHE_DIR = REPO_ROOT / "results" / "extractions"

SYSTEM = """You are a senior claims adjuster. You receive an insurance policy, a damage description,
an invoice WITHOUT prices and sometimes a photo. For every invoice line a secret threshold t
exists: the highest gross total (incl. VAT, whole line, quantity x unit price) that a claims
expert applying THIS policy to THIS claim would indemnify. Estimate t for every line.

Work policy-first; the policy wording decides far more than market prices do:
1. Find the insured peril and the cover sections actually in force. Optional covers or
   endorsements count only if the description or schedule says they were taken.
2. Per line: is it excluded, unrelated to the damage, pre-existing, maintenance, an improvement,
   or an ancillary/site charge the policy does not carry? Then t = 0. Check definitions and
   exclusions literally (volume limits, "standing on its own", external surfaces, combined
   lines that mix insured and uninsured work, who is paid, etc.).
3. If indemnified, start from a fair, ordinary market price for the work actually needed
   (not the most thorough or expensive variant), then apply the policy's arithmetic:
   caps and sub-limits incl. VAT (an aggregate sub-limit is shared rateably across the
   affected lines), percentage reductions for breached duties, no betterment over the
   pre-loss standard, repair capped at replacement value or purchase price, deductibles.
   Figures in the description (purchase price, age, sizes, volumes) are there to be used.
4. REFERENCE CASES, when given, are earlier cases with what was later revealed about t.
   They are the best calibration available: reuse their price level for similar work and
   their coverage logic for similar clauses. Without a close reference stay moderate:
   small service and disposal lines tend to be cheaper than first guessed.

Return JSON {"items": [...]} with one object per invoice line, in invoice order:
- item_id: position number exactly as printed
- description: short
- reasoning: one or two sentences naming the decisive clause(s) and the arithmetic
- indemnified_prob: probability (0..1) that t > 0
- line_total_gross_eur: your best estimate of t IF it is indemnified, gross total for the
  whole line after all policy arithmetic. Always give a positive number, even if you
  believe the line is excluded.
- price_log_sd: uncertainty of that number as a log standard deviation (0.05 = fixed by the
  policy or a near-identical reference, 0.3 = normal, 0.7 = rough guess)
Do not skip lines. Do not invent lines."""


# Price level for the Claude stages (c2f/anthropic_extract.py). The revealed thresholds sit at
# the upper end of what teams charge (about 1.3x the median charge on covered lines), and
# "ordinary market price" left the model too low on labour, materials and standard devices.
# Backtest claude-opus-5-5@low, games 10-19, 5 runs each, price book of earlier games only:
# price error 0.30 -> 0.23, replayed net 22.7k -> 29.3k (better in 8 of 10 games; game 12
# worse, where the higher level overshot two large lines).
_STEP3 = """3. If indemnified, start from a fair, ordinary market price for the work actually needed
   (not the most thorough or expensive variant), then apply the policy's arithmetic:"""
_STEP3_UPPER = """3. If indemnified, t is the HIGHEST price a claims expert would still accept as appropriate
   for the work actually needed: the upper end of the customary price range, not the typical
   or cheapest offer. Start there, then apply the policy's arithmetic:"""
assert _STEP3 in SYSTEM
SYSTEM_UPPER = SYSTEM.replace(_STEP3, _STEP3_UPPER)

@dataclass
class ItemEstimate:
    item_id: str
    description: str
    quantity: float
    unit: str
    covered_prob: float
    related_prob: float
    unit_price_gross_eur: float
    price_log_sd: float
    rationale: str = ""


def _user_content(case: Case, with_image: bool = True, reference: str = "") -> list[dict]:
    ref = f"REFERENCE CASES (earlier games, revealed thresholds):\n{reference}\n\n" if reference else ""
    parts: list[dict] = [{"type": "text", "text":
        f"{ref}POLICY:\n{case.policy}\n\nDAMAGE DESCRIPTION:\n{case.description}\n\n"
        + (f"INVOICE (text extracted from PDF):\n{case.invoice_text}" if case.invoice_text
           else "INVOICE: see attached PDF")}]
    if not case.invoice_text and case.invoice_pdf.exists():
        b64 = base64.b64encode(case.invoice_pdf.read_bytes()).decode()
        parts.append({"type": "file", "file": {"filename": "invoices.pdf",
                                               "file_data": f"data:application/pdf;base64,{b64}"}})
    img = image_for_model(case.image) if (with_image and case.image is not None) else None
    if img is not None:
        mime, data = img
        b64 = base64.b64encode(data).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return parts


def _parse(payload: str) -> list[ItemEstimate]:
    items = json.loads(payload)["items"]
    out = []
    for it in items:
        if "line_total_gross_eur" in it:       # x3 schema: whole-line total, single probability
            qty, unit_price = 1.0, max(float(it.get("line_total_gross_eur") or 0), 0.0)
            cov, rel = float(it.get("indemnified_prob", 1)), 1.0
        else:
            qty, unit_price = float(it.get("quantity") or 1), max(float(it.get("unit_price_gross_eur", 0)), 0.0)
            cov, rel = float(it.get("covered_prob", 1)), float(it.get("related_prob", 1))
        out.append(ItemEstimate(
            item_id=str(it["item_id"]),
            description=str(it.get("description", "")),
            quantity=qty,
            unit=str(it.get("unit", "")),
            covered_prob=min(max(cov, 0.0), 1.0),
            related_prob=min(max(rel, 0.0), 1.0),
            unit_price_gross_eur=unit_price,
            price_log_sd=max(float(it.get("price_log_sd", 0.5)), 0.0),
            rationale=str(it.get("reasoning", it.get("rationale", ""))),
        ))
    return out


def cache_path(case_id: str, model: str) -> Path:
    return CACHE_DIR / f"{case_id}__{EXTRACTOR_VERSION}__{model}.json"


def load_cached(case_id: str, model: str) -> list[ItemEstimate] | None:
    p = cache_path(case_id, model)
    if not p.exists():
        return None
    return [ItemEstimate(**d) for d in json.loads(p.read_text())["items"]]


def save_cache(case_id: str, model: str, items: list[ItemEstimate]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(case_id, model).write_text(json.dumps(
        {"case_id": case_id, "extractor_version": EXTRACTOR_VERSION, "model": model,
         "items": [asdict(i) for i in items]}, indent=2, ensure_ascii=False))


def _model_args(model: str, temperature: float, effort: str | None) -> dict:
    """gpt-4.x takes a temperature; later (reasoning) models take a reasoning effort instead."""
    if model.startswith(("gpt-4", "chat")):
        return {"temperature": temperature}
    return {"reasoning_effort": effort} if effort else {}


KEY_VARS = ("OPENAI_KEY", "OPENAI_KEY2")   # tried in this order; OPENAI_API_KEY as a last resort
_dead_keys: set[str] = set()               # env var names whose key was refused in this process


def _api_keys() -> list[tuple[str, str]]:
    """(env var, key) for every configured key, keys that still work first. .env is re-read
    on every call, so a key added while the scheduler runs is picked up without a restart."""
    from .env import load_env
    load_env(REPO_ROOT / ".env")
    keys = [(v, os.environ[v].strip()) for v in (*KEY_VARS, "OPENAI_API_KEY")
            if os.environ.get(v, "").strip()]
    seen, out = set(), []
    for v, k in keys:
        if k not in seen:
            seen.add(k)
            out.append((v, k))
    return sorted(out, key=lambda vk: vk[0] in _dead_keys)


def _create_with_failover(timeout: float, **kwargs):
    """chat.completions.create with key failover: if a key is refused (invalid, no
    permission, out of credit or rate limited) the same request goes to the next key.
    These refusals come back within about a second, so the time window is not at risk.
    Other errors (timeouts, bad requests) are raised unchanged."""
    import openai

    last: Exception | None = None
    for var, key in _api_keys():
        client = openai.OpenAI(api_key=key, timeout=timeout, max_retries=0)
        try:
            resp = client.chat.completions.create(**kwargs)
            _dead_keys.discard(var)
            return resp
        except (openai.AuthenticationError, openai.PermissionDeniedError,
                openai.RateLimitError) as e:
            _dead_keys.add(var)
            last = e
            print(f"[extract] {var} refused ({type(e).__name__}); trying the next key")
    if last is None:
        raise RuntimeError("no OpenAI key configured (OPENAI_KEY / OPENAI_KEY2)")
    raise last


def extract(case: Case, model: str, timeout: float, temperature: float = 0.0,
            use_cache: bool = True, with_image: bool = True, reference: str = "",
            effort: str | None = None, tag: str = "", save: bool = True) -> list[ItemEstimate]:
    """with_image=False skips the photo: faster, used as the first-stage submission.
    reference: price-book text (c2f/pricebook.py). tag: extra cache-key suffix."""
    cid = case.case_id if (with_image or case.image is None) else f"{case.case_id}-noimg"
    cid += tag
    if use_cache and (hit := load_cached(cid, model)) is not None:
        return hit
    resp = _create_with_failover(
        timeout,
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": _user_content(case, with_image, reference)}],
        **_model_args(model, temperature, effort),
    )
    items = _parse(resp.choices[0].message.content)
    if save:
        save_cache(cid, model, items)
    return items
