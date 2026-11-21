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

EXTRACTOR_VERSION = "x2"   # x2: photo shrunk to <= 1024 px JPEG before sending
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

SYSTEM = """You are a senior claims adjuster for a German property and motor insurer.
You receive an insurance policy, a damage description, an invoice WITHOUT prices
(line items with quantities) and sometimes a photo.

For EVERY invoice line item return:
- item_id: the position number/identifier exactly as printed on the invoice
- description, quantity, unit
- covered_prob: probability (0..1) the policy covers this item (exclusions, deductibles do not make it 0)
- related_prob: probability (0..1) the item is genuinely needed for the reported damage
- unit_price_gross_eur: the fair unit price a claims expert would accept, in EUR INCLUDING VAT
  (assume 19% German VAT unless the invoice states otherwise); typical German market rates
- price_log_sd: your uncertainty about that price as a log standard deviation (0.1 = very sure, 0.8 = rough guess)
- rationale: one short sentence

Return JSON: {"items": [ ... ]}. Do not skip items. Do not invent items."""


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


def _user_content(case: Case, with_image: bool = True) -> list[dict]:
    parts: list[dict] = [{"type": "text", "text":
        f"POLICY:\n{case.policy}\n\nDAMAGE DESCRIPTION:\n{case.description}\n\n"
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
        out.append(ItemEstimate(
            item_id=str(it["item_id"]),
            description=str(it.get("description", "")),
            quantity=float(it.get("quantity") or 1),
            unit=str(it.get("unit", "")),
            covered_prob=min(max(float(it.get("covered_prob", 1)), 0.0), 1.0),
            related_prob=min(max(float(it.get("related_prob", 1)), 0.0), 1.0),
            unit_price_gross_eur=max(float(it.get("unit_price_gross_eur", 0)), 0.0),
            price_log_sd=max(float(it.get("price_log_sd", 0.5)), 0.0),
            rationale=str(it.get("rationale", "")),
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


def extract(case: Case, model: str, timeout: float, temperature: float = 0.0,
            use_cache: bool = True, with_image: bool = True) -> list[ItemEstimate]:
    """with_image=False skips the photo: faster, used as the first-stage submission."""
    cid = case.case_id if (with_image or case.image is None) else f"{case.case_id}-noimg"
    if use_cache and (hit := load_cached(cid, model)) is not None:
        return hit
    from openai import OpenAI  # imported lazily so pricing/tests run without it

    client = OpenAI(api_key=os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY"),
                    timeout=timeout, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": _user_content(case, with_image)}],
    )
    items = _parse(resp.choices[0].message.content)
    save_cache(cid, model, items)
    return items
