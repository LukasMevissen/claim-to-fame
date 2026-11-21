"""Claude (Anthropic SDK) version of the case assessment, same prompt and output as extract.py.

Used for shadow runs only (c2f/shadow.py): nothing here is ever submitted.
Needs ANTHROPIC_API_KEY in .env.
"""

from __future__ import annotations

import base64
import os

from .case_io import Case
from .extract import SYSTEM, ItemEstimate, _parse, image_for_model

# Models with a server-side refusal fallback. If the fallback model answers instead, the
# result records it (served_by), so a comparison never credits the wrong model.
API_BASE = "https://api.anthropic.com"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_HAS_FALLBACK = ("claude-fable-5-2", "claude-opus-5-5", "claude-sonnet-5-5")

_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "item_id": {"type": "string"},
            "description": {"type": "string"},
            "reasoning": {"type": "string"},
            "indemnified_prob": {"type": "number"},
            "line_total_gross_eur": {"type": "number"},
            "price_log_sd": {"type": "number"},
        },
        "required": ["item_id", "description", "reasoning", "indemnified_prob",
                     "line_total_gross_eur", "price_log_sd"],
        "additionalProperties": False,
    }}},
    "required": ["items"],
    "additionalProperties": False,
}


def extract_claude(case: Case, model: str, timeout: float, reference: str = "",
                   effort: str | None = "low", with_image: bool = True
                   ) -> tuple[list[ItemEstimate], dict]:
    """Returns (items, meta). meta: served_by, input_tokens, output_tokens."""
    import anthropic

    # base_url is pinned: an ANTHROPIC_BASE_URL in the environment (e.g. a proxy configured
    # for a coding assistant) must never receive this project's key or case data.
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"),
                                 base_url=API_BASE, timeout=timeout, max_retries=0)
    ref = f"REFERENCE CASES (earlier games, revealed thresholds):\n{reference}\n\n" if reference else ""
    content: list[dict] = []
    img = image_for_model(case.image) if (with_image and case.image is not None) else None
    if img is not None:
        mime, data = img
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": mime, "data": base64.standard_b64encode(data).decode()}})
    content.append({"type": "text", "text":
                    f"{ref}POLICY:\n{case.policy}\n\nDAMAGE DESCRIPTION:\n{case.description}\n\n"
                    f"INVOICE (text extracted from PDF):\n{case.invoice_text}"})
    output_config: dict = {"format": {"type": "json_schema", "schema": _SCHEMA}}
    if effort:
        output_config["effort"] = effort
    kwargs = dict(model=model, max_tokens=16000, system=SYSTEM,
                  messages=[{"role": "user", "content": content}], output_config=output_config)
    if model in _HAS_FALLBACK:
        resp = client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
    else:
        resp = client.messages.create(**kwargs)
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"{model}: stop_reason={resp.stop_reason}")
    text = next(b.text for b in resp.content if b.type == "text")
    meta = {"served_by": resp.model, "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens}
    return _parse(text), meta
