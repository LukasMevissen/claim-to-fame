"""Web access for the bill-check agent (c2f/webcheck.py): search, page reading, HTTP cache.

Everything fetched is cached on disk (results/webcheck/http/), so a backtest can be re-run
without touching the sites again and every price the agent quotes can be traced to the page
it came from. Sites behind a bot challenge (Cloudflare "Just a moment", 403/429/503) are
retried through the Decodo scraping API when DECODO_AUTH is set in .env; without it the
tool reports the block to the model, which then uses another source.
"""

from __future__ import annotations

import base64
import hashlib
import html as htmllib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests

from .config import REPO_ROOT

CACHE = REPO_ROOT / "results" / "webcheck" / "http"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "de-DE,de;q=0.9,en;q=0.6",
           "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"}
DECODO_URL = "https://scraper-api.decodo.com/v2/scrape"
BLOCK_MARKS = ("just a moment", "cf-challenge", "challenge-platform", "attention required",
               "captcha-delivery", "px-captcha", "access denied", "enable javascript and cookies")

stats = {"requests": 0, "cache_hits": 0, "blocked": 0, "decodo": 0}


def _key(method: str, url: str, body: str = "") -> Path:
    h = hashlib.sha1(f"{method} {url} {body}".encode()).hexdigest()[:20]
    host = re.sub(r"[^a-z0-9.-]", "", urlparse(url).netloc.lower())[:40]
    return CACHE / f"{host}__{h}.json"


def looks_blocked(status: int, text: str) -> bool:
    if status in (202, 401, 403, 429, 503):
        return True
    head = text[:6000].lower()
    return len(text) < 20000 and any(m in head for m in BLOCK_MARKS)


def _decodo(url: str, timeout: float) -> tuple[int, str]:
    """Fetch through the Decodo Web Scraping API (headless render, solves bot challenges)."""
    auth = os.environ.get("DECODO_AUTH", "").strip()      # the "Basic ..." token from the dashboard
    if not auth:
        raise RuntimeError("DECODO_AUTH not set")
    if not auth.lower().startswith("basic "):
        auth = "Basic " + auth
    r = requests.post(DECODO_URL, timeout=timeout + 60,
                      headers={"Authorization": auth, "Content-Type": "application/json"},
                      json={"url": url, "headless": "html", "geo": "Germany", "locale": "de-de"})
    r.raise_for_status()
    res = r.json()["results"][0]
    stats["decodo"] += 1
    return int(res.get("status_code") or 200), res.get("content") or ""


def _raw(url: str, method: str, data: dict | str | None, timeout: float,
         headers: dict) -> tuple[int, str]:
    """One request. curl when available (python-requests is refused by several sites and by
    the search engines because of its TLS fingerprint), requests otherwise."""
    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-s", "-L", "--compressed", "-m", str(int(timeout)), "-X", method,
               "-w", "\n%{http_code}"]
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]
        if isinstance(data, dict):
            for k, v in data.items():
                cmd += ["--data-urlencode", f"{k}={v}"]
        elif data:
            cmd += ["--data-raw", data]
        try:
            out = subprocess.run(cmd + [url], capture_output=True, timeout=timeout + 5).stdout
        except subprocess.TimeoutExpired:
            return 0, "timeout"
        body, _, code = out.rpartition(b"\n")
        m = re.search(rb"charset=[\"']?([\w-]+)", body[:3000], re.I)
        enc = m.group(1).decode("ascii", "replace") if m else "utf-8"
        try:
            text = body.decode(enc, "replace")
        except LookupError:
            text = body.decode("utf-8", "replace")
        status = int(code) if code.strip().isdigit() else 0
        return status, text if status else "connection failed"
    try:
        r = requests.request(method, url, data=data, timeout=timeout, headers=headers)
        return r.status_code, r.text
    except requests.RequestException as e:
        return 0, repr(e)[:300]


def http(url: str, method: str = "GET", data: dict | str | None = None, timeout: float = 20,
         use_cache: bool = True, headers: dict | None = None) -> tuple[int, str]:
    """(status, body). Cached; blocked pages go through Decodo when a key is configured.
    status 0 = network error (body holds the message)."""
    body_key = json.dumps(data, sort_keys=True) if isinstance(data, dict) else (data or "")
    p = _key(method, url, body_key)
    if use_cache and p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if not (d.get("blocked") and os.environ.get("DECODO_AUTH")):   # retry blocks once a key exists
            stats["cache_hits"] += 1
            return d["status"], d["text"]
    stats["requests"] += 1
    status, text = _raw(url, method, data, timeout, {**HEADERS, **(headers or {})})
    blocked = status == 0 or looks_blocked(status, text)
    if blocked and method == "GET" and os.environ.get("DECODO_AUTH"):
        try:
            status, text = _decodo(url, timeout)
            blocked = looks_blocked(status, text)
        except Exception as e:  # noqa: BLE001  the direct answer stays the result
            text = text or repr(e)[:300]
    if blocked:
        stats["blocked"] += 1
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"url": url, "method": method, "status": status, "blocked": blocked,
                             "fetched": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text},
                            ensure_ascii=False), encoding="utf-8")
    return status, text


# ---------------------------------------------------------------- html -> text

_DROP = re.compile(r"<(script|style|noscript|svg|head|nav|footer|form|iframe)\b.*?</\1>", re.S | re.I)
_BREAK = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|section|article|table|tr|dt|dd|blockquote)\b[^>]*>", re.I)
_CELL = re.compile(r"</(td|th)>", re.I)
_TAG = re.compile(r"<[^>]+>")


def to_text(page: str) -> str:
    """Readable text: one line per block, table cells joined with ' | '."""
    s = _DROP.sub(" ", page)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = _CELL.sub(" | ", s)
    s = _BREAK.sub("\n", s)
    s = htmllib.unescape(_TAG.sub(" ", s)).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip(" |") for ln in s.splitlines()]
    out, prev = [], None
    for ln in lines:
        if len(ln) >= 3 and ln != prev:
            out.append(ln)
            prev = ln
    return "\n".join(out)


PRICE = re.compile(r"(€|\bEUR\b|\bEuro\b)", re.I)


def excerpt(text: str, find: str = "", max_chars: int = 3500) -> str:
    """The parts of a page that matter for a price check: lines that carry a price or one of
    the `find` keywords, each with one line of context, in page order."""
    lines = text.splitlines()
    kws = [k.lower() for k in re.split(r"[,;\s]+", find) if len(k) >= 3]
    score = []
    for ln in lines:
        low = ln.lower()
        has_price = bool(PRICE.search(ln)) and bool(re.search(r"\d", ln))
        score.append(2 * has_price + sum(k in low for k in kws))
    keep: set[int] = set()
    # best lines first, until the budget is used; context lines ride along
    for i in sorted(range(len(lines)), key=lambda i: (-(score[i] >= 3), -(score[i] >= 2), i)):
        if score[i] < (2 if not kws else 1):
            continue
        add = {j for j in (i - 1, i, i + 1) if 0 <= j < len(lines)}
        if sum(len(lines[j]) + 1 for j in keep | add) > max_chars:
            break
        keep |= add
    if not keep:
        return text[:max_chars]
    out, last = [], -2
    for i in sorted(keep):
        if i != last + 1 and out:
            out.append("...")
        out.append(lines[i][:400])
        last = i
    return "\n".join(out)


# ---------------------------------------------------------------- tools

def _ddg(query: str) -> list[dict]:
    status, page = http(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}&kl=de-de")
    out = []
    if status != 200:
        return out
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=class="result__a"|$)',
                         page, re.S):
        href, title, rest = m.groups()
        q = parse_qs(urlparse(htmllib.unescape(href)).query)
        url = unquote(q["uddg"][0]) if "uddg" in q else htmllib.unescape(href)
        if "duckduckgo.com/y.js" in url:          # ad
            continue
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', rest, re.S)
        out.append({"title": title, "url": url, "snippet": sn.group(1) if sn else ""})
    return out


def _bing(query: str) -> list[dict]:
    status, page = http(f"https://www.bing.com/search?q={quote_plus(query)}&setlang=de&cc=de")
    out = []
    if status != 200:
        return out
    for block in page.split('<li class="b_algo"')[1:]:
        m = re.search(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        url = htmllib.unescape(m.group(1))
        u = parse_qs(urlparse(url).query).get("u", [""])[0]
        if "bing.com/ck/" in url and u.startswith("a1"):      # redirect: base64url target
            try:
                url = base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode()
            except Exception:  # noqa: BLE001
                continue
        sn = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        out.append({"title": m.group(2), "url": url, "snippet": sn.group(1) if sn else ""})
    return out


def search(query: str, n: int = 8) -> list[dict]:
    """Web search, German region: [{title, url, snippet}]. DuckDuckGo first, Bing when it
    rate-limits (it answers 202 after a few quick requests)."""
    for engine in (_ddg, _bing):
        hits = engine(query)
        if hits:
            clean = lambda x: htmllib.unescape(_TAG.sub("", x)).strip()  # noqa: E731
            return [{"title": clean(h["title"]), "url": h["url"], "snippet": clean(h["snippet"])[:300]}
                    for h in hits[:n]]
    return []


def read(url: str, find: str = "", max_chars: int = 3500) -> str:
    """Price-relevant excerpt of a page, or a one-line error the model can act on."""
    status, page = http(url)
    if status == 0:
        return f"ERROR: could not reach {urlparse(url).netloc}: {page[:120]}"
    if looks_blocked(status, page):
        return (f"BLOCKED: {urlparse(url).netloc} answered {status} with a bot challenge. "
                "Use another source.")
    if status >= 400:
        return f"ERROR: HTTP {status}"
    return excerpt(to_text(page), find, max_chars)
