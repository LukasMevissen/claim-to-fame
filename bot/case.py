"""Descifrar y leer un caso: description.txt, policy.txt, invoices.pdf, imágenes."""
import glob
import os
import re
import subprocess

import config


def find_zip(game_id):
    """Los zips de Drive se llaman case_000.zip ... case_100.zip (supuesto: ronda n -> case_n)."""
    hits = sorted(glob.glob(os.path.join(config.CASES_DIR, f"case_{int(game_id):03d}.zip")))
    if not hits:
        raise FileNotFoundError(f"no encuentro el zip de la ronda {game_id} en {config.CASES_DIR}")
    return hits[0]


def extract(zip_path, key, game_id):
    out = os.path.join(config.WORK_DIR, str(game_id))
    os.makedirs(out, exist_ok=True)
    subprocess.run([config.SEVENZIP, "x", zip_path, f"-p{key}", f"-o{out}", "-y"],
                   check=True, capture_output=True, timeout=20)
    return out


def _read(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        return ""
    with open(hits[0], encoding="utf-8", errors="replace") as f:
        return f.read()


def invoice_text(pdf_path):
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        out = subprocess.run(["pdftotext", "-layout", pdf_path, "-"], capture_output=True, text=True)
        return out.stdout


def guess_indices(text):
    """Índices de línea por regex (para el envío de emergencia antes del LLM)."""
    idx = sorted({int(m.group(1)) for m in re.finditer(r"(?m)^\s*(\d{1,3})[\s\.\)|]", text)})
    # nos quedamos con la secuencia 1..n más larga
    n = 0
    while n + 1 in idx:
        n += 1
    return list(range(1, n + 1)) if n else idx


def load(root):
    pdfs = glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True)
    inv = invoice_text(pdfs[0]) if pdfs else ""
    imgs = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
                  + glob.glob(os.path.join(root, "**", "*.jp*g"), recursive=True))
    return {"description": _read(root, "description.txt"),
            "policy": _read(root, "policy.txt"),
            "invoice_text": inv,
            "indices": guess_indices(inv),
            "images": imgs}
