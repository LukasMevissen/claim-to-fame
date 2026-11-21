"""Load cases: decrypt the encrypted zip with 7-Zip and read its files."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Case:
    case_id: str
    policy: str
    description: str
    invoice_pdf: Path
    invoice_text: str          # text extracted from invoices.pdf ("" if not possible)
    image: Path | None


def decrypt(zip_path: Path, password: str, out_dir: Path) -> Path:
    """Decrypt an AES case zip into out_dir (pyzipper; falls back to 7-Zip if missing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pyzipper
    except ImportError:
        pyzipper = None
    if pyzipper is not None:
        with pyzipper.AESZipFile(zip_path) as z:
            z.setpassword(password.encode())
            z.extractall(out_dir)
        return _case_root(out_dir)
    exe = next((e for e in ("7z", "7zz", "7za") if shutil.which(e)), None)
    if exe is None:
        raise RuntimeError("Neither pyzipper nor 7-Zip available (pip install pyzipper).")
    subprocess.run([exe, "x", "-y", f"-p{password}", f"-o{out_dir}", str(zip_path)],
                   check=True, capture_output=True, timeout=20)
    return _case_root(out_dir)


def _case_root(d: Path) -> Path:
    """Directory that actually contains policy.txt (zips sometimes nest a folder)."""
    hits = list(d.rglob("policy.txt"))
    if not hits:
        raise FileNotFoundError(f"no policy.txt under {d}")
    return hits[0].parent


def pdf_text(pdf: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages).strip()
    except Exception:
        return ""


# "<n> <description of 10+ chars>"; quantity lines such as "2 hrs" or "1 pcs" are too short.
_POS_LINE = re.compile(r"^\s*(?:Pos(?:ition)?\.?\s*)?(\d{1,3})[\s.):-]+\S.{9,}", re.IGNORECASE)


def invoice_positions(text: str) -> list[int]:
    """Line item numbers 1..n read from the invoice text, in order ([] if none found).

    Only the running sequence 1, 2, 3, ... is accepted and the number must be followed by
    a description, so dates, postcodes and quantity lines ("2 hrs") are skipped.
    """
    expected, out = 1, []
    for line in text.splitlines():
        m = _POS_LINE.match(line)
        if m and int(m.group(1)) == expected:
            out.append(expected)
            expected += 1
    return out


def load_case(case_dir: Path, case_id: str | None = None) -> Case:
    root = _case_root(Path(case_dir))
    img = next((p for p in sorted(root.iterdir())
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")), None)
    pdf = root / "invoices.pdf"
    return Case(
        case_id=case_id or root.name,
        policy=(root / "policy.txt").read_text(encoding="utf-8", errors="replace"),
        description=(root / "description.txt").read_text(encoding="utf-8", errors="replace"),
        invoice_pdf=pdf,
        invoice_text=pdf_text(pdf) if pdf.exists() else "",
        image=img,
    )
