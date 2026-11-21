"""Load cases: decrypt the encrypted zip with 7-Zip and read its files."""

from __future__ import annotations

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
    """Decrypt a case zip into out_dir using 7-Zip (7z, 7za or 7zz)."""
    exe = next((e for e in ("7z", "7zz", "7za") if shutil.which(e)), None)
    if exe is None:
        raise RuntimeError("7-Zip not found. Install it (pixi/brew/apt: p7zip).")
    out_dir.mkdir(parents=True, exist_ok=True)
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


def load_case(case_dir: Path, case_id: str | None = None) -> Case:
    root = _case_root(Path(case_dir))
    img = next((p for p in root.glob("images.*")), None)
    pdf = root / "invoices.pdf"
    return Case(
        case_id=case_id or root.name,
        policy=(root / "policy.txt").read_text(encoding="utf-8", errors="replace"),
        description=(root / "description.txt").read_text(encoding="utf-8", errors="replace"),
        invoice_pdf=pdf,
        invoice_text=pdf_text(pdf) if pdf.exists() else "",
        image=img,
    )
