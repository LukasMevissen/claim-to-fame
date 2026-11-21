from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .pricing import OpponentType, PricingParams

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


@dataclass(frozen=True)
class Config:
    raw: dict
    path: Path
    sha: str

    @property
    def pricing(self) -> PricingParams:
        p = dict(self.raw.get("pricing", {}))
        opps = tuple(OpponentType(**o) for o in p.pop("opponents", []))
        return PricingParams(**p, **({"opponents": opps} if opps else {}))

    def section(self, name: str) -> dict:
        return self.raw.get(name, {})


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    data = path.read_bytes()
    return Config(tomllib.loads(data.decode()), path, hashlib.sha256(data).hexdigest()[:12])
