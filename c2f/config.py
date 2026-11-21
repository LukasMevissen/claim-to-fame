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
        learned = self.learned_opponents(p.pop("opponents_from", ""), p.pop("team_name", ""),
                                         p.pop("min_limit_obs", 3.0))
        opps = learned or opps
        return PricingParams(**p, **({"opponents": opps} if opps else {}))

    @staticmethod
    def learned_opponents(path: str, us: str, min_obs: float) -> tuple[OpponentType, ...]:
        """Opponent population fitted from the public transactions
        (scripts/update_opponents.py), or () when no belief file exists yet, in which
        case the hand-set [[pricing.opponents]] mix is used."""
        if not path:
            return ()
        f = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
        if not f.exists():
            return ()
        from .opponents import load_belief, opponent_types
        return opponent_types(load_belief(f), us=us, min_limits=min_obs)

    def section(self, name: str) -> dict:
        return self.raw.get(name, {})


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    data = path.read_bytes()
    return Config(tomllib.loads(data.decode()), path, hashlib.sha256(data).hexdigest()[:12])
