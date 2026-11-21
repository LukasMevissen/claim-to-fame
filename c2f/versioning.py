"""Version stamp attached to every run, submission and evaluation."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass

from . import __version__
from .config import REPO_ROOT, Config
from .extract import EXTRACTOR_VERSION


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


@dataclass(frozen=True)
class Stamp:
    agent_version: str
    extractor_version: str
    git_sha: str
    git_dirty: bool
    git_tag: str
    config_path: str
    config_sha: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"v{self.agent_version}-{self.git_sha[:7]}{'-dirty' if self.git_dirty else ''}"


def stamp(cfg: Config) -> Stamp:
    return Stamp(
        agent_version=__version__,
        extractor_version=EXTRACTOR_VERSION,
        git_sha=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        git_tag=_git("describe", "--tags", "--exact-match"),
        config_path=str(cfg.path.relative_to(REPO_ROOT)) if cfg.path.is_relative_to(REPO_ROOT) else str(cfg.path),
        config_sha=cfg.sha,
    )
