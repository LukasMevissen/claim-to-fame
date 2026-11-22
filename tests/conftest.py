"""Shared fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _live_dir_in_tmp(monkeypatch, tmp_path):
    """play_game writes results/live/game<N>__<role>.json; tests must never touch the repo's."""
    import c2f.__main__ as m
    monkeypatch.setattr(m, "LIVE_DIR", tmp_path / "live")
