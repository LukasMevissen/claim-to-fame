"""Cliente de la plataforma, según API_HANDBOOK.md y starter_script.py."""
import time
from datetime import datetime

import requests

import config

S = requests.Session()


def _h():
    return {"X-API-Key": config.TEAM_KEY}


def _ts(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()


def list_games():
    """GET /api/games/list -> [{"id": 1, "start_time": "2025-06-01T12:00:00Z"}, ...]"""
    r = S.get(f"{config.API_BASE}/api/games/list", headers=_h(), timeout=5)
    r.raise_for_status()
    raw = r.json()
    games = raw.get("games", raw) if isinstance(raw, dict) else raw
    out = []
    for g in games:
        start = _ts(g.get("start_time") or g.get("start") or g.get("starts_at"))
        end = _ts(g.get("end_time") or g.get("end") or g.get("deadline")) or (start + config.ROUND_S if start else None)
        out.append({"id": g.get("id") or g.get("game_id"), "start": start, "end": end, "raw": g})
    return out


def get_key(game_id, wait_s=15):
    """Pide la clave hasta que esté disponible."""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        r = S.get(f"{config.API_BASE}/api/games/{game_id}/key", headers=_h(), timeout=5)
        if r.ok and r.content:
            key = r.json().get("decryption_key")      # 403 = aún no ha empezado
            if key:
                return str(key)
        time.sleep(0.5)
    raise TimeoutError("la clave no llegó")


def submit(game_id, decisions):
    """decisions: {index: {"charge_price": a, "acceptance_limit": b}}"""
    body = [{"index": int(i), "charge_price": float(d["charge_price"]),
             "acceptance_limit": float(d["acceptance_limit"])} for i, d in sorted(decisions.items())]
    r = S.put(f"{config.API_BASE}/api/games/{game_id}/submissions", headers=_h(), json=body, timeout=5)
    r.raise_for_status()
    return r.json() if r.content else {}


def get_results(game_id):
    """La API no tiene endpoint de resultados. El leaderboard público
    (c2f.public.quantco.cloud/leaderboard, pestañas Games / Transactions) se rellena al
    acabar las rondas: mirar ahí qué publica y de dónde lo carga. Confirmar con los
    organizadores que leer esa página es correcto (las reglas prohíben sacar datos
    de otros equipos atacando la API). Debe devolver, cuando la ronda esté resuelta:
        {index: {"t": valor_justo, "others": {equipo: {"a": .., "b": ..}}}}
    o None si aún no hay resultados. Rellenar con lo que exponga la plataforma/leaderboard."""
    return None
