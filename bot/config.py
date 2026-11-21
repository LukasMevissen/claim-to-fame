"""Configuración. Reglas sacadas del Deep Dive de Claim to Fame."""
import os

# ---------------------------------------------------------------- plataforma
API_BASE = os.getenv("C2F_API", "https://c2f.public.quantco.cloud")
TEAM_KEY = os.getenv("TEAM_API_KEY", os.getenv("C2F_KEY", ""))
CASES_DIR = os.getenv("C2F_CASES", "cases")   # zips cifrados descargados de Google Drive
WORK_DIR = os.getenv("C2F_WORK", "work")      # casos descomprimidos
DATA_DIR = os.getenv("C2F_DATA", "data")      # un JSON por ronda: predicciones, envíos, resultados
SEVENZIP = os.getenv("SEVENZIP", "7z")

# ---------------------------------------------------------------- reglas
VAT = 1.19            # se envía el total BRUTO de la línea: cantidad x precio unitario x 1.19
PENALTY = 1.5         # rechazar un cargo justo cuesta 1.5 x a
# Tope c de pago (GAME_DESCRIPTION): c >= 4t y nunca por debajo de un suelo absoluto desconocido.
# Usamos c = max(4t, CAP_FLOOR) como estimación conservadora.
CAP_MULT = 4.0
CAP_FLOOR = float(os.getenv("C2F_CAP_FLOOR", "100"))   # SUPUESTO: ajustar si se averigua

# ---------------------------------------------------------------- tiempos
ROUND_S = 60
POLL_S = 1.0
FINAL_SUBMIT_BY_S = 50    # el último envío "bueno" antes de este segundo; reenviar es gratis

# ---------------------------------------------------------------- LLM
# OpenAI (Chat Completions). Keys: OPENAI_KEY (starter kit) u OPENAI_API_KEY.
OPENAI_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
LLM_FAST_MODEL = os.getenv("LLM_FAST_MODEL", "gpt-4o-mini")   # 1st pass, text only
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")                  # 2nd pass, with image
LLM_TIMEOUT_S = 35

# ---------------------------------------------------------------- decisión
N_SAMPLES = 1500
GRID_SIZE = 200
EMERGENCY_B = float(os.getenv("C2F_EMERGENCY_B", "1e5"))  # sin estimación: aceptar casi todo (ver README)
