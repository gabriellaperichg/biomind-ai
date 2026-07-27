from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "biomind_app.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

SESSION_COOKIE_NAME = "biomind_session"
SESSION_DURATION_HOURS = 8

# False apenas durante desenvolvimento HTTP local.
# Antes de utilizar HTTPS com dados reais, alterar para True.
COOKIE_SECURE = os.getenv("BIOMIND_COOKIE_SECURE", "false").lower() == "true"