from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from pwdlib import PasswordHash

from config import SESSION_DURATION_HOURS


password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return password_hasher.verify(password, password_hash)


def generate_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_expiration() -> datetime:
    return datetime.now(timezone.utc) + timedelta(
        hours=SESSION_DURATION_HOURS
    )