from __future__ import annotations

from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import SESSION_COOKIE_NAME
from database import get_db
from models import User, UserSession, utc_now
from security import hash_session_token


def get_current_user(
    session_token: Annotated[
        str | None,
        Cookie(alias=SESSION_COOKIE_NAME),
    ] = None,
    db: Session = Depends(get_db),
) -> User:
    """
    Identifica a usuária por meio do cookie de sessão.

    Valida:
    - existência do cookie;
    - existência da sessão;
    - revogação;
    - expiração;
    - existência da usuária;
    - situação ativa da conta.
    """

    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Não autenticado.",
        )

    token_hash = hash_session_token(session_token)

    user_session = db.scalar(
        select(UserSession).where(
            UserSession.token_hash == token_hash,
            UserSession.revoked_at.is_(None),
        )
    )

    if user_session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessão inválida ou expirada.",
        )

    now = utc_now()

    if user_session.expires_at <= now:
        if user_session.revoked_at is None:
            user_session.revoked_at = now
            db.commit()

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessão inválida ou expirada.",
        )

    user = db.get(User, user_session.user_id)

    if user is None or not user.is_active:
        if user_session.revoked_at is None:
            user_session.revoked_at = now
            db.commit()

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessão inválida.",
        )

    # Atualiza a última utilização da sessão.
    user_session.last_seen_at = now
    db.commit()

    return user


def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Permite o acesso somente a administradoras.
    """

    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acesso não autorizado.",
        )

    return current_user