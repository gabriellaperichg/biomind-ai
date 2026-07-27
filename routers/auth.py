from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from config import (
    COOKIE_SECURE,
    SESSION_COOKIE_NAME,
    SESSION_DURATION_HOURS,
)
from database import get_db
from dependencies import get_current_user
from models import (
    AuditLog,
    User,
    UserSession,
    new_uuid,
    utc_now,
)
from security import (
    generate_session_token,
    hash_password,
    hash_session_token,
    session_expiration,
    verify_password,
)


router = APIRouter()


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

MAX_FAILED_ATTEMPTS = 5
LOCK_MINUTES = 15
MIN_PASSWORD_LENGTH = 10


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(
        min_length=1,
        max_length=200,
    )


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(
        min_length=1,
        max_length=200,
    )

    new_password: str = Field(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=200,
    )

    confirm_password: str = Field(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=200,
    )


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def serialize_user(user: User) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "must_change_password": user.must_change_password,
        "created_at": user.created_at,
        "last_login_at": user.last_login_at,
    }


def add_audit_log(
    db: Session,
    *,
    action: str,
    user_id: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    log = AuditLog(
        id=new_uuid(),
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        metadata_json=metadata or {},
    )

    db.add(log)


def get_client_ip(request: Request) -> str | None:
    """
    Obtém o IP da conexão.

    No futuro, se houver proxy reverso, será necessário configurar
    corretamente o uso de X-Forwarded-For.
    """

    if request.client is None:
        return None

    return request.client.host


def invalid_credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="E-mail ou senha inválidos.",
    )


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


@router.post("/login")
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> dict:
    normalized_email = payload.email.strip().lower()
    now = utc_now()

    user = db.scalar(
        select(User).where(
            User.email == normalized_email,
        )
    )

    # Não revelar se o e-mail existe.
    if user is None:
        add_audit_log(
            db,
            action="login_failed",
            metadata={
                "reason": "invalid_credentials",
                "ip_address": get_client_ip(request),
            },
        )

        db.commit()

        raise invalid_credentials_exception()

    if not user.is_active:
        add_audit_log(
            db,
            action="login_failed",
            user_id=user.id,
            entity_type="user",
            entity_id=user.id,
            metadata={
                "reason": "inactive_user",
                "ip_address": get_client_ip(request),
            },
        )

        db.commit()

        raise invalid_credentials_exception()

    if user.locked_until is not None and user.locked_until > now:
        add_audit_log(
            db,
            action="login_blocked",
            user_id=user.id,
            entity_type="user",
            entity_id=user.id,
            metadata={
                "reason": "temporary_lock",
                "ip_address": get_client_ip(request),
            },
        )

        db.commit()

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Muitas tentativas de acesso. "
                "Tente novamente mais tarde."
            ),
        )

    password_is_valid = verify_password(
        payload.password,
        user.password_hash,
    )

    if not password_is_valid:
        user.failed_login_attempts += 1

        if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = now + timedelta(
                minutes=LOCK_MINUTES,
            )

        add_audit_log(
            db,
            action="login_failed",
            user_id=user.id,
            entity_type="user",
            entity_id=user.id,
            metadata={
                "reason": "invalid_credentials",
                "failed_attempts": user.failed_login_attempts,
                "ip_address": get_client_ip(request),
            },
        )

        db.commit()

        raise invalid_credentials_exception()

    # Login válido.
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = now

    raw_token = generate_session_token()
    token_hash = hash_session_token(raw_token)

    user_session = UserSession(
        id=new_uuid(),
        user_id=user.id,
        token_hash=token_hash,
        created_at=now,
        expires_at=session_expiration(),
        last_seen_at=now,
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    db.add(user_session)

    add_audit_log(
        db,
        action="login_success",
        user_id=user.id,
        entity_type="session",
        entity_id=user_session.id,
        metadata={
            "ip_address": get_client_ip(request),
        },
    )

    db.commit()

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=raw_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=SESSION_DURATION_HOURS * 60 * 60,
        path="/",
    )

    return {
        "status": "ok",
        "user": serialize_user(user),
        "must_change_password": user.must_change_password,
    }


# ---------------------------------------------------------------------------
# Usuária autenticada
# ---------------------------------------------------------------------------


@router.get("/me")
def get_authenticated_user(
    current_user: User = Depends(get_current_user),
) -> dict:
    return {
        "status": "ok",
        "user": serialize_user(current_user),
    }


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


@router.post("/logout")
def logout(
    response: Response,
    session_token: Annotated[
        str | None,
        Cookie(alias=SESSION_COOKIE_NAME),
    ] = None,
    db: Session = Depends(get_db),
) -> dict:
    if session_token:
        token_hash = hash_session_token(session_token)

        user_session = db.scalar(
            select(UserSession).where(
                UserSession.token_hash == token_hash,
                UserSession.revoked_at.is_(None),
            )
        )

        if user_session is not None:
            user_session.revoked_at = utc_now()

            add_audit_log(
                db,
                action="logout",
                user_id=user_session.user_id,
                entity_type="session",
                entity_id=user_session.id,
            )

            db.commit()

    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
    )

    return {
        "status": "ok",
        "message": "Sessão encerrada.",
    }


# ---------------------------------------------------------------------------
# Alteração de senha
# ---------------------------------------------------------------------------


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if payload.new_password != payload.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A confirmação da nova senha não corresponde.",
        )

    if not verify_password(
        payload.current_password,
        current_user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A senha atual está incorreta.",
        )

    if verify_password(
        payload.new_password,
        current_user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A nova senha deve ser diferente da senha atual.",
        )

    current_user.password_hash = hash_password(
        payload.new_password
    )

    current_user.must_change_password = False
    current_user.failed_login_attempts = 0
    current_user.locked_until = None
    current_user.updated_at = utc_now()

    # Revoga todas as sessões, inclusive a atual.
    db.execute(
        update(UserSession)
        .where(
            UserSession.user_id == current_user.id,
            UserSession.revoked_at.is_(None),
        )
        .values(
            revoked_at=utc_now(),
        )
    )

    add_audit_log(
        db,
        action="password_changed",
        user_id=current_user.id,
        entity_type="user",
        entity_id=current_user.id,
    )

    db.commit()

    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
    )

    return {
        "status": "ok",
        "message": (
            "Senha alterada. Entre novamente com a nova senha."
        ),
        "requires_login": True,
    }