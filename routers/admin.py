from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import get_db
from dependencies import require_admin
from models import (
    AuditLog,
    User,
    UserSession,
    new_uuid,
    utc_now,
)
from security import hash_password


router = APIRouter()


MIN_PASSWORD_LENGTH = 10


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CreateUserRequest(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=120,
    )

    email: EmailStr

    password: str = Field(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=200,
    )

    role: Literal["admin", "biomedica"] = "biomedica"

    is_active: bool = True

    must_change_password: bool = True


class UpdateUserRequest(BaseModel):
    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=120,
    )

    email: EmailStr | None = None

    role: Literal["admin", "biomedica"] | None = None

    is_active: bool | None = None


class ResetPasswordRequest(BaseModel):
    temporary_password: str = Field(
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
        "failed_login_attempts": user.failed_login_attempts,
        "locked_until": user.locked_until,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
        "last_login_at": user.last_login_at,
    }


def add_audit_log(
    db: Session,
    *,
    admin: User,
    action: str,
    entity_type: str,
    entity_id: str,
    metadata: dict | None = None,
) -> None:
    db.add(
        AuditLog(
            id=new_uuid(),
            user_id=admin.id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            metadata_json=metadata or {},
        )
    )


def find_user_or_404(
    db: Session,
    user_id: str,
) -> User:
    user = db.get(User, user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuário não encontrado.",
        )

    return user


def revoke_user_sessions(
    db: Session,
    user_id: str,
) -> None:
    db.execute(
        update(UserSession)
        .where(
            UserSession.user_id == user_id,
            UserSession.revoked_at.is_(None),
        )
        .values(
            revoked_at=utc_now(),
        )
    )


# ---------------------------------------------------------------------------
# Listagem de usuários
# ---------------------------------------------------------------------------


@router.get("/users")
def list_users(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    users = db.scalars(
        select(User).order_by(
            User.name.asc(),
            User.email.asc(),
        )
    ).all()

    return {
        "status": "ok",
        "users": [
            serialize_user(user)
            for user in users
        ],
    }


# ---------------------------------------------------------------------------
# Criação de usuário
# ---------------------------------------------------------------------------


@router.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    payload: CreateUserRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    normalized_email = payload.email.strip().lower()

    existing_user = db.scalar(
        select(User).where(
            User.email == normalized_email,
        )
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Já existe um usuário com este e-mail.",
        )

    user = User(
        id=new_uuid(),
        name=payload.name.strip(),
        email=normalized_email,
        password_hash=hash_password(payload.password),
        role=payload.role,
        is_active=payload.is_active,
        must_change_password=payload.must_change_password,
        failed_login_attempts=0,
        created_at=utc_now(),
        updated_at=utc_now(),
    )

    db.add(user)

    try:
        db.flush()

        add_audit_log(
            db,
            admin=admin,
            action="user_created",
            entity_type="user",
            entity_id=user.id,
            metadata={
                "role": user.role,
                "is_active": user.is_active,
            },
        )

        db.commit()
        db.refresh(user)

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Não foi possível criar o usuário.",
        ) from exc

    return {
        "status": "ok",
        "user": serialize_user(user),
    }


# ---------------------------------------------------------------------------
# Atualização de usuário
# ---------------------------------------------------------------------------


@router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    payload: UpdateUserRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    user = find_user_or_404(db, user_id)

    updated_fields: list[str] = []

    if payload.name is not None:
        user.name = payload.name.strip()
        updated_fields.append("name")

    if payload.email is not None:
        normalized_email = payload.email.strip().lower()

        email_owner = db.scalar(
            select(User).where(
                User.email == normalized_email,
                User.id != user.id,
            )
        )

        if email_owner is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Já existe um usuário com este e-mail.",
            )

        user.email = normalized_email
        updated_fields.append("email")

    if payload.role is not None:
        if user.id == admin.id and payload.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "A administradora não pode remover "
                    "o próprio perfil administrativo."
                ),
            )

        user.role = payload.role
        updated_fields.append("role")

    if payload.is_active is not None:
        if user.id == admin.id and not payload.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "A administradora não pode desativar "
                    "a própria conta."
                ),
            )

        user.is_active = payload.is_active
        updated_fields.append("is_active")

        if not payload.is_active:
            revoke_user_sessions(
                db,
                user.id,
            )

    user.updated_at = utc_now()

    add_audit_log(
        db,
        admin=admin,
        action="user_updated",
        entity_type="user",
        entity_id=user.id,
        metadata={
            "updated_fields": updated_fields,
        },
    )

    try:
        db.commit()
        db.refresh(user)

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Não foi possível atualizar o usuário.",
        ) from exc

    return {
        "status": "ok",
        "user": serialize_user(user),
    }


# ---------------------------------------------------------------------------
# Redefinição de senha
# ---------------------------------------------------------------------------


@router.post("/users/{user_id}/reset-password")
def reset_user_password(
    user_id: str,
    payload: ResetPasswordRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    user = find_user_or_404(db, user_id)

    user.password_hash = hash_password(
        payload.temporary_password
    )

    user.must_change_password = True
    user.failed_login_attempts = 0
    user.locked_until = None
    user.updated_at = utc_now()

    revoke_user_sessions(
        db,
        user.id,
    )

    add_audit_log(
        db,
        admin=admin,
        action="password_reset_by_admin",
        entity_type="user",
        entity_id=user.id,
    )

    db.commit()

    return {
        "status": "ok",
        "message": (
            "Senha temporária definida. "
            "O usuário deverá alterá-la no próximo acesso."
        ),
    }


# ---------------------------------------------------------------------------
# Revogação de sessões
# ---------------------------------------------------------------------------


@router.post("/users/{user_id}/revoke-sessions")
def revoke_sessions(
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    user = find_user_or_404(db, user_id)

    revoke_user_sessions(
        db,
        user.id,
    )

    add_audit_log(
        db,
        admin=admin,
        action="sessions_revoked_by_admin",
        entity_type="user",
        entity_id=user.id,
    )

    db.commit()

    return {
        "status": "ok",
        "message": "Sessões encerradas.",
    }