"""
Modelos do banco de dados da aplicação Biomind.

Este banco guarda:
- usuários;
- sessões de login;
- chats;
- mensagens;
- fontes das respostas;
- registros de auditoria.

O banco da aplicação deve permanecer separado do ChromaDB.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def new_uuid() -> str:
    """
    Gera um identificador UUID em formato texto.

    Exemplo:
    7dc10ed2-254a-4cc3-a881-0a890fe9eec4
    """

    return str(uuid4())


def utc_now() -> datetime:
    """
    Retorna a data e hora atual em UTC sem timezone anexado.

    O SQLite não preserva timezone de forma confiável. Por isso, todas as
    datas são gravadas em UTC e tratadas como UTC pela aplicação.
    """

    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Usuários
# ---------------------------------------------------------------------------


class User(Base):
    """
    Usuária autorizada a acessar o Biomind.

    Perfis permitidos:
    - admin
    - biomedica
    """

    __tablename__ = "users"

    __table_args__ = (
        CheckConstraint(
            "role IN ('admin', 'biomedica')",
            name="ck_users_role",
        ),
        CheckConstraint(
            "failed_login_attempts >= 0",
            name="ck_users_failed_login_attempts_positive",
        ),
        Index(
            "ix_users_active_role",
            "is_active",
            "role",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )

    # NOCASE impede, no SQLite, que estes dois e-mails sejam diferentes:
    # USUARIO@EMPRESA.COM
    # usuario@empresa.com
    email: Mapped[str] = mapped_column(
        String(255, collation="NOCASE"),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="biomedica",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    must_change_password: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    # -----------------------------------------------------------------------
    # Relacionamentos
    # -----------------------------------------------------------------------

    sessions: Mapped[list["UserSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    chats: Mapped[list["Chat"]] = relationship(
        back_populates="user",
        passive_deletes=True,
    )

    audit_logs: Mapped[list["AuditLog"]] = relationship(
        back_populates="user",
        passive_deletes=True,
    )

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_locked(self) -> bool:
        if self.locked_until is None:
            return False

        return self.locked_until > utc_now()

    def normalize_email(self) -> None:
        """
        Normaliza o e-mail antes de salvar.

        Deve ser chamada pelos serviços ou rotas que criam e atualizam usuários.
        """

        self.email = self.email.strip().lower()


# ---------------------------------------------------------------------------
# Sessões
# ---------------------------------------------------------------------------


class UserSession(Base):
    """
    Sessão autenticada de uma usuária.

    O token original é enviado ao navegador em um cookie HTTP-only.

    O banco armazena apenas o SHA-256 do token.
    """

    __tablename__ = "sessions"

    __table_args__ = (
        Index(
            "ix_sessions_user_expires",
            "user_id",
            "expires_at",
        ),
        Index(
            "ix_sessions_user_revoked",
            "user_id",
            "revoked_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    token_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        index=True,
    )

    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    ip_address: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    # -----------------------------------------------------------------------
    # Relacionamentos
    # -----------------------------------------------------------------------

    user: Mapped["User"] = relationship(
        back_populates="sessions",
    )

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    @property
    def is_expired(self) -> bool:
        return self.expires_at <= utc_now()

    @property
    def is_valid(self) -> bool:
        return not self.is_revoked and not self.is_expired


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------


class Chat(Base):
    """
    Conversa pertencente a uma usuária.

    A exclusão é lógica: deleted_at recebe uma data, mas o registro continua
    no banco.
    """

    __tablename__ = "chats"

    __table_args__ = (
        Index(
            "ix_chats_user_updated",
            "user_id",
            "updated_at",
        ),
        Index(
            "ix_chats_user_deleted",
            "user_id",
            "deleted_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
        default="Nova conversa",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        index=True,
    )

    # -----------------------------------------------------------------------
    # Relacionamentos
    # -----------------------------------------------------------------------

    user: Mapped["User"] = relationship(
        back_populates="chats",
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        self.deleted_at = utc_now()
        self.updated_at = utc_now()


# ---------------------------------------------------------------------------
# Mensagens
# ---------------------------------------------------------------------------


class Message(Base):
    """
    Mensagem enviada pela usuária ou resposta gerada pelo Biomind.

    Roles:
    - user
    - assistant

    Exemplos de status:
    - processando
    - ok
    - sem_material
    - erro_modelo
    - erro_indice
    - erro_interno
    """

    __tablename__ = "messages"

    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_messages_role",
        ),
        Index(
            "ix_messages_chat_created",
            "chat_id",
            "created_at",
        ),
        Index(
            "ix_messages_chat_role",
            "chat_id",
            "role",
        ),
        Index(
            "ix_messages_status_created",
            "status",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    chat_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "chats.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="ok",
        index=True,
    )

    best_similarity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    disclaimer: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    prompt_version: Mapped[str | None] = mapped_column(
        String(80),
        nullable=True,
    )

    generation_time_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
    )

    # -----------------------------------------------------------------------
    # Relacionamentos
    # -----------------------------------------------------------------------

    chat: Mapped["Chat"] = relationship(
        back_populates="messages",
    )

    sources: Mapped[list["MessageSource"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MessageSource.source_number",
    )


# ---------------------------------------------------------------------------
# Fontes das mensagens
# ---------------------------------------------------------------------------


class MessageSource(Base):
    """
    Fonte documental utilizada em uma resposta do Biomind.

    Normalmente deve ser associada apenas a mensagens com role='assistant'.
    """

    __tablename__ = "message_sources"

    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "source_number",
            name="uq_message_sources_message_number",
        ),
        CheckConstraint(
            "source_number >= 1",
            name="ck_message_sources_number_positive",
        ),
        Index(
            "ix_message_sources_message",
            "message_id",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    message_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "messages.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    source_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    source_name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    pages: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )

    similarity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    chunk_ids_json: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
    )

    # -----------------------------------------------------------------------
    # Relacionamentos
    # -----------------------------------------------------------------------

    message: Mapped["Message"] = relationship(
        back_populates="sources",
    )


# ---------------------------------------------------------------------------
# Auditoria
# ---------------------------------------------------------------------------


class AuditLog(Base):
    """
    Registro técnico de ações relevantes.

    Não deve armazenar:
    - senha;
    - token;
    - cookie;
    - pergunta clínica completa;
    - resposta clínica completa;
    - dados identificáveis do paciente.
    """

    __tablename__ = "audit_logs"

    __table_args__ = (
        Index(
            "ix_audit_logs_user_created",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_audit_logs_action_created",
            "action",
            "created_at",
        ),
        Index(
            "ix_audit_logs_entity",
            "entity_type",
            "entity_id",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
    )

    # Pode ser nulo, por exemplo, em uma tentativa de login com e-mail
    # inexistente.
    user_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    entity_type: Mapped[str | None] = mapped_column(
        String(80),
        nullable=True,
    )

    entity_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        index=True,
    )

    # -----------------------------------------------------------------------
    # Relacionamentos
    # -----------------------------------------------------------------------

    user: Mapped["User | None"] = relationship(
        back_populates="audit_logs",
    )


# ---------------------------------------------------------------------------
# Exportações
# ---------------------------------------------------------------------------

__all__ = [
    "User",
    "UserSession",
    "Chat",
    "Message",
    "MessageSource",
    "AuditLog",
    "new_uuid",
    "utc_now",
]