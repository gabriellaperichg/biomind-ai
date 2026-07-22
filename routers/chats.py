from __future__ import annotations

import time
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from services.rag_service import RagServiceError, answer_question
from database import get_db
from dependencies import require_password_changed
from models import (
    AuditLog,
    Chat,
    Message,
    MessageSource,
    User,
    new_uuid,
    utc_now,
)


router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CreateChatRequest(BaseModel):
    title: str | None = Field(
        default=None,
        max_length=160,
    )


class AskRequest(BaseModel):
    texto: str = Field(
        min_length=3,
        max_length=10_000,
    )


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def add_audit_log(
    db: Session,
    *,
    user: User,
    action: str,
    entity_type: str,
    entity_id: str,
    metadata: dict | None = None,
) -> None:
    db.add(
        AuditLog(
            id=new_uuid(),
            user_id=user.id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            metadata_json=metadata or {},
        )
    )


def find_owned_chat(
    db: Session,
    *,
    chat_id: str,
    user_id: str,
    load_messages: bool = False,
) -> Chat:
    statement = select(Chat).where(
        Chat.id == chat_id,
        Chat.user_id == user_id,
        Chat.deleted_at.is_(None),
    )

    if load_messages:
        statement = statement.options(
            selectinload(Chat.messages).selectinload(
                Message.sources
            )
        )

    chat = db.scalar(statement)

    if chat is None:
        # Não informa se o chat existe e pertence a outra pessoa.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat não encontrado.",
        )

    return chat


def create_chat_title(question: str) -> str:
    """
    Cria um título simples a partir da primeira pergunta,
    sem chamar o Ollama novamente.
    """

    normalized = " ".join(question.strip().split())

    if len(normalized) <= 60:
        return normalized

    return f"{normalized[:57].rstrip()}..."


def serialize_source(source: MessageSource) -> dict:
    return {
        "id": source.id,
        "source_number": source.source_number,
        "source_name": source.source_name,
        "pages": source.pages,
        "similarity": source.similarity,
        "chunk_ids": source.chunk_ids_json,
        "created_at": source.created_at,
    }


def serialize_message(message: Message) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "status": message.status,
        "best_similarity": message.best_similarity,
        "disclaimer": message.disclaimer,
        "model_name": message.model_name,
        "embedding_model": message.embedding_model,
        "prompt_version": message.prompt_version,
        "generation_time_ms": message.generation_time_ms,
        "created_at": message.created_at,
        "sources": [
            serialize_source(source)
            for source in message.sources
        ],
    }


def serialize_chat_summary(chat: Chat) -> dict:
    return {
        "id": chat.id,
        "title": chat.title,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
    }


# ---------------------------------------------------------------------------
# Listar chats
# ---------------------------------------------------------------------------


@router.get("")
@router.get("/")
def list_chats(
    current_user: User = Depends(require_password_changed),
    db: Session = Depends(get_db),
) -> dict:
    chats = db.scalars(
        select(Chat)
        .where(
            Chat.user_id == current_user.id,
            Chat.deleted_at.is_(None),
        )
        .order_by(
            Chat.updated_at.desc(),
        )
    ).all()

    return {
        "status": "ok",
        "chats": [
            serialize_chat_summary(chat)
            for chat in chats
        ],
    }


# ---------------------------------------------------------------------------
# Criar chat
# ---------------------------------------------------------------------------


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
)
@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
)
def create_chat(
    payload: CreateChatRequest,
    current_user: User = Depends(require_password_changed),
    db: Session = Depends(get_db),
) -> dict:
    title = "Nova conversa"

    if payload.title:
        normalized_title = " ".join(
            payload.title.strip().split()
        )

        if normalized_title:
            title = normalized_title[:160]

    chat = Chat(
        id=new_uuid(),
        user_id=current_user.id,
        title=title,
        created_at=utc_now(),
        updated_at=utc_now(),
    )

    db.add(chat)

    add_audit_log(
        db,
        user=current_user,
        action="chat_created",
        entity_type="chat",
        entity_id=chat.id,
    )

    db.commit()
    db.refresh(chat)

    return {
        "status": "ok",
        "chat": serialize_chat_summary(chat),
    }


# ---------------------------------------------------------------------------
# Abrir chat e histórico
# ---------------------------------------------------------------------------


@router.get("/{chat_id}")
def get_chat(
    chat_id: str,
    current_user: User = Depends(require_password_changed),
    db: Session = Depends(get_db),
) -> dict:
    chat = find_owned_chat(
        db,
        chat_id=chat_id,
        user_id=current_user.id,
        load_messages=True,
    )

    return {
        "status": "ok",
        "chat": {
            **serialize_chat_summary(chat),
            "messages": [
                serialize_message(message)
                for message in chat.messages
            ],
        },
    }


# ---------------------------------------------------------------------------
# Excluir chat
# ---------------------------------------------------------------------------


@router.delete("/{chat_id}")
def delete_chat(
    chat_id: str,
    current_user: User = Depends(require_password_changed),
    db: Session = Depends(get_db),
) -> dict:
    chat = find_owned_chat(
        db,
        chat_id=chat_id,
        user_id=current_user.id,
    )

    chat.soft_delete()

    add_audit_log(
        db,
        user=current_user,
        action="chat_deleted",
        entity_type="chat",
        entity_id=chat.id,
    )

    db.commit()

    return {
        "status": "ok",
        "message": "Conversa excluída.",
    }


# ---------------------------------------------------------------------------
# Enviar pergunta
# ---------------------------------------------------------------------------


@router.post("/{chat_id}/ask")
def ask_chat(
    chat_id: str,
    payload: AskRequest,
    current_user: User = Depends(require_password_changed),
    db: Session = Depends(get_db),
) -> dict:
    chat = find_owned_chat(
        db,
        chat_id=chat_id,
        user_id=current_user.id,
    )

    question = payload.texto.strip()

    if len(question) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Descreva o caso com pelo menos 3 caracteres.",
        )

    # Salva a pergunta antes de chamar o RAG.
    user_message = Message(
        id=new_uuid(),
        chat_id=chat.id,
        role="user",
        content=question,
        status="ok",
        created_at=utc_now(),
    )

    db.add(user_message)

    if chat.title == "Nova conversa":
        chat.title = create_chat_title(question)

    chat.updated_at = utc_now()

    db.commit()
    db.refresh(user_message)

    start_time = time.perf_counter()

    try:
        result = answer_question(question)

    except RagServiceError as exc:
        measured_time_ms = round(
            (time.perf_counter() - start_time) * 1000
        )

        # Não salvar traceback ou conteúdo sensível.
        technical_error = type(exc).__name__

        assistant_error_message = Message(
            id=new_uuid(),
            chat_id=chat.id,
            role="assistant",
            content=(
                "Não foi possível gerar a resposta neste momento."
            ),
            status="erro_modelo",
            generation_time_ms=measured_time_ms,
            error_message=technical_error,
            created_at=utc_now(),
        )

        db.add(assistant_error_message)

        chat.updated_at = utc_now()

        add_audit_log(
            db,
            user=current_user,
            action="chat_answer_failed",
            entity_type="chat",
            entity_id=chat.id,
            metadata={
                "error_type": technical_error,
                "generation_time_ms": measured_time_ms,
            },
        )

        db.commit()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "O Biomind está temporariamente indisponível."
            ),
        ) from exc

    assistant_message = Message(
        id=new_uuid(),
        chat_id=chat.id,
        role="assistant",
        content=result["answer"],
        status=result["status"],
        best_similarity=result["best_similarity"],
        disclaimer=result["disclaimer"],
        model_name=result["model_name"],
        embedding_model=result["embedding_model"],
        prompt_version=result["prompt_version"],
        generation_time_ms=result["generation_time_ms"],
        created_at=utc_now(),
    )

    db.add(assistant_message)
    db.flush()

    saved_sources: list[MessageSource] = []

    for source_data in result["sources"]:
        message_source = MessageSource(
            id=new_uuid(),
            message_id=assistant_message.id,
            source_number=source_data["source_number"],
            source_name=source_data["source_name"],
            pages=source_data["pages"],
            similarity=source_data["similarity"],
            chunk_ids_json=source_data["chunk_ids"],
            created_at=utc_now(),
        )

        db.add(message_source)
        saved_sources.append(message_source)

    chat.updated_at = utc_now()

    add_audit_log(
        db,
        user=current_user,
        action="chat_answer_generated",
        entity_type="chat",
        entity_id=chat.id,
        metadata={
            "status": result["status"],
            "source_count": len(saved_sources),
            "generation_time_ms": result["generation_time_ms"],
        },
    )

    db.commit()
    db.refresh(assistant_message)

    return {
        "status": result["status"],
        "chat_id": chat.id,
        "user_message_id": user_message.id,
        "message_id": assistant_message.id,
        "answer": assistant_message.content,
        "disclaimer": assistant_message.disclaimer,
        "best_sim": assistant_message.best_similarity,
        "model_name": assistant_message.model_name,
        "embedding_model": assistant_message.embedding_model,
        "prompt_version": assistant_message.prompt_version,
        "generation_time_ms": (
            assistant_message.generation_time_ms
        ),
        "sources": [
            {
                "source_number": source.source_number,
                "source_name": source.source_name,
                "pages": source.pages,
                "similarity": source.similarity,
                "chunk_ids": source.chunk_ids_json,
            }
            for source in saved_sources
        ],
    }