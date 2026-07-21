"""
Serviço central de auditoria.

A auditoria registra apenas dados técnicos mínimos. Nunca grave:
- senhas;
- tokens;
- cookies;
- perguntas clínicas completas;
- respostas clínicas completas;
- dados identificáveis do paciente.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from models import AuditLog, new_uuid


SENSITIVE_KEYS = {
    "password",
    "senha",
    "token",
    "cookie",
    "authorization",
    "question",
    "pergunta",
    "answer",
    "resposta",
    "content",
    "conteudo",
    "texto",
    "patient",
    "paciente",
    "cpf",
    "telefone",
    "address",
    "endereco",
}


def sanitize_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Remove campos potencialmente sensíveis antes de salvar a auditoria.

    O filtro é defensivo; as rotas também devem evitar enviar conteúdo
    clínico para esta função.
    """

    if not metadata:
        return {}

    sanitized: dict[str, Any] = {}

    for key, value in metadata.items():
        normalized_key = str(key).strip().lower()

        if normalized_key in SENSITIVE_KEYS:
            sanitized[str(key)] = "[removido]"
            continue

        if isinstance(value, Mapping):
            sanitized[str(key)] = sanitize_metadata(value)
            continue

        if isinstance(value, (list, tuple, set)):
            sanitized[str(key)] = [
                _sanitize_collection_item(item)
                for item in value
            ]
            continue

        sanitized[str(key)] = _safe_scalar(value)

    return sanitized


def _sanitize_collection_item(value: Any) -> Any:
    if isinstance(value, Mapping):
        return sanitize_metadata(value)

    return _safe_scalar(value)


def _safe_scalar(value: Any) -> Any:
    """
    Mantém apenas valores simples e limita textos para evitar logs enormes.
    """

    if value is None or isinstance(value, (bool, int, float)):
        return value

    text = str(value)

    if len(text) > 500:
        return f"{text[:497]}..."

    return text


def add_audit_log(
    db: Session,
    *,
    action: str,
    user_id: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AuditLog:
    """
    Adiciona um registro de auditoria à transação atual.

    Esta função NÃO executa commit. A rota ou serviço chamador controla
    a transação juntamente com as outras alterações.
    """

    audit_log = AuditLog(
        id=new_uuid(),
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        metadata_json=sanitize_metadata(metadata),
    )

    db.add(audit_log)

    return audit_log
