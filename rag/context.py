from __future__ import annotations

from collections import defaultdict
import html
import os
from typing import Any


MAX_CONTEXT_CHARS = max(
    1000,
    int(os.getenv("BIOMIND_MAX_CONTEXT_CHARS", "10000")),
)


def fetch_parent_context(col, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Substitui children isolados pelos blocos completos do mesmo parent."""
    expanded: list[dict[str, Any]] = []
    seen_parent: set[str] = set()

    for item in selected:
        metadata = item.get("metadata") or {}
        parent_id = str(metadata.get("parent_id") or "")

        if not parent_id or parent_id in seen_parent:
            if not parent_id:
                expanded.append(item)
            continue

        seen_parent.add(parent_id)
        fetched = col.get(
            where={"parent_id": parent_id},
            include=["documents", "metadatas"],
        )

        ids = fetched.get("ids") or []
        docs = fetched.get("documents") or []
        metas = fetched.get("metadatas") or []

        children = []
        for rid, text, meta in zip(ids, docs, metas):
            if text is None or meta is None:
                continue
            children.append(
                {
                    "id": str(rid),
                    "text": str(text),
                    "metadata": meta,
                }
            )

        children.sort(key=lambda x: int(x["metadata"].get("child_index", 0)))
        merged = dict(item)
        if children:
            merged["text"] = "\n\n".join(child["text"] for child in children)
            merged["parent_child_ids"] = [child["id"] for child in children]
            merged["metadata"] = dict(metadata)
            merged["metadata"]["page_start"] = min(
                int(child["metadata"].get("page_start", metadata.get("page_start", 1)))
                for child in children
            )
            merged["metadata"]["page_end"] = max(
                int(child["metadata"].get("page_end", metadata.get("page_end", 1)))
                for child in children
            )
        expanded.append(merged)

    return expanded


def build_context(trechos: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    used: list[dict[str, Any]] = []
    total = 0

    for idx, trecho in enumerate(trechos, start=1):
        meta = trecho.get("metadata") or {}
        source = html.escape(
            str(meta.get("source") or trecho.get("source") or "Fonte não identificada"),
            quote=True,
        )
        pages = str(
            meta.get("page_start", trecho.get("pages", 1))
        )
        page_end = str(meta.get("page_end", pages))
        pages = pages if pages == page_end else f"{pages}-{page_end}"

        section = html.escape(
            str(meta.get("section_path") or meta.get("section") or "geral"),
            quote=True,
        )

        block = (
            f'<trecho id="{idx}" fonte="{source}" paginas="{html.escape(pages)}" '
            f'secao="{section}" '
            f'tipo_documento="{html.escape(str(meta.get("document_type", "unknown")))}" '
            f'tipo_conteudo="{html.escape(str(meta.get("content_type", "general")))}" '
            f'tema="{html.escape(str(meta.get("topic", "general_trichology")))}">\n'
            f'{trecho["text"]}\n'
            f'</trecho>'
        )

        if blocks and total + len(block) > MAX_CONTEXT_CHARS:
            continue

        if not blocks and len(block) > MAX_CONTEXT_CHARS:
            block = block[: MAX_CONTEXT_CHARS - 100] + "\n[trecho truncado]\n</trecho>"

        blocks.append(block)
        used.append(trecho)
        total += len(block)

    return "\n\n".join(blocks), used
