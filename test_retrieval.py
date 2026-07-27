"""Inspeciona a recuperação e o reranking sem chamar o Ollama.

Uso:
    python test_retrieval.py "descrição do caso"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import biomind_core as core


DEFAULT_QUESTION = (
    "Paciente feminina, queda na coroa, couro cabeludo oleoso e escamando, "
    "toma anticoncepcional e antidepressivo."
)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    report = core.debug_recuperacao(question)

    print("=" * 88)
    print("CONFIGURAÇÃO")
    print("=" * 88)
    print(json.dumps(report["config"], ensure_ascii=False, indent=2))

    print("\n" + "=" * 88)
    print("VARIAÇÕES DE CONSULTA")
    print("=" * 88)
    for index, variant in enumerate(report["query_variants"], start=1):
        print(f"[{index}] {variant}")

    print("\n" + "=" * 88)
    print(f"TRECHOS SELECIONADOS: {len(report['selected'])}")
    print("=" * 88)
    for index, item in enumerate(report["selected"], start=1):
        print(f"\n--- SELECIONADO {index} ---")
        print(f"Fonte: {item['source']} | páginas: {item['pages']}")
        print(
            f"Similaridade: {item['sim']} | score ajustado: {item.get('score')} | "
            f"documento: {item.get('document_type')} | conteúdo: {item.get('content_type')}"
        )
        print("Motivos:", "; ".join(item.get("reasons") or []))
        print(item["text"][:1800])

    print("\n" + "=" * 88)
    print("CANDIDATOS E DESCARTES")
    print("=" * 88)
    for index, item in enumerate(report["candidates"], start=1):
        status = "SELECIONADO" if item["selected"] else "DESCARTADO"
        print(
            f"\n[{index}] {status} | {item['source']} | "
            f"sim={item['semantic_similarity']} | score={item['adjusted_score']}"
        )
        print(
            f"Tipo: {item['document_type']} / {item['content_type']} / {item['topic']}"
        )
        if item.get("discard_reason"):
            print("Descarte:", item["discard_reason"])
        print("Motivos:", "; ".join(item.get("reasons") or []))
        print("Prévia:", item.get("preview", "").replace("\n", " ")[:500])

    output = Path("debug_retrieval_report.json")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nRelatório completo: {output.resolve()}")


if __name__ == "__main__":
    main()
