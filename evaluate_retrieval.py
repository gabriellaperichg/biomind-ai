"""Executa casos de regressão da recuperação sem chamar o Ollama."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import biomind_core as core
from retrieval_quality import normalize_text


def main() -> None:
    cases = json.loads(Path("evaluation_cases.json").read_text(encoding="utf-8"))
    failures = 0

    for case in cases:
        report = core.debug_recuperacao(case["question"])
        selected = report["selected"]
        combined_text = normalize_text(
            " ".join(f"{item['source']} {item['text']}" for item in selected)
        )
        sources = normalize_text(" ".join(item["source"] for item in selected))
        content_types = {str(item.get("content_type")) for item in selected}

        has_expected = any(
            normalize_text(term) in combined_text
            for term in case.get("expected_any_terms", [])
        )
        forbidden_source = [
            term
            for term in case.get("forbidden_source_terms", [])
            if normalize_text(term) in sources
        ]
        forbidden_types = sorted(
            content_types & set(case.get("forbidden_content_types", []))
        )

        passed = bool(selected) and has_expected and not forbidden_source and not forbidden_types
        if not passed:
            failures += 1

        print("\n" + ("PASSOU" if passed else "FALHOU"), "-", case["name"])
        print("  selecionados:", len(selected))
        print("  fontes:", ", ".join(item["source"] for item in selected) or "nenhuma")
        if forbidden_source:
            print("  fontes proibidas:", forbidden_source)
        if forbidden_types:
            print("  tipos proibidos:", forbidden_types)
        if not has_expected:
            print("  nenhum termo esperado foi encontrado")

    print(f"\nResultado: {len(cases) - failures}/{len(cases)} casos passaram.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
