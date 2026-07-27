from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env", override=True)

import biomind_core as core


QUESTIONS = [
    "Paciente feminina, queda na coroa, couro cabeludo oleoso e escamando, toma anticoncepcional e antidepressivo.",
    "Paciente mulher em tratamento capilar com mesoterapia há 5 meses e, pela primeira vez, relatou muita coceira e caroços doloridos. Como prosseguir?",
]


def main() -> None:
    output_csv = Path("benchmark_biomind.csv")
    output_json = Path("benchmark_biomind.json")

    rows: list[dict[str, object]] = []

    print("Configuração carregada:")
    print(f"Modelo Ollama: {core.OLLAMA_MODEL}")
    print(f"Modelo de embeddings: {core.EMBED_MODEL}")
    print(f"TOP_K: {core.TOP_K}")
    print(f"Contexto máximo RAG: {core.MAX_CONTEXT_CHARS}")
    print(f"Num ctx Ollama: {core.OLLAMA_NUM_CTX}")
    print(f"Num predict: {core.OLLAMA_NUM_PREDICT}")
    print()

    for index, question in enumerate(QUESTIONS, start=1):
        print("=" * 80)
        print(f"Teste {index}/{len(QUESTIONS)}")
        print(question)
        print("=" * 80)

        started = time.perf_counter()
        result = core.responder(question)
        elapsed = time.perf_counter() - started

        answer = result.get("answer") or result.get("message") or ""
        sources = result.get("sources") or []

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question_number": index,
            "question": question,
            "status": result.get("status"),
            "elapsed_seconds": round(elapsed, 2),
            "answer_characters": len(str(answer)),
            "source_count": len(sources),
            "best_similarity": result.get("best_sim"),
            "model": result.get("model_name", core.OLLAMA_MODEL),
            "embedding_model": result.get("embedding_model", core.EMBED_MODEL),
            "top_k": core.TOP_K,
            "max_context_chars": core.MAX_CONTEXT_CHARS,
            "ollama_num_ctx": core.OLLAMA_NUM_CTX,
            "ollama_num_predict": core.OLLAMA_NUM_PREDICT,
        }
        rows.append(row)

        print(f"Status: {row['status']}")
        print(f"Tempo: {row['elapsed_seconds']} s")
        print(f"Fontes: {row['source_count']}")
        print(f"Caracteres da resposta: {row['answer_characters']}")
        print()

    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Arquivos gerados:")
    print(output_csv.resolve())
    print(output_json.resolve())


if __name__ == "__main__":
    main()
