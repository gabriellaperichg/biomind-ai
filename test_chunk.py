"""
Biomind — Etapa 1 do pipeline: ingestão + chunking dos PDFs.

O que ele faz:
  1. Lê todos os PDFs da pasta ./pdfs
  2. Extrai o texto página por página
  3. Quebra cada página em chunks, guardando metadados (arquivo + página)
  4. Salva tudo em chunks.jsonl
  5. Mostra uma prévia pra você OLHAR os chunks e ajustar o tamanho

Rodar:  python 01_chunk.py
"""

import os
import glob
import json
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- knobs que você ajusta olhando o resultado ----
PDF_DIR = "pdfs"
OUT_FILE = "chunks.jsonl"
# Medimos em CARACTERES (não tokens) de propósito: é 100% offline, sem baixar
# nada. Regra de bolso: ~1 token ≈ 4 caracteres. Então 1500 caracteres ≈ ~375
# tokens. Se quiser chunks maiores (~800 tokens), use ~3200. Ajuste olhando a prévia.
CHUNK_SIZE = 1500     # tamanho-alvo de cada chunk, em caracteres
CHUNK_OVERLAP = 200   # quanto cada chunk repete do anterior (não perder contexto na borda)

# RecursiveCharacterTextSplitter tenta cortar primeiro em parágrafo,
# depois linha, depois frase, depois palavra — assim o corte cai num
# ponto natural em vez de no meio de uma palavra.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def carregar_chunks():
    chunks = []
    paginas_sem_texto = []  # provavelmente escaneadas -> precisam de OCR

    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        print(f"Nenhum PDF encontrado em ./{PDF_DIR}/ — coloque os arquivos lá e rode de novo.")
        return chunks, paginas_sem_texto, 0

    for caminho in pdfs:
        nome = os.path.basename(caminho)
        doc = fitz.open(caminho)
        for n_pagina, pagina in enumerate(doc, start=1):
            texto = pagina.get_text("text").strip()
            if not texto:
                paginas_sem_texto.append((nome, n_pagina))
                continue
            for pedaco in splitter.split_text(texto):
                chunks.append({
                    "text": pedaco,
                    "source": nome,
                    "page": n_pagina,
                })
        doc.close()

    return chunks, paginas_sem_texto, len(pdfs)


def main():
    chunks, sem_texto, n_pdfs = carregar_chunks()
    if not chunks:
        return

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n{len(chunks)} chunks gerados de {n_pdfs} PDF(s) -> {OUT_FILE}")

    if sem_texto:
        print(f"\n  Atenção: {len(sem_texto)} página(s) sem texto extraível "
              f"(provavelmente escaneadas — vão precisar de OCR):")
        for nome, pag in sem_texto[:10]:
            print(f"   - {nome}, página {pag}")

    # Prévia: olhe estes chunks. Estão completos? Cortam no meio de uma ideia?
    print("\n--- prévia dos primeiros chunks (é isso que a busca vai enxergar) ---")
    for i, c in enumerate(chunks[:3]):
        preview = c["text"].replace("\n", " ")
        if len(preview) > 280:
            preview = preview[:280] + "…"
        print(f"\n[chunk {i}]  ({c['source']}, pág. {c['page']})")
        print(preview)

    # Estatística simples pra calibrar o tamanho
    tamanhos = [len(c["text"]) for c in chunks]
    print(f"\nTamanho dos chunks (caracteres): "
          f"menor={min(tamanhos)}, média={sum(tamanhos)//len(tamanhos)}, maior={max(tamanhos)}")


if __name__ == "__main__":
    main()