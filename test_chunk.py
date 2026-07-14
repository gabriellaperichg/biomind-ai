"""
Biomind — Etapa 1: ingestão + chunking dos PDFs (v3).

Novidades da v3:
  - Remove automaticamente linhas que se repetem em várias páginas/documentos
    (cabeçalhos, rodapés, avisos de copyright) — sem precisar cadastrar cada uma.
  - Ainda dá pra cadastrar trechos específicos em BOILERPLATE_PATTERNS.
  - Continua empacotando por frases inteiras (não corta no meio).

Rodar:  python 01_chunk.py
"""

import os
import re
import glob
import json
import fitz  # PyMuPDF
from collections import Counter

PDF_DIR = "pdfs"
OUT_FILE = "chunks.jsonl"
MAX_CHARS = 1800        # tamanho-alvo do chunk (caracteres)
OVERLAP_CHARS = 250     # continuidade entre chunks
REPETICAO_MIN = 3       # linha que aparece em >= N páginas do corpus = cabeçalho/rodapé
MIN_LEN_LIXO = 15       # ignora linhas curtíssimas na detecção (números de página etc.)

# ---------------------------------------------------------------------------
# PULAR PÁGINAS INICIAIS (capa, ficha catalográfica, dedicatória, prefácio, sumário).
# Livros têm isso; artigos normalmente não. A chave é um pedaço do nome do arquivo
# (basta ser único), o valor é até que página pular.
#
# Exemplo: "3BocciOzone": 20   ->  pula as páginas 1 a 20 desse PDF.
# Descubra o número certo abrindo o PDF e vendo onde começa o Capítulo 1.
# ---------------------------------------------------------------------------
PULAR_PAGINAS_INICIAIS = {
    "Interventionsforfemalepatternhairloss-Review": 2,
    "TABELATACO-TabelaBrasileiradeComposiodeAlimentos": 10,
}

# Trechos específicos pra remover (regex). A detecção automática pega cabeçalhos e
# rodapés repetidos; use isto para blocos que aparecem UMA vez só (copyright de editora).
BOILERPLATE_PATTERNS = [
    r"Esse material é rastreável.*?Código Penal Brasileiro\.",
    r"Copyright.{0,4}Lippincott Williams & Wilkins.*?prohibited\.?",
    # avisos de "todos os direitos reservados" / "all rights reserved" (PT e EN)
    r"(Todos os direitos reservados|All rights reserved).{0,800}?(sem autorização escrita.*?\.|without written permission.*?\.)",
    r"Nenhuma parte deste (material|livro).{0,600}?(do Editor|Publisher)[^.]*\.",
    r"No part of this (material|book|work).{0,600}?(the Publisher|permission)[^.]*\.",
]


def pular_ate(nome_arquivo: str) -> int:
    """Retorna até qual página pular neste PDF (0 = não pula nada)."""
    for chave, ate in PULAR_PAGINAS_INICIAIS.items():
        if chave.lower() in nome_arquivo.lower():
            return ate
    return 0


def remover_boilerplate(texto: str) -> str:
    for pat in BOILERPLATE_PATTERNS:
        texto = re.sub(pat, " ", texto, flags=re.S | re.I)
    return texto


def _linhas_norm(texto: str):
    return [l.strip() for l in texto.split("\n") if l.strip()]


def detectar_linhas_repetidas(paginas_por_pdf) -> set:
    freq = Counter()
    for paginas in paginas_por_pdf.values():
        for txt in paginas:
            for l in set(_linhas_norm(txt)):   # conta 1x por página
                freq[l] += 1
    return {l for l, c in freq.items() if c >= REPETICAO_MIN and len(l) >= MIN_LEN_LIXO}


def remover_linhas(texto: str, lixo: set) -> str:
    if not lixo:
        return texto
    return "\n".join(l for l in texto.split("\n") if l.strip() not in lixo)


def limpar_texto(texto: str) -> str:
    texto = remover_boilerplate(texto)
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)        # junta hífen quebrado (fim de linha)
    texto = re.sub(r"\n[ \t]*\n", "\uE000", texto)        # protege parágrafos
    texto = texto.replace("\n", " ")                      # quebra de largura -> espaço
    texto = texto.replace("\uE000", "\n\n")               # restaura parágrafos

    # marcadores de página de revista grudados no texto: "| 967", "968 |", "| 969 |"
    texto = re.sub(r"\|\s*\d{1,4}\s*\|", " ", texto)      # | 969 |
    texto = re.sub(r"(?:^|\s)\|\s*\d{1,4}(?=\s)", " ", texto)   # | 967
    texto = re.sub(r"(?:^|\s)\d{1,4}\s*\|(?=\s)", " ", texto)   # 968 |

    # hífen de quebra que sobrou no meio do texto: "side ef- fects" -> "side effects"
    # (só junta quando a 2ª parte é minúscula, pra não colar palavras compostas legítimas)
    texto = re.sub(r"(\w)-\s+([a-záéíóúâêôãõç])", r"\1\2", texto)

    texto = re.sub(r"[ \t]{2,}", " ", texto)
    return texto.strip()


def em_frases(paragrafo: str):
    partes = re.split(r"(?<=[.!?])\s+", paragrafo.strip())
    return [p.strip() for p in partes if p.strip()]


def empacotar(frases, max_chars, overlap_chars):
    chunks, atual, tam = [], [], 0
    for f in frases:
        if tam + len(f) + 1 > max_chars and atual:
            chunks.append(" ".join(atual))
            back, b = [], 0
            for s in reversed(atual):
                if b + len(s) > overlap_chars:
                    break
                back.insert(0, s)
                b += len(s) + 1
            atual, tam = list(back), sum(len(x) + 1 for x in back)
        atual.append(f)
        tam += len(f) + 1
    if atual:
        chunks.append(" ".join(atual))
    return chunks


def chunks_da_pagina(bruto: str, lixo=frozenset()):
    texto = remover_linhas(bruto, lixo)
    texto = limpar_texto(texto)
    frases = []
    for par in texto.split("\n\n"):
        frases.extend(em_frases(par))
    return empacotar(frases, MAX_CHARS, OVERLAP_CHARS)


def main():
    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        print(f"Nenhum PDF em ./{PDF_DIR}/ — coloque os arquivos lá e rode de novo.")
        return

    # Passo 1: extrai todas as páginas e detecta cabeçalhos/rodapés repetidos
    paginas_por_pdf = {}
    for caminho in pdfs:
        doc = fitz.open(caminho)
        paginas_por_pdf[caminho] = [pg.get_text("text") for pg in doc]
        doc.close()
    lixo = detectar_linhas_repetidas(paginas_por_pdf)

    # Passo 2: monta os chunks já sem o lixo repetido
    chunks, sem_texto, puladas = [], [], 0
    for caminho in pdfs:
        nome = os.path.basename(caminho)
        ate = pular_ate(nome)
        for n_pagina, bruto in enumerate(paginas_por_pdf[caminho], start=1):
            if n_pagina <= ate:            # páginas iniciais do livro (capa, prefácio…)
                puladas += 1
                continue
            if not bruto.strip():
                sem_texto.append((nome, n_pagina))
                continue
            for pedaco in chunks_da_pagina(bruto, lixo):
                chunks.append({"text": pedaco, "source": nome, "page": n_pagina})

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n{len(chunks)} chunks gerados de {len(pdfs)} PDF(s) -> {OUT_FILE}")

    if puladas:
        print(f"{puladas} página(s) inicial(is) puladas (capa/prefácio/sumário), "
              f"conforme PULAR_PAGINAS_INICIAIS.")

    if lixo:
        print(f"\n{len(lixo)} linha(s) repetidas removidas automaticamente (cabeçalho/rodapé). Exemplos:")
        for l in list(lixo)[:5]:
            print("   -", (l[:80] + "…") if len(l) > 80 else l)

    if sem_texto:
        print(f"\n  Atenção: {len(sem_texto)} página(s) sem texto (provável scan -> OCR).")

    if chunks:
        print("\n--- prévia ---")
        for i, c in enumerate(chunks[:2]):
            prev = c["text"]
            print(f"\n[chunk {i}] ({c['source']}, pág. {c['page']})\n{prev[:300]}"
                  + ("…" if len(prev) > 300 else ""))


if __name__ == "__main__":
    main()