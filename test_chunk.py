"""
Biomind — Etapa 1: ingestão + chunking dos PDFs (v4).

Melhorias principais:
- Detecta cabeçalhos/rodapés repetidos por PDF e apenas nas bordas da página.
- Remove boilerplates cadastrados.
- Reconstrói o documento antes de separar frases, evitando cortes entre páginas.
- Mantém page_start e page_end.
- Faz overlap por unidades inteiras.
- Gera hash/ID estável e uma auditoria ao final.

Rodar:
    python 01_chunk.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


PDF_DIR = "pdfs"
OUT_FILE = "chunks.jsonl"

MAX_CHARS = 1800
OVERLAP_CHARS = 250
MAX_UNIT_CHARS = 3000

REPETICAO_MIN = 3
MIN_LEN_LIXO = 15
MAX_LEN_LIXO = 220
LINHAS_BORDA = 4

PULAR_PAGINAS_INICIAIS = {
    "Interventionsforfemalepatternhairloss-Review": 2,
    "TABELATACO-TabelaBrasileiradeComposiodeAlimentos": 10,
    "3BocciOzoneANewMedicalDrugLivroCompleto_Traduzido.en.pt-9": 13,
    "LIVROGuia_uso_medico_OzonoPortugues": 12,
    "madri2020": 15,
    "EBOOKSUPLEMENTACAOCAPILAR": 5,
}

BOILERPLATE_PATTERNS = [
    r"Esse material é rastreável.*?Código Penal Brasileiro\.",
    r"Copyright.{0,4}Lippincott Williams & Wilkins.*?prohibited\.?",
    r"(Todos os direitos reservados|All rights reserved).{0,800}?"
    r"(sem autorização escrita.*?\.|without written permission.*?\.)",
    r"Nenhuma parte deste (material|livro).{0,600}?"
    r"(do Editor|Publisher)[^.]*\.",
    r"No part of this (material|book|work).{0,600}?"
    r"(the Publisher|permission)[^.]*\.",
]

ABREVIACOES = (
    "Dr.", "Dra.", "Prof.", "Profa.", "Sr.", "Sra.",
    "Fig.", "Figs.", "Eq.", "Eqs.", "p.", "pp.",
    "vol.", "n.", "no.", "nº.",
)

DOT_PROTEGIDO = "\uE001"


@dataclass(frozen=True)
class PageSpan:
    page: int
    start: int
    end: int


@dataclass
class Unit:
    text: str
    start: int
    end: int
    page_start: int
    page_end: int
    forced_split: bool = False


def pular_ate(nome_arquivo: str) -> int:
    nome = nome_arquivo.lower()
    for chave, ate in PULAR_PAGINAS_INICIAIS.items():
        if chave.lower() in nome:
            return ate
    return 0


def hash_texto(texto: str) -> str:
    normalizado = re.sub(r"\s+", " ", texto).strip().lower()
    return hashlib.sha256(normalizado.encode("utf-8")).hexdigest()


def linhas_nao_vazias(texto: str) -> list[str]:
    return [linha.strip() for linha in texto.splitlines() if linha.strip()]


def normalizar_linha_repetida(linha: str) -> str:
    linha = re.sub(r"\s+", " ", linha.strip().lower())
    return re.sub(r"\b\d{1,4}\b", "<n>", linha)


def detectar_linhas_repetidas(
    paginas_por_pdf: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Detecta assinaturas de cabeçalho/rodapé separadamente por PDF."""
    lixo_por_pdf: dict[str, set[str]] = {}

    for caminho, paginas in paginas_por_pdf.items():
        ate = pular_ate(os.path.basename(caminho))
        frequencia: Counter[str] = Counter()

        for numero, texto in enumerate(paginas, start=1):
            if numero <= ate:
                continue

            linhas = linhas_nao_vazias(texto)
            candidatas = linhas[:LINHAS_BORDA] + linhas[-LINHAS_BORDA:]

            for linha in set(candidatas):
                if MIN_LEN_LIXO <= len(linha) <= MAX_LEN_LIXO:
                    frequencia[normalizar_linha_repetida(linha)] += 1

        lixo_por_pdf[caminho] = {
            assinatura
            for assinatura, quantidade in frequencia.items()
            if quantidade >= REPETICAO_MIN
        }

    return lixo_por_pdf


def remover_linhas_repetidas(texto: str, lixo: set[str]) -> str:
    if not lixo:
        return texto

    mantidas = []
    for linha in texto.splitlines():
        limpa = linha.strip()
        assinatura = normalizar_linha_repetida(limpa) if limpa else ""

        if (
            limpa
            and MIN_LEN_LIXO <= len(limpa) <= MAX_LEN_LIXO
            and assinatura in lixo
        ):
            continue

        mantidas.append(linha)

    return "\n".join(mantidas)


def remover_boilerplate(texto: str) -> str:
    for padrao in BOILERPLATE_PATTERNS:
        texto = re.sub(padrao, " ", texto, flags=re.S | re.I)
    return texto


def limpar_texto(texto: str) -> str:
    texto = remover_boilerplate(texto)
    texto = texto.replace("\u00ad", "")  # soft hyphen

    # Linha contendo somente o número da página.
    texto = re.sub(
        r"(?im)^\s*(?:p[aá]gina\s*)?\d{1,4}\s*$",
        " ",
        texto,
    )

    # Palavra quebrada por hífen no fim de uma linha da mesma página.
    texto = re.sub(
        r"([A-Za-zÀ-ÖØ-öø-ÿ])-\s*\n\s*([A-Za-zÀ-ÖØ-öø-ÿ])",
        r"\1\2",
        texto,
    )

    # Preserva parágrafos; remove quebras causadas pela largura da página.
    texto = re.sub(r"\n[ \t]*\n+", "\uE000", texto)
    texto = texto.replace("\n", " ")
    texto = texto.replace("\uE000", "\n\n")

    # Marcadores de página de revista: | 969 |, | 967, 968 |
    texto = re.sub(r"\|\s*\d{1,4}\s*\|", " ", texto)
    texto = re.sub(r"(?:^|\s)\|\s*\d{1,4}(?=\s)", " ", texto)
    texto = re.sub(r"(?:^|\s)\d{1,4}\s*\|(?=\s)", " ", texto)

    texto = re.sub(r"[ \t]{2,}", " ", texto)
    texto = re.sub(r" *\n\n *", "\n\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def corrigir_hifen_entre_paginas(
    paginas: list[dict[str, object]],
) -> list[dict[str, object]]:
    paginas = [dict(pagina) for pagina in paginas]

    for i in range(len(paginas) - 1):
        atual = str(paginas[i]["text"])
        seguinte = str(paginas[i + 1]["text"])

        if (
            re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]-$", atual)
            and re.match(r"^[a-zà-öø-ÿ]", seguinte)
        ):
            paginas[i]["text"] = atual[:-1]
            paginas[i + 1]["sem_espaco_antes"] = True

    return paginas


def juntar_paginas(
    paginas: list[dict[str, object]],
) -> tuple[str, list[PageSpan]]:
    partes: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0

    for i, pagina in enumerate(paginas):
        if i > 0:
            separador = "" if pagina.get("sem_espaco_antes") else " "
            partes.append(separador)
            cursor += len(separador)

        texto = str(pagina["text"])
        inicio = cursor
        partes.append(texto)
        cursor += len(texto)

        spans.append(
            PageSpan(
                page=int(pagina["page"]),
                start=inicio,
                end=cursor,
            )
        )

    return "".join(partes), spans


def proteger_abreviacoes(texto: str) -> str:
    protegido = texto

    for abreviacao in sorted(ABREVIACOES, key=len, reverse=True):
        padrao = re.compile(r"(?<!\w)" + re.escape(abreviacao), re.I)
        protegido = padrao.sub(
            abreviacao.replace(".", DOT_PROTEGIDO),
            protegido,
        )

    protegido = re.sub(r"(?<=\d)\.(?=\d)", DOT_PROTEGIDO, protegido)
    protegido = re.sub(
        r"\bet al\.(?=\s*[,;(]|\s*\d{4})",
        f"et al{DOT_PROTEGIDO}",
        protegido,
        flags=re.I,
    )
    protegido = re.sub(
        r"\b([A-Z])\.(?=\s*[A-ZÀ-ÖØ-Þ])",
        rf"\1{DOT_PROTEGIDO}",
        protegido,
    )
    return protegido


def paginas_do_intervalo(
    inicio: int,
    fim: int,
    spans: list[PageSpan],
) -> tuple[int, int]:
    paginas = [
        span.page
        for span in spans
        if inicio < span.end and fim > span.start
    ]

    if paginas:
        return min(paginas), max(paginas)

    mais_proxima = min(
        spans,
        key=lambda span: min(
            abs(inicio - span.start),
            abs(inicio - span.end),
        ),
    )
    return mais_proxima.page, mais_proxima.page


def extrair_unidades(texto: str, spans: list[PageSpan]) -> list[Unit]:
    """Separa sentenças/parágrafos sem considerar a página como fronteira."""
    protegido = proteger_abreviacoes(texto)
    fronteira = re.compile(r'([.!?]["”’\)\]]*)\s+|\n{2,}')

    unidades: list[Unit] = []
    inicio = 0

    def adicionar(fim_bruto: int) -> None:
        trecho_bruto = texto[inicio:fim_bruto]
        esquerda = len(trecho_bruto) - len(trecho_bruto.lstrip())
        direita = len(trecho_bruto) - len(trecho_bruto.rstrip())

        inicio_real = inicio + esquerda
        fim_real = fim_bruto - direita

        if inicio_real >= fim_real:
            return

        page_start, page_end = paginas_do_intervalo(
            inicio_real,
            fim_real,
            spans,
        )
        unidades.append(
            Unit(
                text=texto[inicio_real:fim_real],
                start=inicio_real,
                end=fim_real,
                page_start=page_start,
                page_end=page_end,
            )
        )

    for match in fronteira.finditer(protegido):
        fim = (
            match.start(1) + len(match.group(1))
            if match.group(1)
            else match.start()
        )
        adicionar(fim)
        inicio = match.end()

    adicionar(len(texto))
    return unidades


def fragmentar_unidades_longas(
    unidades: list[Unit],
    spans: list[PageSpan],
) -> list[Unit]:
    resultado: list[Unit] = []

    for unidade in unidades:
        if len(unidade.text) <= MAX_UNIT_CHARS:
            resultado.append(unidade)
            continue

        pos = 0
        while pos < len(unidade.text):
            fim_maximo = min(pos + MAX_UNIT_CHARS, len(unidade.text))
            corte = fim_maximo

            if fim_maximo < len(unidade.text):
                minimo = pos + int(MAX_UNIT_CHARS * 0.55)
                for separador in ("; ", ": ", ", ", " "):
                    achado = unidade.text.rfind(separador, pos, fim_maximo)
                    if achado >= minimo:
                        corte = achado if separador == " " else achado + 1
                        break

            bruto = unidade.text[pos:corte]
            esquerda = len(bruto) - len(bruto.lstrip())
            direita = len(bruto) - len(bruto.rstrip())
            inicio_rel = pos + esquerda
            fim_rel = corte - direita

            if inicio_rel < fim_rel:
                inicio_abs = unidade.start + inicio_rel
                fim_abs = unidade.start + fim_rel
                page_start, page_end = paginas_do_intervalo(
                    inicio_abs,
                    fim_abs,
                    spans,
                )
                resultado.append(
                    Unit(
                        text=unidade.text[inicio_rel:fim_rel],
                        start=inicio_abs,
                        end=fim_abs,
                        page_start=page_start,
                        page_end=page_end,
                        forced_split=True,
                    )
                )

            pos = max(corte, pos + 1)
            while pos < len(unidade.text) and unidade.text[pos].isspace():
                pos += 1

    return resultado


def tamanho_unidades(unidades: list[Unit]) -> int:
    if not unidades:
        return 0
    return sum(len(u.text) for u in unidades) + len(unidades) - 1


def unidades_overlap(unidades: list[Unit]) -> list[Unit]:
    selecionadas: list[Unit] = []
    tamanho = 0

    for unidade in reversed(unidades):
        novo = tamanho + len(unidade.text) + (1 if selecionadas else 0)

        # Não duplica uma sentença enorme somente para cumprir overlap.
        if (
            not selecionadas
            and len(unidade.text) > OVERLAP_CHARS
            and len(unidade.text) > MAX_CHARS // 2
        ):
            break

        if selecionadas and novo > OVERLAP_CHARS:
            break

        selecionadas.insert(0, unidade)
        tamanho = novo

    return selecionadas


def montar_chunk(
    unidades: list[Unit],
    source: str,
    document_id: str,
    indice: int,
) -> dict[str, object]:
    texto = " ".join(u.text for u in unidades)
    content_hash = hash_texto(texto)
    page_start = min(u.page_start for u in unidades)
    page_end = max(u.page_end for u in unidades)

    return {
        "chunk_id": f"{document_id}_{indice:05d}_{content_hash[:10]}",
        "document_id": document_id,
        "text": texto,
        "source": source,
        "page": page_start,  # compatibilidade com seu formato anterior
        "page_start": page_start,
        "page_end": page_end,
        "char_count": len(texto),
        "content_hash": content_hash,
        "forced_split": any(u.forced_split for u in unidades),
    }


def empacotar_unidades(
    unidades: list[Unit],
    source: str,
) -> list[dict[str, object]]:
    document_id = re.sub(
        r"[^a-z0-9]+",
        "_",
        Path(source).stem.lower(),
    ).strip("_")

    chunks: list[dict[str, object]] = []
    atual: list[Unit] = []

    for unidade in unidades:
        if atual and tamanho_unidades(atual + [unidade]) > MAX_CHARS:
            chunks.append(
                montar_chunk(atual, source, document_id, len(chunks))
            )
            atual = unidades_overlap(atual)

            if atual and tamanho_unidades(atual + [unidade]) > MAX_CHARS:
                atual = []

        atual.append(unidade)

    if atual:
        chunks.append(montar_chunk(atual, source, document_id, len(chunks)))

    return chunks


def extrair_paginas_pdf(caminho: str) -> list[str]:
    try:
        with fitz.open(caminho) as documento:
            if documento.needs_pass:
                raise RuntimeError("PDF protegido por senha")
            return [pagina.get_text("text") for pagina in documento]
    except Exception as exc:
        raise RuntimeError(f"Não foi possível ler '{caminho}': {exc}") from exc


def preparar_paginas(
    caminho: str,
    paginas_brutas: list[str],
    lixo: set[str],
) -> tuple[list[dict[str, object]], list[tuple[str, int]], int]:
    nome = os.path.basename(caminho)
    ate = pular_ate(nome)
    paginas_limpas: list[dict[str, object]] = []
    sem_texto: list[tuple[str, int]] = []
    puladas = 0

    for numero, bruto in enumerate(paginas_brutas, start=1):
        if numero <= ate:
            puladas += 1
            continue

        if not bruto.strip():
            sem_texto.append((nome, numero))
            continue

        texto = remover_linhas_repetidas(bruto, lixo)
        texto = limpar_texto(texto)

        if not texto:
            sem_texto.append((nome, numero))
            continue

        paginas_limpas.append({"page": numero, "text": texto})

    return corrigir_hifen_entre_paginas(paginas_limpas), sem_texto, puladas


def auditar_chunks(chunks: list[dict[str, object]]) -> None:
    if not chunks:
        return

    tamanhos = sorted(int(c["char_count"]) for c in chunks)
    atravessam = sum(c["page_start"] != c["page_end"] for c in chunks)
    forçados = sum(bool(c["forced_split"]) for c in chunks)

    print("\n--- auditoria ---")
    print(f"Mínimo: {tamanhos[0]} caracteres")
    print(f"Mediana: {tamanhos[len(tamanhos) // 2]} caracteres")
    print(f"Máximo: {tamanhos[-1]} caracteres")
    print(f"Acima de MAX_CHARS: {sum(t > MAX_CHARS for t in tamanhos)}")
    print(f"Muito curtos (<250): {sum(t < 250 for t in tamanhos)}")
    print(f"Atravessam páginas: {atravessam}")
    print(f"Com divisão forçada: {forçados}")


def main() -> None:
    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        print(f"Nenhum PDF em ./{PDF_DIR}/")
        return

    paginas_por_pdf: dict[str, list[str]] = {}
    erros: list[str] = []

    for caminho in pdfs:
        try:
            paginas_por_pdf[caminho] = extrair_paginas_pdf(caminho)
        except RuntimeError as exc:
            erros.append(str(exc))

    if not paginas_por_pdf:
        print("Nenhum PDF pôde ser processado.")
        return

    lixo_por_pdf = detectar_linhas_repetidas(paginas_por_pdf)

    todos_chunks: list[dict[str, object]] = []
    sem_texto_total: list[tuple[str, int]] = []
    total_puladas = 0
    hashes_vistos: set[tuple[str, str]] = set()

    for caminho, paginas_brutas in paginas_por_pdf.items():
        nome = os.path.basename(caminho)
        paginas, sem_texto, puladas = preparar_paginas(
            caminho,
            paginas_brutas,
            lixo_por_pdf.get(caminho, set()),
        )

        sem_texto_total.extend(sem_texto)
        total_puladas += puladas

        if not paginas:
            continue

        texto_documento, spans = juntar_paginas(paginas)
        unidades = extrair_unidades(texto_documento, spans)
        unidades = fragmentar_unidades_longas(unidades, spans)
        chunks_documento = empacotar_unidades(unidades, nome)

        # Remove apenas duplicatas exatas dentro do mesmo documento.
        for chunk in chunks_documento:
            chave = (str(chunk["document_id"]), str(chunk["content_hash"]))
            if chave not in hashes_vistos:
                hashes_vistos.add(chave)
                todos_chunks.append(chunk)

    with open(OUT_FILE, "w", encoding="utf-8") as arquivo:
        for chunk in todos_chunks:
            arquivo.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(
        f"\n{len(todos_chunks)} chunks gerados de "
        f"{len(paginas_por_pdf)} PDF(s) -> {OUT_FILE}"
    )

    if total_puladas:
        print(f"{total_puladas} página(s) inicial(is) puladas.")

    total_lixo = sum(len(x) for x in lixo_por_pdf.values())
    if total_lixo:
        print(f"{total_lixo} cabeçalho(s)/rodapé(s) repetido(s) detectado(s).")

    if sem_texto_total:
        print(
            f"Atenção: {len(sem_texto_total)} página(s) sem texto "
            "(provável scan; podem exigir OCR)."
        )

    if erros:
        print(f"\n{len(erros)} PDF(s) não processado(s):")
        for erro in erros:
            print(" -", erro)

    auditar_chunks(todos_chunks)

    if todos_chunks:
        print("\n--- prévia ---")
        for i, chunk in enumerate(todos_chunks[:2]):
            paginas = (
                str(chunk["page_start"])
                if chunk["page_start"] == chunk["page_end"]
                else f"{chunk['page_start']}-{chunk['page_end']}"
            )
            previa = str(chunk["text"])
            print(
                f"\n[chunk {i}] ({chunk['source']}, pág. {paginas})\n"
                f"{previa[:400]}"
                + ("…" if len(previa) > 400 else "")
            )


if __name__ == "__main__":
    main()