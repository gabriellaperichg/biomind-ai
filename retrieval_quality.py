"""
Heurísticas locais para melhorar a qualidade da recuperação RAG do Biomind.

Este módulo não realiza diagnóstico. Ele apenas classifica documentos/chunks,
identifica a intenção da consulta e reordena candidatos sem enviar dados para
serviços externos.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_STOPWORDS = {
    "a", "ao", "aos", "as", "com", "da", "das", "de", "do", "dos",
    "e", "em", "esta", "este", "foi", "há", "na", "nas", "no", "nos",
    "o", "os", "ou", "para", "por", "que", "se", "sem", "sob", "uma",
    "um", "uso", "usa", "usando", "paciente", "pacientes", "relata",
}

ADMINISTRATIVE_TERMS = (
    "termo de consentimento",
    "termo de informacao e consentimento",
    "consentimento informado",
    "declaro que fui informado",
    "declaro que fui informada",
    "autorizo a realizacao",
    "assinatura e carimbo",
    "assinatura do paciente",
    "assinatura da paciente",
)

CONSENT_QUERY_TERMS = (
    "consentimento",
    "termo de consentimento",
    "orientacao pre procedimento",
    "orientacao pos procedimento",
    "riscos do procedimento",
    "efeitos adversos do procedimento",
)

PROCEDURE_TERMS = (
    "mesoterapia",
    "microagulhamento",
    "intradermoterapia",
    "intralesional",
    "injetavel",
    "injecao",
    "aplicacao de corticoide",
    "laser",
    "led",
    "plasma rico em plaquetas",
    "prp",
)

TREATMENT_TERMS = (
    "tratamento",
    "conduta",
    "prescrever",
    "prescricao",
    "minoxidil",
    "finasterida",
    "dutasterida",
    "espironolactona",
    "suplementacao",
    "dose",
    "mg por dia",
    "posologia",
)

ASSESSMENT_TERMS = (
    "anamnese",
    "investigar",
    "avaliacao",
    "exame fisico",
    "tricoscopia",
    "dermatoscopia",
    "diagnostico diferencial",
    "historia clinica",
    "histórico clinico",
    "sinais e sintomas",
)

DIAGNOSIS_TERMS = (
    "diagnostico",
    "hipotese",
    "diferencial",
    "alopecia androgenetica",
    "efluvio telogeno",
    "dermatite seborreica",
    "alopecia areata",
    "alopecia cicatricial",
)

ADVERSE_EVENT_TERMS = (
    "efeito adverso",
    "evento adverso",
    "complicacao",
    "atrofia",
    "necrose",
    "edema",
    "eritema",
    "reacao alergica",
    "dor intensa",
    "agravamento apos",
    "nodulo",
    "nodulos",
    "caroco",
    "carocos",
    "secrecao",
    "pus",
    "abscesso",
    "granuloma",
    "infeccao",
    "calor local",
)

SCARRING_ALOPECIA_TERMS = (
    "alopecia cicatricial",
    "alopecia fibrosante frontal",
    "liquen plano pilar",
    "fapd",
    "lupus eritematoso discoide",
    "fibrose perifolicular",
    "perda de ostios",
    "ausencia de orificios foliculares",
)

SCALP_INFLAMMATION_TERMS = (
    "oleosidade",
    "oleoso",
    "descamacao",
    "escamacao",
    "escamas",
    "caspa",
    "prurido",
    "coceira",
    "ardor",
    "ardencia",
    "vermelhidao",
    "eritema",
    "inflamacao",
    "crostas",
)

HAIR_PATTERN_TERMS = (
    "coroa",
    "vertice",
    "topo",
    "temporal",
    "temporas",
    "frontal",
    "linha frontal",
    "rarefacao",
    "afinamento",
    "miniaturizacao",
    "padrao feminino",
    "padrao masculino",
)

MEDICATION_TERMS = (
    "anticoncepcional",
    "antidepressivo",
    "medicamento",
    "medicamentos",
    "farmaco",
    "remedio",
    "isotretinoina",
    "hormonio",
    "hormonal",
)

LAB_TERMS = (
    "ferritina",
    "vitamina b12",
    "vitamina d",
    "tsh",
    "hemograma",
    "exames laboratoriais",
    "deficiencia",
)

SOURCE_TYPE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("consent_form", ("consentimento", "termo", "autorizacao")),
    ("guideline", ("guideline", "diretriz", "consenso", "recomendacoes")),
    ("systematic_review", ("systematic review", "meta-analysis", "metaanalysis")),
    ("review", ("review", "revisao", "overview")),
    ("clinical_case", ("caso", "case report", "case series")),
    ("procedure_protocol", ("protocolo", "procedimento", "mesoterapia", "intralesional", "injection", "injections")),
)

SECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(conduta|tratamento|terapeutica|terapêutica)\s*:", "treatment"),
    (r"\b(procedimentos?|tecnicas?|técnicas?)\s*:", "procedure"),
    (r"\b(diagnostico|diagnóstico|hipoteses?|hipóteses?)\s*:", "diagnosis"),
    (r"\b(exames?|avaliacao|avaliação|anamnese|investigacao|investigação)\s*:", "assessment"),
    (r"\b(efeitos? colaterais?|eventos? adversos?|complicacoes?|complicações?)\s*:", "adverse_event"),
    (r"\b(observacoes?|observações?|conclusao|conclusão)\s*:", "discussion"),
)


@dataclass(frozen=True)
class QueryProfile:
    normalized: str
    tokens: frozenset[str]
    asks_procedure: bool
    asks_treatment: bool
    asks_consent: bool
    asks_diagnosis: bool
    has_scalp_inflammation: bool
    has_hair_pattern: bool
    has_medication: bool
    has_labs: bool
    has_adverse_event: bool


@dataclass(frozen=True)
class CandidateScore:
    adjusted_score: float
    lexical_coverage: float
    rejected: bool
    reasons: tuple[str, ...]
    document_type: str
    content_type: str
    topic: str
    source_quality: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _text_shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    """Transforma o texto em sequências de palavras para deduplicação."""
    words = normalize_text(text).split()
    if not words:
        return set()
    if len(words) < size:
        return {tuple(words)}
    return {
        tuple(words[index:index + size])
        for index in range(len(words) - size + 1)
    }


def content_similarity(text_a: str, text_b: str, size: int = 5) -> float:
    """
    Mede semelhança textual entre 0 e 1.

    Usa o maior valor entre Jaccard e contenção. A contenção é importante
    quando um chunk está quase inteiro dentro de outro chunk mais longo.
    """
    shingles_a = _text_shingles(text_a, size=size)
    shingles_b = _text_shingles(text_b, size=size)
    if not shingles_a or not shingles_b:
        return 0.0

    intersection = shingles_a & shingles_b
    if not intersection:
        return 0.0

    union = shingles_a | shingles_b
    jaccard = len(intersection) / len(union) if union else 0.0
    containment = len(intersection) / min(len(shingles_a), len(shingles_b))
    return max(jaccard, containment)


def contains_any(text: str, terms: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(term) in normalized for term in terms)


def clinical_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) >= 3 and token not in _STOPWORDS and not token.isdigit()
    }


def profile_query(query: str) -> QueryProfile:
    normalized = normalize_text(query)
    return QueryProfile(
        normalized=normalized,
        tokens=frozenset(clinical_tokens(query)),
        asks_procedure=contains_any(query, PROCEDURE_TERMS),
        asks_treatment=contains_any(
            query,
            (
                "tratamento", "tratar", "conduta", "o que fazer", "terapia",
                "medicamento", "suplemento", "procedimento",
            ),
        ),
        asks_consent=contains_any(query, CONSENT_QUERY_TERMS),
        asks_diagnosis=contains_any(
            query,
            ("diagnostico", "hipotese", "diferencial", "o que pode ser"),
        ),
        has_scalp_inflammation=contains_any(query, SCALP_INFLAMMATION_TERMS),
        has_hair_pattern=contains_any(query, HAIR_PATTERN_TERMS),
        has_medication=contains_any(query, MEDICATION_TERMS),
        has_labs=contains_any(query, LAB_TERMS),
        has_adverse_event=(
            contains_any(query, ADVERSE_EVENT_TERMS)
            or (
                contains_any(query, PROCEDURE_TERMS)
                and contains_any(
                    query,
                    (
                        "coceira", "prurido", "dor", "dolorido", "nodulo",
                        "caroco", "edema", "inchaco", "vermelhidao",
                        "secrecao", "pus", "calor local",
                    ),
                )
            )
        ),
    )


def infer_document_type(source: str, text: str = "") -> str:
    source_norm = normalize_text(Path(source).stem)
    text_norm = normalize_text(text[:2500])
    combined = f"{source_norm} {text_norm}"

    if contains_any(combined, ADMINISTRATIVE_TERMS):
        return "consent_form"

    for document_type, hints in SOURCE_TYPE_HINTS:
        if any(normalize_text(hint) in source_norm for hint in hints):
            return document_type

    if "doi" in text_norm or "abstract" in text_norm or "references" in text_norm:
        return "scientific_article"

    if re.search(r"\bcaso\s+\d+\b", text_norm) or "paciente do sexo" in text_norm:
        return "clinical_case"

    return "educational_material"


def infer_content_type(text: str, section: str | None = None) -> str:
    normalized = normalize_text(text)
    section_norm = normalize_text(section or "")

    for pattern, content_type in SECTION_PATTERNS:
        if re.search(pattern, str(text), flags=re.I):
            return content_type

    if section_norm in {"treatment", "procedure", "diagnosis", "assessment", "adverse event"}:
        return section_norm.replace(" ", "_")

    scores = {
        "procedure": sum(normalize_text(term) in normalized for term in PROCEDURE_TERMS),
        "treatment": sum(normalize_text(term) in normalized for term in TREATMENT_TERMS),
        "assessment": sum(normalize_text(term) in normalized for term in ASSESSMENT_TERMS),
        "diagnosis": sum(normalize_text(term) in normalized for term in DIAGNOSIS_TERMS),
        "adverse_event": sum(normalize_text(term) in normalized for term in ADVERSE_EVENT_TERMS),
    }

    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return "general"

    # Tratamento e procedimento aparecem juntos com frequência. Quando há
    # complicações, essa finalidade deve prevalecer.
    if scores["adverse_event"] >= 2:
        return "adverse_event"
    if scores["procedure"] >= 2 and scores["procedure"] >= scores["treatment"]:
        return "procedure"
    return best_type


def infer_topic(text: str) -> str:
    normalized = normalize_text(text)
    topic_terms = (
        # Temas específicos devem prevalecer sobre sinais genéricos como eritema.
        ("scarring_alopecia", SCARRING_ALOPECIA_TERMS),
        ("androgenetic_alopecia", ("alopecia androgenetica", "padrao feminino", "padrao masculino", "miniaturizacao")),
        ("telogen_effluvium", ("efluvio telogeno", "telogen effluvium")),
        ("alopecia_areata", ("alopecia areata",)),
        ("procedure_complication", ADVERSE_EVENT_TERMS),
        ("scalp_inflammation", SCALP_INFLAMMATION_TERMS),
        ("medication_history", MEDICATION_TERMS),
        ("nutrition_labs", LAB_TERMS),
    )
    for topic, terms in topic_terms:
        if any(normalize_text(term) in normalized for term in terms):
            return topic
    return "general_trichology"


def infer_source_quality(document_type: str) -> str:
    if document_type in {"guideline", "systematic_review", "review", "scientific_article"}:
        return "high"
    if document_type in {"clinical_case", "educational_material", "procedure_protocol"}:
        return "medium"
    return "low"


def extract_section_title(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    for pattern, content_type in SECTION_PATTERNS:
        match = re.search(pattern, compact, flags=re.I)
        if match and match.start() < 180:
            return content_type

    first = compact[:160]
    heading_match = re.match(r"^([^.!?]{3,80}):", first)
    if heading_match:
        return normalize_text(heading_match.group(1)).replace(" ", "_")[:60]
    return "general"


def enrich_chunk_metadata(source: str, text: str, section: str | None = None) -> dict[str, str]:
    document_type = infer_document_type(source, text)
    content_type = infer_content_type(text, section)
    topic = infer_topic(text)
    source_quality = infer_source_quality(document_type)
    resolved_section = section or extract_section_title(text)
    return {
        "document_type": document_type,
        "content_type": content_type,
        "topic": topic,
        "source_quality": source_quality,
        "section": resolved_section,
    }


def build_embedding_text(source: str, text: str, metadata: dict[str, Any]) -> str:
    labels = (
        f"tipo de documento: {metadata.get('document_type', 'desconhecido')}; "
        f"tipo de conteudo: {metadata.get('content_type', 'geral')}; "
        f"tema: {metadata.get('topic', 'tricologia')}; "
        f"secao: {metadata.get('section', 'geral')}"
    )
    return f"{labels}.\n{text}"


def build_query_variants(query: str, *, max_variants: int = 4) -> list[str]:
    profile = profile_query(query)
    variants = [query.strip()]

    focus_parts: list[str] = []
    if profile.has_scalp_inflammation:
        focus_parts.append(
            "oleosidade descamacao prurido inflamacao do couro cabeludo "
            "avaliacao e diagnostico diferencial"
        )
    if profile.has_hair_pattern:
        focus_parts.append(
            "padrao de rarefacao coroa vertice temporas afinamento "
            "tricoscopia e diagnostico diferencial"
        )
    if profile.has_medication:
        focus_parts.append(
            "anamnese medicamentosa nome dose tempo de uso alteracao recente "
            "e relacao temporal com a queda"
        )
    if profile.has_labs:
        focus_parts.append("exames laboratoriais e deficiencias conforme historia clinica")

    if focus_parts:
        variants.append("; ".join(focus_parts))

    if profile.asks_procedure and profile.has_adverse_event:
        variants.append(
            f"{query.strip()} evento adverso complicacao pos procedimento "
            "prurido coceira nodulos dolorosos reacao inflamatoria reacao alergica "
            "infeccao granuloma abscesso sinais de alerta encaminhamento"
        )

    variants.append(
        f"{query.strip()} investigacao inicial anamnese exame tricoscopia "
        "hipoteses a diferenciar sinais de alerta"
    )

    if profile.asks_treatment or profile.asks_procedure:
        variants.append(
            f"{query.strip()} criterios indicacoes contraindicacoes riscos e limites da fonte"
        )

    deduplicated: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = normalize_text(variant)
        if key and key not in seen:
            seen.add(key)
            deduplicated.append(variant)
        if len(deduplicated) >= max_variants:
            break
    return deduplicated


def _lexical_coverage(profile: QueryProfile, text: str) -> float:
    if not profile.tokens:
        return 0.0
    text_tokens = clinical_tokens(text)
    return len(profile.tokens & text_tokens) / len(profile.tokens)


def score_candidate(
    query: str,
    *,
    semantic_similarity: float,
    source: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> CandidateScore:
    metadata = metadata or {}
    profile = profile_query(query)

    document_type = str(
        metadata.get("document_type") or infer_document_type(source, text)
    )
    content_type = str(
        metadata.get("content_type") or infer_content_type(text, metadata.get("section"))
    )
    topic = str(metadata.get("topic") or infer_topic(text))
    source_quality = str(
        metadata.get("source_quality") or infer_source_quality(document_type)
    )

    reasons: list[str] = []
    rejected = False
    score = float(semantic_similarity)
    coverage = _lexical_coverage(profile, f"{source} {text}")
    score += min(0.12, coverage * 0.14)
    if coverage >= 0.45:
        reasons.append("boa cobertura dos termos do caso")

    if document_type == "consent_form" and not profile.asks_consent:
        rejected = True
        reasons.append("termo de consentimento fora da finalidade da pergunta")

    if source_quality == "high":
        score += 0.06
        reasons.append("fonte de maior hierarquia documental")
    elif source_quality == "low":
        score -= 0.10
        reasons.append("fonte administrativa ou de baixa utilidade clínica geral")

    if content_type == "assessment":
        score += 0.07 if not profile.asks_treatment else 0.03
        reasons.append("conteúdo de avaliação/investigação")
    elif content_type == "diagnosis":
        score += 0.05
        reasons.append("conteúdo de diferenciação diagnóstica")
    elif content_type == "treatment" and not profile.asks_treatment:
        score -= 0.13
        reasons.append("conteúdo de tratamento antes da confirmação")
    elif content_type == "procedure" and not profile.asks_procedure:
        score -= 0.22
        reasons.append("procedimento não solicitado")
    elif content_type == "adverse_event" and not profile.asks_procedure:
        score -= 0.18
        reasons.append("complicação de procedimento não relacionada à pergunta")

    if document_type == "clinical_case" and content_type == "general":
        score -= 0.05
        reasons.append("caso semelhante sem orientação de avaliação")

    if document_type == "procedure_protocol" and not profile.asks_procedure:
        score -= 0.12
        reasons.append("documento centrado em procedimento não solicitado")

    if document_type == "clinical_case" and content_type in {"treatment", "procedure"}:
        score -= 0.18
        reasons.append("risco de transferir conduta de outro caso")
        if not profile.asks_treatment and not profile.asks_procedure:
            rejected = True
            reasons.append("conduta de caso clínico não serve como investigação inicial")

    normalized_text = normalize_text(text)
    if profile.has_scalp_inflammation and contains_any(normalized_text, SCALP_INFLAMMATION_TERMS):
        score += 0.08
        reasons.append("cobre sinais inflamatórios/descamação")
    if profile.has_hair_pattern and contains_any(normalized_text, HAIR_PATTERN_TERMS):
        score += 0.06
        reasons.append("cobre o padrão/localização da queda")
    if profile.has_medication and contains_any(normalized_text, MEDICATION_TERMS):
        score += 0.05
        reasons.append("cobre histórico medicamentoso")
    if profile.has_labs and contains_any(normalized_text, LAB_TERMS):
        score += 0.04
        reasons.append("cobre exames/laboratório")

    # Um documento de procedimento pode ser útil quando a pergunta realmente
    # trata desse procedimento, mas não deve dominar uma consulta clínica geral.
    if profile.asks_procedure and content_type in {"procedure", "adverse_event"}:
        score += 0.10
        reasons.append("procedimento solicitado pela consulta")

    if profile.asks_procedure and profile.has_adverse_event:
        if content_type == "adverse_event" or contains_any(text, ADVERSE_EVENT_TERMS):
            score += 0.16
            reasons.append("cobre possível evento adverso pós-procedimento")
        elif content_type == "procedure":
            score -= 0.04
            reasons.append("procedimento genérico sem foco em complicação")

        source_norm = normalize_text(source)
        if contains_any(source_norm, ("indicacoes", "beneficios", "intervalo entre sessoes")):
            score -= 0.08
            reasons.append("fonte de indicação geral pouco útil para evento adverso")

    return CandidateScore(
        adjusted_score=round(score, 6),
        lexical_coverage=round(coverage, 6),
        rejected=rejected,
        reasons=tuple(reasons),
        document_type=document_type,
        content_type=content_type,
        topic=topic,
        source_quality=source_quality,
    )
