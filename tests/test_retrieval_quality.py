from retrieval_quality import (
    build_query_variants,
    enrich_chunk_metadata,
    score_candidate,
)


QUESTION = (
    "Paciente feminina, queda na coroa, couro cabeludo oleoso e escamando, "
    "toma anticoncepcional e antidepressivo."
)


def test_consent_form_is_rejected_for_general_clinical_question():
    result = score_candidate(
        QUESTION,
        semantic_similarity=0.70,
        source="Termo-de-consentimento-corticoide.pdf",
        text="Declaro que fui informada e autorizo a realização do procedimento.",
    )
    assert result.rejected is True
    assert result.document_type == "consent_form"


def test_treatment_chunk_from_other_clinical_case_is_rejected():
    text = (
        "Caso 3. Paciente do sexo feminino. Alopecia androgenética confirmada. "
        "Conduta: prescrever minoxidil e suplementação."
    )
    result = score_candidate(
        QUESTION,
        semantic_similarity=0.62,
        source="AAG-Caso3.pdf",
        text=text,
    )
    assert result.document_type == "clinical_case"
    assert result.content_type == "treatment"
    assert result.rejected is True


def test_assessment_content_outscores_unrequested_procedure_content():
    assessment = score_candidate(
        QUESTION,
        semantic_similarity=0.55,
        source="Diretriz-avaliacao-queda-feminina.pdf",
        text=(
            "Avaliação: registrar padrão da rarefação, evolução, sinais de inflamação, "
            "tricoscopia e relação temporal com medicamentos."
        ),
    )
    procedure = score_candidate(
        QUESTION,
        semantic_similarity=0.58,
        source="Protocolo-mesoterapia.pdf",
        text="Procedimento: mesoterapia com aplicação intradérmica e microagulhamento.",
    )
    assert assessment.adjusted_score > procedure.adjusted_score


def test_query_variants_include_medication_and_scalp_focus():
    variants = " ".join(build_query_variants(QUESTION))
    assert "anamnese medicamentosa" in variants
    assert "descamacao" in variants


def test_metadata_marks_adverse_procedure_report():
    metadata = enrich_chunk_metadata(
        "Hairlossatinjectionsitesofmesotherapyforalopecia.pdf",
        "Após mesoterapia houve dor intensa, eritema, edema, atrofia e cicatriz.",
    )
    assert metadata["content_type"] == "adverse_event"
    assert metadata["topic"] == "procedure_complication"
