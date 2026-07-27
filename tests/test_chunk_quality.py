from chunk import limpar_texto


def test_cleaning_removes_license_identity():
    text = (
        "Conteúdo licenciado para Pessoa de Teste - 109.772.806-40\n"
        "Avaliação: observar descamação e oleosidade."
    )
    cleaned = limpar_texto(text)
    assert "109.772.806-40" not in cleaned
    assert "Avaliação" in cleaned


def test_cleaning_creates_semantic_boundaries_before_treatment():
    text = (
        "Paciente com queda difusa. Diagnóstico: confirmado por tricoscopia. "
        "Conduta: iniciar tratamento."
    )
    cleaned = limpar_texto(text)
    assert "\n\nDiagnóstico:" in cleaned
    assert "\n\nConduta:" in cleaned


def test_case_description_and_treatment_become_separate_chunks():
    from chunk import PageSpan, empacotar_unidades, extrair_unidades

    raw = (
        "Paciente feminina com oleosidade e queda na coroa. "
        "Exames: ferritina reduzida. "
        "Conduta: prescrever minoxidil e suplementação."
    )
    cleaned = limpar_texto(raw)
    units = extrair_unidades(cleaned, [PageSpan(1, 0, len(cleaned))])
    chunks = empacotar_unidades(units, "AAG-Caso3.pdf")

    assert len(chunks) >= 3
    assert chunks[0]["content_type"] == "general"
    assert any(chunk["content_type"] == "treatment" for chunk in chunks)
    assert "prescrever" not in chunks[0]["text"].lower()
