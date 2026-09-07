from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value).strip()


def build_query_variants(query: str, max_variants: int = 4) -> list[str]:
    """Cria poucas variações determinísticas para ampliar a cobertura."""
    base = query.strip()
    normalized = normalize(base)
    variants: list[str] = [base]

    if any(word in normalized for word in ("remedio", "medicamento", "farmaco", "antidepressivo", "anticoncepcional")):
        variants.append(f"{base} anamnese medicamentosa efeitos relacionados")

    if any(word in normalized for word in ("descamacao", "escamando", "coceira", "prurido", "oleoso", "inflamacao")):
        variants.append(f"{base} avaliação do couro cabeludo inflamação descamação")

    if any(word in normalized for word in ("queda", "coroa", "frontal", "difusa", "rareifacao", "rarefacao")):
        variants.append(f"{base} padrão localização evolução da perda capilar")

    if any(word in normalized for word in ("procedimento", "mesoterapia", "microagulhamento", "injecao", "intralesional")):
        variants.append(f"{base} complicações eventos adversos avaliação pós procedimento")

    # Remove duplicatas preservando a ordem.
    result: list[str] = []
    seen = set()
    for item in variants:
        key = normalize(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
        if len(result) >= max_variants:
            break
    return result
