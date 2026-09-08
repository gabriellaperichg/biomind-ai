from __future__ import annotations

from dataclasses import dataclass
import re

from ingestion.parsers import ParsedDocument, ParsedPage


@dataclass
class StructuralBlock:
    document_id: str
    page_start: int
    page_end: int
    title: str
    section_path: list[str]
    text: str
    block_type: str = "text"
    has_visual_content: bool = False


NUMBERED_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*[\)\.]?|[IVXLCDM]+[\)\.]?)\s+.{2,180}$",
    re.I,
)

KNOWN_HEADINGS = re.compile(
    r"^(defini[cç][aã]o|objetivo|respons[aá]veis?|materiais?|"
    r"descri[cç][aã]o|procedimentos?|tratamento|diagn[oó]stico|"
    r"avalia[cç][aã]o|anamnese|resultados?|m[eé]todos?|discuss[aã]o|"
    r"conclus[aã]o|refer[eê]ncias?|contraindica[cç][oõ]es?|"
    r"efeitos?\s+adversos?|casos?\s+cl[ií]nicos?)\s*:?\s*$",
    re.I,
)


def _looks_like_heading(line: str) -> bool:
    line = " ".join(line.split()).strip()
    if not line or len(line) > 180:
        return False
    if KNOWN_HEADINGS.match(line):
        return True
    if NUMBERED_HEADING.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 6:
        upper = sum(c.isupper() for c in letters)
        if upper / len(letters) >= 0.82:
            return True
    return False


def _clean_lines(text: str) -> list[str]:
    return [" ".join(line.split()).strip() for line in text.splitlines() if line.strip()]


def build_structure(document: ParsedDocument) -> list[StructuralBlock]:
    blocks: list[StructuralBlock] = []
    section_path: list[str] = []

    for page in document.pages:
        if document.document_type == "lecture_slides":
            title = page.title or document.title
            content = page.text.strip()
            section = section_path.copy()
            if title and title not in section:
                # Slide title vira a seção final.
                section = [*section[:2], title]
            blocks.append(
                StructuralBlock(
                    document_id=document.document_id,
                    page_start=page.page,
                    page_end=page.page,
                    title=title,
                    section_path=section,
                    text=content,
                    block_type="slide",
                    has_visual_content=page.image_count > 0,
                )
            )
            continue

        lines = _clean_lines(page.text)
        buffer: list[str] = []
        current_title = ""

        def flush() -> None:
            nonlocal buffer, current_title
            if not buffer:
                return
            content = "\n".join(buffer).strip()
            if not content:
                buffer = []
                return
            blocks.append(
                StructuralBlock(
                    document_id=document.document_id,
                    page_start=page.page,
                    page_end=page.page,
                    title=current_title or page.title or document.title,
                    section_path=section_path.copy(),
                    text=content,
                    block_type="section" if current_title else "text",
                    has_visual_content=page.image_count > 0,
                )
            )
            buffer = []

        for line in lines:
            if _looks_like_heading(line):
                flush()
                current_title = line.rstrip(":")
                # Atualiza hierarquia simples: numerados profundos viram níveis.
                m = re.match(r"^(\d+(?:\.\d+)*)", line)
                if m:
                    level = len(m.group(1).split("."))
                    section_path[:] = section_path[: max(0, level - 1)]
                elif section_path and current_title.lower() in {x.lower() for x in section_path}:
                    pass
                section_path.append(current_title)
            else:
                buffer.append(line)

        flush()

    if not blocks:
        return [
            StructuralBlock(
                document_id=document.document_id,
                page_start=1,
                page_end=max(1, document.pages[-1].page if document.pages else 1),
                title=document.title,
                section_path=[],
                text="",
            )
        ]

    return blocks
