from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(
    r"[a-z0-9]+(?:[-_/][a-z0-9]+)*",
    re.I,
)


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.lower())

    return "".join(
        c for c in value
        if not unicodedata.combining(c)
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


class BM25Index:

    def __init__(
        self,
        ids: list[str],
        texts: list[str],
        metadata: list[dict] | None = None,
    ):
        self.ids = ids
        self.texts = texts

        self.metadata = metadata or [
            {}
            for _ in ids
        ]

        self.tokens = [
            tokenize(text)
            for text in texts
        ]

        self.doc_len = [
            len(tokens)
            for tokens in self.tokens
        ]

        self.avgdl = (
            sum(self.doc_len)
            / max(len(self.doc_len), 1)
        )

        self.k1 = 1.5
        self.b = 0.75

        df = Counter()

        for tokens in self.tokens:
            for token in set(tokens):
                df[token] += 1

        n = len(self.tokens)

        self.idf = {
            token: math.log(
                1
                + (n - freq + 0.5)
                / (freq + 0.5)
            )
            for token, freq in df.items()
        }

    def search(
        self,
        query: str,
        k: int = 20,
    ) -> list[dict]:

        qtokens = tokenize(query)

        if not qtokens:
            return []

        results = []

        for idx, tokens in enumerate(self.tokens):

            counts = Counter(tokens)

            score = 0.0

            for token in qtokens:

                freq = counts.get(token)

                if not freq:
                    continue

                idf = self.idf.get(token, 0.0)

                denom = (
                    freq
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b
                        * self.doc_len[idx]
                        / max(self.avgdl, 1)
                    )
                )

                score += (
                    idf
                    * (
                        freq
                        * (self.k1 + 1)
                    )
                    / denom
                )

            if score > 0:

                results.append(
                    {
                        "chunk_id": self.ids[idx],
                        "score": score,
                        "text": self.texts[idx],
                        "metadata": self.metadata[idx],
                    }
                )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:k]


def build_index(
    records: list[dict],
    path: str | Path,
) -> None:

    payload = {
        "ids": [
            str(r["chunk_id"])
            for r in records
        ],

        "texts": [
            str(r["text"])
            for r in records
        ],

        "metadata": [
            {
                key: value
                for key, value in r.items()
                if key not in {
                    "chunk_id",
                    "text",
                }
            }
            for r in records
        ],
    }

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(path).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_index(
    path: str | Path,
) -> BM25Index | None:

    path = Path(path)

    if not path.exists():
        return None

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return BM25Index(
        ids=data.get("ids", []),
        texts=data.get("texts", []),
        metadata=data.get("metadata", []),
    )