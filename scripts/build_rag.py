"""Executa a reconstrução completa da base RAG V4.

O banco de usuários da aplicação não é alterado por este script.
"""
from __future__ import annotations
import subprocess
import sys


def main() -> None:
    commands = [
        [sys.executable, "-m", "scripts.ingest_v4"],
        [sys.executable, "embed.py", "build"],
    ]
    for command in commands:
        print("\n>>>", " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    print("\nRAG V4 reconstruído com sucesso.")


if __name__ == "__main__":
    main()
