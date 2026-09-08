"""Cria uma cópia de segurança do banco da aplicação.

Não altera o banco original.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import shutil


def main() -> None:
    source = Path("data/biomind_app.db")
    if not source.exists():
        raise SystemExit(f"Banco não encontrado: {source}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = source.with_name(f"biomind_app_backup_{stamp}.db")
    shutil.copy2(source, target)
    print(f"Backup criado: {target}")


if __name__ == "__main__":
    main()
