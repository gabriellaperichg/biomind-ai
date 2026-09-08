"""Mostra os bancos SQLite existentes sem modificá-los."""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path


def inspect_db(path: Path) -> None:
    print(f"\nBanco: {path}")
    print(f"Tamanho: {path.stat().st_size:,} bytes")
    with sqlite3.connect(path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (table,) in tables:
            count = conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            print(f"  {table}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["data/biomind_app.db"])
    args = parser.parse_args()

    for value in args.paths:
        path = Path(value)
        if not path.exists():
            print(f"[NÃO ENCONTRADO] {path}")
            continue
        inspect_db(path)


if __name__ == "__main__":
    main()
