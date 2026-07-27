"""Cria a primeira conta administradora pelo terminal local.

Uso:
    python -m scripts.create_admin
"""

from __future__ import annotations

from getpass import getpass

from email_validator import EmailNotValidError, validate_email
from sqlalchemy import select

from database import SessionLocal
from models import User, new_uuid, utc_now
from security import hash_password


MIN_PASSWORD_LENGTH = 10


def read_email() -> str:
    while True:
        raw = input("E-mail: ").strip().lower()

        try:
            return validate_email(
                raw,
                check_deliverability=False,
            ).normalized.lower()
        except EmailNotValidError as exc:
            print(f"E-mail inválido: {exc}")


def read_password() -> str:
    while True:
        password = getpass("Senha inicial: ")
        confirmation = getpass("Confirme a senha: ")

        if len(password) < MIN_PASSWORD_LENGTH:
            print(
                f"A senha deve ter pelo menos {MIN_PASSWORD_LENGTH} caracteres."
            )
            continue

        if password != confirmation:
            print("As senhas não correspondem.")
            continue

        return password


def main() -> None:
    name = input("Nome da administradora: ").strip()

    if len(name) < 2:
        raise SystemExit("O nome deve ter pelo menos 2 caracteres.")

    email = read_email()
    password = read_password()

    with SessionLocal() as db:
        existing = db.scalar(
            select(User).where(User.email == email)
        )

        if existing is not None:
            raise SystemExit("Já existe uma conta com esse e-mail.")

        now = utc_now()

        user = User(
            id=new_uuid(),
            name=name,
            email=email,
            password_hash=hash_password(password),
            role="admin",
            is_active=True,
            must_change_password=False,
            failed_login_attempts=0,
            created_at=now,
            updated_at=now,
        )

        db.add(user)
        db.commit()

    print("Administradora criada com sucesso.")


if __name__ == "__main__":
    main()
