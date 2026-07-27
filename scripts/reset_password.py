from __future__ import annotations

from getpass import getpass

from sqlalchemy import select

from database import SessionLocal
from models import User, utc_now
from security import hash_password

MIN_PASSWORD_LENGTH = 10


def main() -> None:
    email = input("E-mail da conta: ").strip().lower()
    password = getpass("Nova senha: ")
    confirmation = getpass("Confirme a nova senha: ")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise SystemExit(
            f"A senha deve ter pelo menos {MIN_PASSWORD_LENGTH} caracteres."
        )
    if password != confirmation:
        raise SystemExit("As senhas não correspondem.")

    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == email))
        if user is None:
            raise SystemExit("Não existe uma conta com esse e-mail no banco atual.")

        user.password_hash = hash_password(password)
        user.is_active = True
        user.failed_login_attempts = 0
        user.locked_until = None
        user.updated_at = utc_now()
        db.commit()

    print("Senha redefinida e conta desbloqueada com sucesso.")


if __name__ == "__main__":
    main()
