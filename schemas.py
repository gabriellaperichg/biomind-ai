from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    must_change_password: bool


class CreateUserRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=10, max_length=200)
    role: str = "biomedica"


class AskRequest(BaseModel):
    texto: str = Field(min_length=3, max_length=10_000)


class CreateChatRequest(BaseModel):
    title: str | None = None