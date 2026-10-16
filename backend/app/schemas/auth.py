"""Auth request/response schemas."""
from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    user_id: int
    email: EmailStr
    full_name: str | None = None
    role: str

    class Config:
        from_attributes = True
