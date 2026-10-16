"""POST /api/v1/auth/register and POST /api/v1/auth/token (login).

Standard OAuth2PasswordBearer JWT flow used by the FastAPI gateway
described in the deck. Role can be 'planner' (default) or 'admin'.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.v1.deps import get_db
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.auth import Token, UserOut
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None
    role: str = "planner"


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new planner or admin account.

    Returns the created user object (without password). Duplicate email
    returns 409 Conflict.
    """
    existing = db.execute(
        select(User).where(User.email == request.email)
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An account with email '{request.email}' already exists.",
        )

    valid_roles = {"planner", "admin"}
    if request.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"role must be one of {sorted(valid_roles)}",
        )

    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        full_name=request.full_name,
        role=request.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut(
        user_id=user.user_id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
    )


@router.post("/token", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """OAuth2 password flow. Returns a JWT bearer token.

    username field is the account email address.
    """
    user = db.execute(
        select(User).where(User.email == form_data.username)
    ).scalar_one_or_none()

    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    token = create_access_token(subject=user.email)
    return Token(access_token=token)


@router.get("/me", response_model=UserOut)
def get_current_user_info(
    db: Session = Depends(get_db),
    # TODO: add Depends(get_current_user) once token validation is wired into deps.py
):
    """Placeholder: returns 501 until the Bearer token dependency is added to deps.py."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Wire up Depends(get_current_user) in deps.py to enable this endpoint.",
    )
