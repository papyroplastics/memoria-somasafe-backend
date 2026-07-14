import secrets
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pwdlib import PasswordHash
from pydantic import BaseModel
from sqlmodel import Session, select

from api.lib.session import (
    get_current_user,
    oauth2_scheme,
    put_access,
    revoke_access,
    revoke_all_access,
    lookup_access,
    token_hash,
)
from common.config import ACCESS_TOKEN_TTL_SECONDS, REFRESH_TOKEN_TTL_SECONDS
from common.db import AuthSession, User, get_session, utcnow

router = APIRouter(prefix="/auth")
password_hash = PasswordHash.recommended()

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_TTL_SECONDS


class RefreshRequest(BaseModel):
    refresh_token: str


class UserPublic(BaseModel):
    id: int
    username: str
    email: str | None = None


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def _new_session(session: Session, user_id: int) -> TokenPair:
    access = secrets.token_urlsafe(32)
    refresh = secrets.token_urlsafe(32)
    row = AuthSession(
        user_id=user_id,
        refresh_hash=token_hash(refresh),
        refresh_expires_at=utcnow() + timedelta(seconds=REFRESH_TOKEN_TTL_SECONDS),
    )
    session.add(row)
    session.commit()
    put_access(access, user_id, row.id)
    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(),
          session: Session = Depends(get_session)) -> TokenPair:
    user = session.exec(select(User).where(User.username == form_data.username)).first()

    if user is None:
        # equalize timing to protect agains user enumeration
        password_hash.verify(form_data.password, "dummypassword") 
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

    if user.disabled or not password_hash.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

    return _new_session(session, user.id)


@router.post("/refresh")
def refresh(body: RefreshRequest,
            session: Session = Depends(get_session)) -> TokenPair:
    row = session.exec(
        select(AuthSession).where(AuthSession.refresh_hash == token_hash(body.refresh_token))
    ).first()
    if row is None or row.revoked or row.refresh_expires_at <= utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid refresh token")
    # Rotate: the old session is revoked and a fresh pair is issued.
    row.revoked = True
    session.add(row)
    session.commit()
    return _new_session(session, row.user_id)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(token: str = Depends(oauth2_scheme),
           session: Session = Depends(get_session)) -> None:
    data = lookup_access(token)
    revoke_access(token)
    if data is not None:
        row = session.get(AuthSession, uuid.UUID(data["session_id"]))
        if row is not None:
            row.revoked = True
            session.add(row)
            session.commit()


@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
def logout_all(user: User = Depends(get_current_user),
               session: Session = Depends(get_session)) -> None:
    revoke_all_access(user.id)
    rows = session.exec(
        select(AuthSession).where(AuthSession.user_id == user.id,
                                  AuthSession.revoked == False)  # noqa: E712
    ).all()
    for row in rows:
        row.revoked = True
        session.add(row)
    session.commit()


@router.get("/me")
def me(user: User = Depends(get_current_user)) -> UserPublic:
    return UserPublic(id=user.id, username=user.username, email=user.email)
