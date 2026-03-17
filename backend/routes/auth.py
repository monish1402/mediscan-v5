"""
MediScan AI - Authentication Routes
POST /api/auth/register
POST /api/auth/login
POST /api/auth/refresh
POST /api/auth/logout
GET  /api/auth/me
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.models.db import RefreshToken, User
from backend.models.schemas import (
    LoginRequest, RefreshRequest, RegisterRequest,
    TokenResponse, UserResponse
)
from backend.utils.auth import (
    ACCESS_TOKEN_EXPIRE, create_access_token, create_refresh_token,
    hash_password, hash_refresh_token, refresh_token_expiry, verify_password
)
from backend.utils.database import get_db
from backend.middleware.auth_middleware import get_current_user

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> User:
    """Create a new user account."""
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists."
        )

    user = User(
        email           = body.email,
        hashed_password = hash_password(body.password),
        full_name       = body.full_name,
        institution     = body.institution,
        role            = body.role,
        is_active       = True,
        is_verified     = False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> dict:
    """Authenticate user and return access + refresh tokens."""
    user = db.query(User).filter(User.email == body.email, User.is_active == True).first()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )

    # Issue tokens
    access_token            = create_access_token(user.id, user.email, user.role)
    raw_refresh, hash_rt    = create_refresh_token()

    # Persist refresh token
    rt = RefreshToken(
        user_id    = user.id,
        token_hash = hash_rt,
        expires_at = refresh_token_expiry(),
    )
    db.add(rt)

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    return {
        "access_token" : access_token,
        "refresh_token": raw_refresh,
        "token_type"   : "bearer",
        "expires_in"   : ACCESS_TOKEN_EXPIRE * 60,
    }


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)) -> dict:
    """Exchange a valid refresh token for a new token pair."""
    token_hash = hash_refresh_token(body.refresh_token)
    now        = datetime.now(timezone.utc)

    rt = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
        RefreshToken.revoked    == False,
        RefreshToken.expires_at > now,
    ).first()

    if not rt:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid or has expired."
        )

    user = db.query(User).filter(User.id == rt.user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")

    # Rotate tokens
    rt.revoked              = True
    access_token            = create_access_token(user.id, user.email, user.role)
    raw_refresh, hash_new   = create_refresh_token()

    new_rt = RefreshToken(
        user_id    = user.id,
        token_hash = hash_new,
        expires_at = refresh_token_expiry(),
    )
    db.add(new_rt)
    db.commit()

    return {
        "access_token" : access_token,
        "refresh_token": raw_refresh,
        "token_type"   : "bearer",
        "expires_in"   : ACCESS_TOKEN_EXPIRE * 60,
    }


@router.post("/logout")
def logout(
    body: RefreshRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Revoke refresh token on logout."""
    token_hash = hash_refresh_token(body.refresh_token)
    rt = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash,
        RefreshToken.user_id    == current_user.id
    ).first()
    if rt:
        rt.revoked = True
        db.commit()
    return {"detail": "Logged out successfully."}


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> User:
    """Return the authenticated user's profile."""
    return current_user
