"""
MediScan AI - Database Models
SQLAlchemy ORM definitions for all tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id              = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email           = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name       = Column(String(255), nullable=False)
    role            = Column(String(50), default="clinician")   # clinician | admin | researcher
    institution     = Column(String(255), nullable=True)
    is_active       = Column(Boolean, default=True)
    is_verified     = Column(Boolean, default=False)
    created_at      = Column(DateTime(timezone=True), default=utcnow)
    last_login      = Column(DateTime(timezone=True), nullable=True)

    diagnoses       = relationship("Diagnosis", back_populates="user", cascade="all, delete-orphan")
    tokens          = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id               = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id          = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Patient context (de-identified)
    patient_ref      = Column(String(100), nullable=True)   # Clinician's internal reference
    patient_age      = Column(Integer, nullable=True)
    patient_sex      = Column(String(10), nullable=True)
    clinical_notes   = Column(Text, nullable=True)

    # Image metadata
    original_filename = Column(String(255), nullable=True)
    image_path        = Column(String(512), nullable=True)   # Path in local storage

    # Model output
    predicted_class    = Column(String(50), nullable=False)
    confidence         = Column(Float, nullable=False)
    risk_level         = Column(String(20), nullable=False)
    clinical_note      = Column(Text, nullable=False)
    all_probabilities  = Column(JSON, nullable=False)
    gradcam_path       = Column(String(512), nullable=True)
    model_version      = Column(String(100), nullable=False)

    # Clinician review
    reviewed           = Column(Boolean, default=False)
    reviewer_verdict   = Column(String(50), nullable=True)   # CONFIRMED | OVERRIDDEN | PENDING
    reviewer_notes     = Column(Text, nullable=True)

    created_at         = Column(DateTime(timezone=True), default=utcnow)
    reviewed_at        = Column(DateTime(timezone=True), nullable=True)

    user               = relationship("User", back_populates="diagnoses")

    def __repr__(self) -> str:
        return f"<Diagnosis {self.id} {self.predicted_class} {self.confidence:.1f}%>"


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id         = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked    = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    user       = relationship("User", back_populates="tokens")

    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_refresh_token_hash"),
    )
