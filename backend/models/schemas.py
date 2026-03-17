"""
MediScan AI - Pydantic Schemas
Request validation and response serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Auth Schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email       : EmailStr
    password    : str = Field(min_length=8)
    full_name   : str = Field(min_length=2, max_length=255)
    institution : Optional[str] = None
    role        : str = Field(default="clinician")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"clinician", "researcher", "admin"}
        if v not in allowed:
            raise ValueError(f"Role must be one of: {allowed}")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit.")
        return v


class LoginRequest(BaseModel):
    email    : EmailStr
    password : str


class TokenResponse(BaseModel):
    access_token  : str
    refresh_token : str
    token_type    : str = "bearer"
    expires_in    : int  # seconds


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id          : str
    email       : str
    full_name   : str
    role        : str
    institution : Optional[str]
    is_active   : bool
    created_at  : datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Diagnosis Schemas
# ---------------------------------------------------------------------------

class DiagnosisCreateRequest(BaseModel):
    patient_ref    : Optional[str] = Field(None, max_length=100)
    patient_age    : Optional[int] = Field(None, ge=0, le=150)
    patient_sex    : Optional[str] = None
    clinical_notes : Optional[str] = None

    @field_validator("patient_sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"M", "F", "Other", "Unknown"}:
            raise ValueError("patient_sex must be M, F, Other, or Unknown")
        return v


class DiagnosisResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_class   : str
    confidence        : float
    all_probabilities : Dict[str, float]
    risk_level        : str
    clinical_note     : str
    gradcam_overlay   : Optional[str]   # base64 PNG
    model_version     : str


class DiagnosisResponse(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    id                 : str
    patient_ref        : Optional[str]
    patient_age        : Optional[int]
    patient_sex        : Optional[str]
    clinical_notes     : Optional[str]
    original_filename  : Optional[str]
    predicted_class    : str
    confidence         : float
    risk_level         : str
    clinical_note      : str
    all_probabilities  : Dict[str, float]
    gradcam_overlay    : Optional[str] = None
    model_version      : str
    reviewed           : bool
    reviewer_verdict   : Optional[str]
    reviewer_notes     : Optional[str]
    created_at         : datetime
    reviewed_at        : Optional[datetime]


class DiagnosisListResponse(BaseModel):
    items      : List[DiagnosisResponse]
    total      : int
    page       : int
    page_size  : int
    total_pages: int


class ReviewRequest(BaseModel):
    verdict : str
    notes   : Optional[str] = None

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        allowed = {"CONFIRMED", "OVERRIDDEN", "PENDING"}
        if v not in allowed:
            raise ValueError(f"Verdict must be one of: {allowed}")
        return v


# ---------------------------------------------------------------------------
# Health / Info Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status        : str
    model_loaded  : bool
    version       : str
    timestamp     : datetime


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name    : str
    architecture  : str
    version       : str
    classes       : List[str]
    image_size    : int
    disclaimer    : str