"""
MediScan AI - History & Review Routes
GET    /api/history               - Paginated diagnosis history
DELETE /api/history/{id}          - Delete a diagnosis
PATCH  /api/history/{id}/review   - Clinician review/verdict
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.middleware.auth_middleware import get_current_user
from backend.models.db import Diagnosis, User
from backend.models.schemas import DiagnosisListResponse, DiagnosisResponse, ReviewRequest
from backend.utils.database import get_db

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=DiagnosisListResponse)
def list_history(
    page         : int          = Query(1, ge=1),
    page_size    : int          = Query(10, ge=1, le=100),
    risk_level   : str | None   = Query(None, description="Filter by risk level: LOW | MODERATE | HIGH"),
    reviewed     : bool | None  = Query(None, description="Filter by reviewed status"),
    current_user : User         = Depends(get_current_user),
    db           : Session      = Depends(get_db),
) -> dict:
    """Return paginated diagnosis history for the authenticated user."""
    query = db.query(Diagnosis).filter(Diagnosis.user_id == current_user.id)

    if risk_level:
        query = query.filter(Diagnosis.risk_level == risk_level.upper())
    if reviewed is not None:
        query = query.filter(Diagnosis.reviewed == reviewed)

    total       = query.count()
    total_pages = max(1, math.ceil(total / page_size))
    items       = (
        query
        .order_by(Diagnosis.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items"      : items,
        "total"      : total,
        "page"       : page,
        "page_size"  : page_size,
        "total_pages": total_pages,
    }


@router.delete("/{diagnosis_id}")
def delete_diagnosis(
    diagnosis_id : str,
    current_user : User    = Depends(get_current_user),
    db           : Session = Depends(get_db),
) -> dict:
    """Permanently delete a diagnosis record. Users can only delete their own."""
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagnosis not found.")
    if diagnosis.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    db.delete(diagnosis)
    db.commit()
    return {"detail": "Diagnosis deleted."}


@router.patch("/{diagnosis_id}/review", response_model=DiagnosisResponse)
def review_diagnosis(
    diagnosis_id : str,
    body         : ReviewRequest,
    current_user : User          = Depends(get_current_user),
    db           : Session       = Depends(get_db),
) -> Diagnosis:
    """Submit clinician review verdict for a diagnosis."""
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagnosis not found.")
    if diagnosis.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    diagnosis.reviewed         = True
    diagnosis.reviewer_verdict = body.verdict
    diagnosis.reviewer_notes   = body.notes
    diagnosis.reviewed_at      = datetime.now(timezone.utc)
    db.commit()
    db.refresh(diagnosis)
    return diagnosis
