"""
MediScan AI - Prediction Routes
POST /api/predict          - Submit image for diagnosis
GET  /api/predict/{id}     - Retrieve a diagnosis by ID
GET  /api/predict/{id}/gradcam - Return Grad-CAM image
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from backend.middleware.auth_middleware import get_current_user
from backend.models.db import Diagnosis, User
from backend.models.schemas import DiagnosisResponse
from backend.utils.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predict", tags=["Diagnosis"])

# Allowed MIME types for uploaded images
ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"}
MAX_BYTES    = 15 * 1024 * 1024  # 15 MB

UPLOAD_DIR   = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", response_model=DiagnosisResponse, status_code=status.HTTP_201_CREATED)
async def create_diagnosis(
    image          : UploadFile = File(..., description="Chest X-ray image (JPEG/PNG/BMP/TIFF)"),
    patient_ref    : str | None = Form(None),
    patient_age    : str | None = Form(None),
    patient_sex    : str | None = Form(None),
    clinical_notes : str | None = Form(None),
    current_user   : User       = Depends(get_current_user),
    db             : Session    = Depends(get_db),
) -> Diagnosis:
    """
    Accept a chest X-ray image, run the EfficientNetB4 model, store and return the diagnosis.
    """
    from model.inference import engine

    # Coerce empty strings to None
    patient_ref    = patient_ref    or None
    patient_sex    = patient_sex    or None
    clinical_notes = clinical_notes or None
    age: int | None = None
    if patient_age and patient_age.strip():
        try:
            age = int(patient_age)
        except ValueError:
            raise HTTPException(status_code=422, detail="patient_age must be an integer.")

    # Validate MIME type
    if image.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Accepted: JPEG, PNG, BMP, TIFF."
        )

    # Read and validate size
    image_bytes = await image.read()
    if len(image_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds maximum allowed size of 15 MB."
        )

    # Ensure model is loaded
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Diagnostic model is not available. Contact the system administrator."
        )

    # Run inference
    try:
        result = engine.predict(image_bytes)
    except Exception as exc:
        logger.exception("Inference failed for user %s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed. Please try again or contact support."
        )

    # Persist image to disk
    image_filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{current_user.id[:8]}_{image.filename}"
    image_path     = UPLOAD_DIR / image_filename
    image_path.write_bytes(image_bytes)

    # Persist Grad-CAM overlay
    gradcam_path = None
    if result.get("gradcam_overlay"):
        import base64
        gradcam_filename = image_filename.replace(".", "_gradcam.")
        gradcam_path     = str(UPLOAD_DIR / gradcam_filename)
        Path(gradcam_path).write_bytes(base64.b64decode(result["gradcam_overlay"]))

    # Store diagnosis record
    diagnosis = Diagnosis(
        user_id           = current_user.id,
        patient_ref       = patient_ref,
        patient_age       = age,
        patient_sex       = patient_sex,
        clinical_notes    = clinical_notes,
        original_filename = image.filename,
        image_path        = str(image_path),
        predicted_class   = result["predicted_class"],
        confidence        = result["confidence"],
        risk_level        = result["risk_level"],
        clinical_note     = result["clinical_note"],
        all_probabilities = result["all_probabilities"],
        gradcam_path      = gradcam_path,
        model_version     = result["model_version"],
    )
    db.add(diagnosis)
    db.commit()
    db.refresh(diagnosis)

    # Attach Grad-CAM base64 to response (not persisted in DB, only on disk)
    diagnosis.__dict__["gradcam_overlay"] = result.get("gradcam_overlay")

    logger.info(
        "Diagnosis %s created — user=%s class=%s confidence=%.2f%%",
        diagnosis.id, current_user.id, diagnosis.predicted_class, diagnosis.confidence
    )
    return diagnosis


@router.get("/{diagnosis_id}", response_model=DiagnosisResponse)
def get_diagnosis(
    diagnosis_id : str,
    current_user : User    = Depends(get_current_user),
    db           : Session = Depends(get_db),
) -> Diagnosis:
    """Retrieve a single diagnosis record. Users can only access their own records."""
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagnosis not found.")
    if diagnosis.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    return diagnosis


@router.get("/{diagnosis_id}/gradcam")
def get_gradcam(
    diagnosis_id : str,
    current_user : User    = Depends(get_current_user),
    db           : Session = Depends(get_db),
) -> Response:
    """Return Grad-CAM overlay image as PNG bytes."""
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagnosis not found.")
    if diagnosis.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    if not diagnosis.gradcam_path or not Path(diagnosis.gradcam_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Grad-CAM image not available.")

    return Response(
        content      = Path(diagnosis.gradcam_path).read_bytes(),
        media_type   = "image/png",
        headers      = {"Cache-Control": "private, max-age=3600"},
    )