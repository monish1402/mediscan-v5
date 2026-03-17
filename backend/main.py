"""
MediScan AI - FastAPI Application Entry Point
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.auth    import router as auth_router
from backend.routes.history import router as history_router
from backend.routes.predict import router as predict_router
from backend.utils.database import init_db
from backend.models.schemas import HealthResponse, ModelInfoResponse

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mediscan")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
MODEL_PATH   = os.getenv("MODEL_PATH", "./model/mediscan_v5.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Initialise database
    init_db()
    logger.info("Database initialised.")

    # Load ML model
    from model.inference import engine
    try:
        engine.load(MODEL_PATH)
        logger.info("Inference engine ready.")
    except FileNotFoundError:
        logger.warning(
            "Model file not found at '%s'. "
            "Download mediscan_v5.pth from Kaggle and place it at that path.",
            MODEL_PATH
        )

    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title       = "MediScan AI",
    description = "AI-Assisted Chest X-Ray Diagnosis — Research Platform",
    version     = "5.0.0",
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
    lifespan    = lifespan,
)

# CORS — restrict to your frontend origin in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(history_router)


# Health check
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health() -> dict:
    from model.inference import engine
    return {
        "status"      : "operational",
        "model_loaded": engine.is_loaded,
        "version"     : "5.0.0",
        "timestamp"   : datetime.now(timezone.utc),
    }


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info() -> dict:
    return {
        "model_name"  : "MediScan AI",
        "architecture": "EfficientNetB4 + Transfer Learning + Grad-CAM",
        "version"     : "5.0.0",
        "classes"     : ["NORMAL", "PNEUMONIA"],
        "image_size"  : 380,
        "disclaimer"  : (
            "This system is intended for research and educational purposes only. "
            "It is not a certified medical device and must not be used as a sole "
            "basis for clinical decision-making. Always consult a qualified "
            "radiologist or physician."
        ),
    }


# Serve React frontend for all non-API routes
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        index = FRONTEND_DIR / "index.html"
        return FileResponse(str(index))