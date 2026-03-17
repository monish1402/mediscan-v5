# MediScan AI — Production Diagnostic Platform
### Version 5.0 | EfficientNetB4 + FastAPI + JWT Auth

> An AI-assisted chest X-ray diagnosis platform for research and educational use.
> Built with EfficientNetB4, transfer learning, Grad-CAM explainability, JWT
> authentication, and a professional clinical-grade web interface.

---

## Architecture Overview

```
mediscan-v5/
│
├── model/
│   ├── kaggle_train.py     — Full training pipeline (run on Kaggle)
│   └── inference.py        — Production inference engine + Grad-CAM
│
├── backend/
│   ├── main.py             — FastAPI application entry point
│   ├── routes/
│   │   ├── auth.py         — Register, login, refresh, logout, /me
│   │   ├── predict.py      — POST /api/predict, GET /api/predict/{id}
│   │   └── history.py      — GET/DELETE /api/history, PATCH review
│   ├── models/
│   │   ├── db.py           — SQLAlchemy ORM: User, Diagnosis, RefreshToken
│   │   └── schemas.py      — Pydantic v2 request/response schemas
│   ├── middleware/
│   │   └── auth_middleware.py  — JWT Bearer dependency
│   └── utils/
│       ├── auth.py         — JWT creation, password hashing
│       └── database.py     — Session factory, init_db
│
├── frontend/
│   └── index.html          — Complete SPA: Auth, Dashboard, Diagnose, History
│
├── docker/
│   └── Dockerfile.backend
├── docker-compose.yml
└── requirements.txt
```

---

## API Reference

### Authentication

| Method | Endpoint              | Auth | Description                |
|--------|-----------------------|------|----------------------------|
| POST   | /api/auth/register    | No   | Create new user account    |
| POST   | /api/auth/login       | No   | Login, receive token pair  |
| POST   | /api/auth/refresh     | No   | Rotate access/refresh tokens |
| POST   | /api/auth/logout      | Yes  | Revoke refresh token       |
| GET    | /api/auth/me          | Yes  | Get current user profile   |

### Diagnosis

| Method | Endpoint                        | Auth | Description                      |
|--------|---------------------------------|------|----------------------------------|
| POST   | /api/predict                    | Yes  | Submit image, get diagnosis      |
| GET    | /api/predict/{id}               | Yes  | Get single diagnosis record      |
| GET    | /api/predict/{id}/gradcam       | Yes  | Get Grad-CAM image as PNG        |

### History

| Method | Endpoint                        | Auth | Description                      |
|--------|---------------------------------|------|----------------------------------|
| GET    | /api/history                    | Yes  | Paginated diagnosis list         |
| DELETE | /api/history/{id}               | Yes  | Delete a diagnosis record        |
| PATCH  | /api/history/{id}/review        | Yes  | Submit clinician review verdict  |

### System

| Method | Endpoint       | Auth | Description         |
|--------|----------------|------|---------------------|
| GET    | /api/health    | No   | Health check        |
| GET    | /api/model-info| No   | Model metadata      |
| GET    | /api/docs      | No   | Swagger UI          |

---

## Authentication Flow

```
1. POST /api/auth/register  →  Create account
2. POST /api/auth/login     →  Get access_token (30min) + refresh_token (7 days)
3. All protected requests   →  Authorization: Bearer <access_token>
4. On 401                   →  POST /api/auth/refresh with refresh_token
5. POST /api/auth/logout    →  Revoke refresh token
```

Refresh tokens are rotated on every use (stored as SHA-256 hash in DB).

---

## Diagnosis API Workflow

```
POST /api/predict
  Content-Type: multipart/form-data
  Authorization: Bearer <token>
  Body:
    image          (file)    — Chest X-ray image
    patient_ref    (string)  — Internal patient reference (optional)
    patient_age    (int)     — Patient age (optional)
    patient_sex    (string)  — M | F | Other | Unknown (optional)
    clinical_notes (string)  — Free-text notes (optional)

Response 201:
  {
    "id"               : "uuid",
    "predicted_class"  : "PNEUMONIA",
    "confidence"       : 94.31,
    "risk_level"       : "HIGH",
    "clinical_note"    : "Radiographic signs consistent with pneumonia...",
    "all_probabilities": { "NORMAL": 5.69, "PNEUMONIA": 94.31 },
    "model_version"    : "MediScan-EfficientNetB4-v5",
    "reviewed"         : false,
    ...
  }
```

---

## Setup & Running

### Step 1 — Train the Model on Kaggle

1. Open Kaggle → New Notebook → Enable GPU T4
2. Add dataset: "Chest X-Ray Images (Pneumonia)" by Paul Mooney
3. Paste the contents of `model/kaggle_train.py` into cells
4. Run all cells (~15 min)
5. Download `mediscan_v5.pth` from the Output tab
6. Place it at `model/mediscan_v5.pth`

### Step 2 — Configure Environment

```bash
cp .env.example .env
# Edit .env and set SECRET_KEY
```

### Step 3 — Run with Docker

```bash
docker-compose up --build
```

Backend: http://localhost:8000
API Docs: http://localhost:8000/api/docs

### Step 4 — Open Frontend

Open `frontend/index.html` in your browser.
Set `const API = "http://localhost:8000"` (already default).

### Alternative: Run Locally Without Docker

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Model Architecture

```
Input (380x380 RGB)
      |
EfficientNetB4 Backbone (pretrained ImageNet)
  - Phase 1 (epochs 1-5):  Backbone frozen, classifier trains
  - Phase 2 (epochs 6-25): Last 30 backbone layers + classifier fine-tune
      |
Custom Classifier Head:
  Dropout(0.4) → Linear(1792→512) → BatchNorm → SiLU → Dropout(0.2) → Linear(512→2)
      |
Softmax → [NORMAL, PNEUMONIA] probabilities
      |
Grad-CAM (last conv block) → Heatmap overlay
```

### Training Details
- Optimizer: AdamW with cosine annealing
- Loss: CrossEntropyLoss with label smoothing (0.1)
- Mixed precision (FP16) for fast GPU training
- Weighted random sampler for class imbalance
- Early stopping (patience=6) on validation AUC

---

## Disclaimer

This platform is intended for **research and educational purposes only**.
It is **not** a certified medical device and must **not** be used as the
sole basis for any clinical decision. All AI-generated findings must be
independently reviewed by a qualified radiologist or physician.
