"""
main.py — FastAPI application for credit card fraud anomaly detection.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.model import detector
from app.schemas import (
    BatchPredictionResponse,
    BatchTransactionInput,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionInput,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        detector.load()
    except Exception as e:
        print(f"[startup] WARNING: {e}")
        print("[startup] API will start but /predict endpoints will return 503 until model is trained.")
    yield


app = FastAPI(
    title="FraudGuard — Credit Card Anomaly Detection API",
    description=(
        "Detects fraudulent credit card transactions using an Isolation Forest model "
        "trained on the Kaggle creditcard dataset. Returns anomaly score, fraud label, "
        "and feature-level explanation for each transaction."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ─── Health & info ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"message": "FraudGuard API is running. Visit /docs for the full API reference."}

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=detector.is_loaded,
        model_version=detector.version,
        training_samples=detector.meta.get("training_samples"),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    m = detector.meta
    return ModelInfoResponse(
        model_type=m["model_type"],
        model_version=m["version"],
        n_estimators=m["n_estimators"],
        contamination=m["contamination"],
        feature_count=m["feature_count"],
        training_samples=m["training_samples"],
        threshold=m["threshold"],
    )

# ─── Prediction endpoints ──────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: TransactionInput):
    """
    Analyse a single transaction and return:
    - **is_fraud**: boolean
    - **label**: FRAUD or NORMAL
    - **anomaly_score**: raw Isolation Forest score (more negative = more anomalous)
    - **confidence**: 0–1 normalised fraud likelihood
    - **explanation**: top features driving the decision
    """
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    result = detector.predict(transaction.model_dump())
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(payload: BatchTransactionInput):
    """
    Analyse up to 500 transactions in a single request.
    Returns per-transaction results plus aggregate counts.
    """
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    transactions = [t.model_dump() for t in payload.transactions]
    results = detector.predict_batch(transactions)

    fraud_count = sum(1 for r in results if r["is_fraud"])
    return BatchPredictionResponse(
        results=[PredictionResponse(**r) for r in results],
        total=len(results),
        fraud_count=fraud_count,
        normal_count=len(results) - fraud_count,
    )