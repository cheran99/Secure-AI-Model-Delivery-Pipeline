from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from .schemas import PredictRequest
import joblib
from .logger import logger
import numpy as np

app = FastAPI(title="Fraud-Score Model")

MODEL_PATH = "model.joblib"

def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"], data.get("features", [])

model, features = load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model)}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        x = np.array(req.features).reshape(1, -1)
        if x.shape[1] != len(features):
            raise HTTPException(status_code=400, detail="unexpected feature length")
        score = model.predict_proba(x)[0, 1] if hasattr(model, "predict_proba") else model.predict(x)[0]
        # Basic auditing/logging: don't log raw features in prod for PII safety — log hashes or summaries instead
        logger.info(f"prediction requested; score={float(score):.4f}")
        return {"score": float(score)}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))