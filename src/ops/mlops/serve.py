"""
Simple FastAPI model serving for latest registered model.

Provides endpoints:
 - GET /health
 - POST /predict  {"model_name": "xgboost", "records": [{...}, ...]}

Notes:
 - If a transformer artifact is present next to the model (or attached in registry),
   incoming JSON records (dicts) will be converted to DataFrame and transformed
   before prediction. Otherwise, the server expects numeric feature arrays.

Run:
  pip install -r requirements.txt
  uvicorn src.ops.mlops.serve:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import os
import pandas as pd

from .registry import ModelRegistry
from ...utils import IOHandler


class PredictRequest(BaseModel):
    model_name: Optional[str] = None
    records: List[Dict[str, Any]]


app = FastAPI(title="Churn Prediction Serving")

# Default registry dir (should match config). Adjust if needed.
REGISTRY_DIR = os.environ.get("MODEL_REGISTRY_DIR", "artifacts/registry")
registry = ModelRegistry(REGISTRY_DIR)


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        models = registry.list_models()
        return {"status": "ok", "models_registered": list(models.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _load_transformer_from_info(info: Dict[str, Any]) -> Optional[Any]:
    # Look for transformer in artifacts or in same folder as model path
    artifacts = info.get("artifacts", {}) or {}
    transformer_path = artifacts.get("transformer")
    if transformer_path and os.path.exists(transformer_path):
        return IOHandler.load_model(transformer_path)

    model_path = info.get("path")
    if model_path:
        # check same folder
        cand = os.path.join(os.path.dirname(model_path), "transformer.joblib")
        if os.path.exists(cand):
            return IOHandler.load_model(cand)

    return None


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        model_name = payload.model_name or (list(registry.list_models().keys())[-1] if registry.list_models() else None)
        if model_name is None:
            raise HTTPException(status_code=400, detail="No model registered")

        models = registry.list_models(model_name)
        if model_name not in models or not models[model_name]:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        latest_info = models[model_name][-1]
        model = IOHandler.load_model(latest_info["path"]) if latest_info.get("path") else registry.get_latest_model(model_name)

        transformer = _load_transformer_from_info(latest_info)

        # Build input DataFrame
        df = pd.DataFrame(payload.records)

        # If transformer exists, apply it; else assume DF is numeric and order matches model
        if transformer is not None:
            try:
                X = transformer.transform(df)
            except Exception:
                # If transform expects fit/transform API, try fit_transform fallback
                X = transformer.fit_transform(df)
            # transformer may return numpy array or DataFrame
            if isinstance(X, (list, tuple)):
                X_pred = X
            else:
                X_pred = X
        else:
            X_pred = df

        # Predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_pred)
            # return probability of positive class if available
            positive = probs[:, 1].tolist()
            preds = model.predict(X_pred).tolist()
        else:
            preds = model.predict(X_pred).tolist()
            positive = None

        return {"model": model_name, "predictions": preds, "prob_positive": positive}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
