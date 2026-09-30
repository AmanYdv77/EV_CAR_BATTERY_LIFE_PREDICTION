"""
FastAPI Model Serving API for EV Battery RUL Prediction Platform.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import json
import joblib
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.utils.data import DataLoader

import config
from data import process_single_battery, normalize_data
from model import BatteryDataset, CNN_LSTM
from evaluate import evaluate_model

# Pydantic Response Schemas
class HealthResponse(BaseModel):
    status: str

class MetricResponse(BaseModel):
    timestamp: str
    git_commit: str
    rmse: float
    mae: float

class BatteryPredictionResponse(BaseModel):
    battery_id: str
    actual: List[float]
    predicted: List[float]
    rmse: float
    mae: float

# Singleton Model & Scaler State
state: Dict[str, Any] = {
    "model": None,
    "scaler_X": None,
    "max_rul": None,
    "device": None
}

def load_artifacts() -> None:
    """Loads model weights and scalers into memory once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["device"] = device
    
    model_path = config.MODELS_DIR / "cnn_lstm.pth"
    scalers_path = config.MODELS_DIR / "scalers.pkl"
    
    if not model_path.exists() or not scalers_path.exists():
        print(f"Warning: Model artifacts not found in {config.MODELS_DIR}. Run train.py first.")
        return
        
    scalers = joblib.load(scalers_path)
    state["scaler_X"] = scalers["scaler_X"]
    state["max_rul"] = float(scalers["max_rul"])
    
    model = CNN_LSTM(feature_len=config.FEATURE_LEN).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    state["model"] = model
    print("CNN-LSTM model and scalers loaded successfully into memory.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(
    title="EV Battery RUL Predictor API",
    description="Inference service for battery remaining useful life estimation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_available_battery_ids() -> List[str]:
    """Returns sorted list of available battery IDs based on existing .mat files."""
    try:
        files = config.get_battery_files()
        return sorted([Path(f).stem for f in files])
    except FileNotFoundError:
        return []

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root_health():
    """Health check endpoint verifying service operational status."""
    return {"status": "ok"}

@app.get("/api/batteries", response_model=List[str], tags=["Batteries"])
async def list_batteries():
    """Returns list of battery IDs available for inference in DATA_DIR."""
    return get_available_battery_ids()

@app.get("/api/predict/{battery_id}", response_model=BatteryPredictionResponse, tags=["Inference"])
async def predict_battery_rul(battery_id: str):
    """
    Runs model inference for a specific battery ID across its discharge cycles.
    Returns actual vs predicted RUL sequences alongside RMSE and MAE.
    """
    if state["model"] is None or state["scaler_X"] is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Ensure models/cnn_lstm.pth and models/scalers.pkl exist."
        )
        
    mat_path = config.DATA_DIR / f"{battery_id}.mat"
    if not mat_path.exists():
        available = get_available_battery_ids()
        raise HTTPException(
            status_code=404,
            detail=f"Battery ID '{battery_id}' not found. Available batteries: {available}"
        )
        
    try:
        X_bat, y_bat = process_single_battery(str(mat_path), resample_len=config.FEATURE_LEN)
        if len(X_bat) <= config.SEQ_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"Battery '{battery_id}' contains fewer cycles ({len(X_bat)}) than sequence length ({config.SEQ_LEN})."
            )
            
        X_scaled, y_scaled, _, _ = normalize_data(
            X_bat, y_bat,
            scaler_X=state["scaler_X"],
            max_rul=state["max_rul"]
        )
        
        dataset = BatteryDataset(X_scaled, y_scaled, seq_len=config.SEQ_LEN)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        eval_result = evaluate_model(
            state["model"],
            loader,
            max_rul=state["max_rul"],
            device=state["device"],
            save_metrics=False
        )
        
        return {
            "battery_id": battery_id,
            "actual": eval_result["actual"],
            "predicted": eval_result["predicted"],
            "rmse": eval_result["rmse"],
            "mae": eval_result["mae"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/api/metrics", response_model=MetricResponse, tags=["Metrics"])
async def get_latest_metrics():
    """Returns the most recent model evaluation metrics from results/metrics.json."""
    metrics_file = config.RESULTS_DIR / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Run training pipeline first."
        )
        
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if isinstance(data, list) and len(data) > 0:
            latest = data[-1]
            return latest
        elif isinstance(data, dict):
            return data
        else:
            raise HTTPException(status_code=404, detail="No metric entries recorded.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")
