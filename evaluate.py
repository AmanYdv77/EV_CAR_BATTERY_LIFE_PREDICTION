"""
Model evaluation, quantitative error metrics (RMSE/MAE), and plotting routines.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import subprocess
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import config

def get_git_commit_hash() -> str:
    """Returns the current HEAD git commit hash or 'unknown'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(config.PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit
    except Exception:
        return "unknown"

def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    max_rul: float,
    device: Optional[torch.device] = None,
    save_metrics: bool = True
) -> Dict[str, Any]:
    """
    Evaluates model across val_loader, rescales predictions to actual cycles,
    computes RMSE/MAE, and optionally records the result in results/metrics.json.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    preds, actuals = [], []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            p = model(xb).cpu().numpy()
            
            # Handle 0D scalar vs 1D array
            p = np.atleast_1d(p)
            y_arr = np.atleast_1d(yb.numpy())
            
            preds.extend(p * max_rul)
            actuals.extend(y_arr * max_rul)
            
    preds_arr = np.maximum(0.0, np.array(preds, dtype=np.float64))
    actuals_arr = np.array(actuals, dtype=np.float64)
    
    if len(preds_arr) == 0:
        return {"rmse": 0.0, "mae": 0.0, "actual": [], "predicted": []}
        
    rmse = float(np.sqrt(mean_squared_error(actuals_arr, preds_arr)))
    mae = float(mean_absolute_error(actuals_arr, preds_arr))
    
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "actual": [round(float(v), 2) for v in actuals_arr],
        "predicted": [round(float(v), 2) for v in preds_arr]
    }
    
    if save_metrics:
        save_metrics_log(result)
        
    return result

def save_metrics_log(metric_entry: Dict[str, Any]) -> None:
    """
    Appends a new metric run entry to results/metrics.json.
    Never overwrites prior runs.
    """
    metrics_file = config.RESULTS_DIR / "metrics.json"
    history: List[Dict[str, Any]] = []
    
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    history = content
                elif isinstance(content, dict):
                    history = [content]
        except Exception:
            history = []
            
    # Save clean summary without full large arrays
    log_item = {
        "timestamp": metric_entry["timestamp"],
        "git_commit": metric_entry["git_commit"],
        "rmse": metric_entry["rmse"],
        "mae": metric_entry["mae"]
    }
    history.append(log_item)
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def generate_prediction_plot(
    actual: List[float],
    predicted: List[float],
    save_path: Optional[Path] = None,
    battery_id: Optional[str] = None
) -> Path:
    """
    Generates and saves the Actual vs Predicted RUL plot to results/prediction_plot.png.
    """
    if save_path is None:
        save_path = config.RESULTS_DIR / "prediction_plot.png"
        
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual RUL", color="#2563EB", linewidth=2)
    plt.plot(predicted, label="Predicted RUL", color="#DC2626", linestyle="--", linewidth=2)
    title = f"Battery Degradation Trajectory (RUL)"
    if battery_id:
        title += f" - {battery_id}"
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Discharge Cycle (Index)", fontsize=10)
    plt.ylabel("Remaining Useful Life (Cycles)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
