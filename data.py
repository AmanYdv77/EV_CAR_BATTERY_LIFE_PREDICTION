"""
Data ingestion, MATLAB parsing, signal resampling, and normalization.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import config

def load_battery_mat(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads raw NASA battery degradation cycle data from a .mat file.
    Dynamically identifies the battery root key (e.g. B0005).
    """
    mat = loadmat(file_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No battery data keys found in {file_path}")
    battery_id = keys[0]
    battery_data = mat[battery_id][0, 0]["cycle"][0]
    
    cycles = []
    for entry in battery_data:
        if entry["type"][0] == "discharge":
            data = entry["data"][0][0]
            cycles.append({
                "voltage": data["Voltage_measured"][0],
                "current": data["Current_measured"][0],
                "temperature": data["Temperature_measured"][0],
                "capacity": float(data["Capacity"][0][0])
            })
    return cycles

def resample(signal: np.ndarray, length: int = config.FEATURE_LEN) -> np.ndarray:
    """
    Resamples a 1D sensor signal to a fixed sequence length via linear interpolation.
    """
    if len(signal) == 0:
        return np.zeros(length, dtype=np.float32)
    return np.interp(np.linspace(0, len(signal) - 1, length), np.arange(len(signal)), signal)

def process_single_battery(file_path: str, resample_len: int = config.FEATURE_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes cycles from a single battery file into features X and target RUL y.
    """
    cycles = load_battery_mat(file_path)
    if not cycles:
        return np.empty((0, 3, resample_len)), np.empty((0,))
        
    caps = [c["capacity"] for c in cycles]
    initial_cap = caps[0]
    
    # End-of-life threshold defined at 70% of initial capacity
    eol_idx = next((i for i, cap in enumerate(caps) if cap <= 0.7 * initial_cap), len(caps) - 1)
    rul = np.array([max(eol_idx - i, 0) for i in range(len(caps))], dtype=np.float32)
    
    X = []
    for c in cycles:
        v_res = resample(c["voltage"], resample_len)
        i_res = resample(c["current"], resample_len)
        t_res = resample(c["temperature"], resample_len)
        X.append(np.stack([v_res, i_res, t_res], axis=0))
        
    return np.array(X, dtype=np.float32), rul

def process_data_from_files(file_list: List[str], resample_len: int = config.FEATURE_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes a list of battery .mat files and aggregates into concatenated arrays.
    """
    all_X, all_y = [], []
    for f in file_list:
        X_bat, y_bat = process_single_battery(f, resample_len)
        if len(X_bat) > 0:
            all_X.append(X_bat)
            all_y.append(y_bat)
            
    if not all_X:
        raise ValueError("No valid cycle data extracted from the provided files.")
        
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

def normalize_data(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    scaler_X: Optional[MinMaxScaler] = None,
    max_rul: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, float]:
    """
    Applies MinMax scaling to 3D input sequences and normalizes RUL targets by max_rul.
    """
    n_samples, n_channels, n_len = X_raw.shape
    X_flat = X_raw.reshape(n_samples, -1)
    
    if scaler_X is None:
        scaler_X = MinMaxScaler()
        X_scaled_flat = scaler_X.fit_transform(X_flat)
    else:
        X_scaled_flat = scaler_X.transform(X_flat)
        
    X_scaled = X_scaled_flat.reshape(n_samples, n_channels, n_len).astype(np.float32)
    
    if max_rul is None:
        max_rul = float(y_raw.max()) if len(y_raw) > 0 else 1.0
        
    y_scaled = (y_raw / max_rul).astype(np.float32) if max_rul > 0 else y_raw.astype(np.float32)
    return X_scaled, y_scaled, scaler_X, max_rul
