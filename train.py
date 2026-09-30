"""
Training pipeline for CNN-LSTM EV Battery RUL model.
"""
from typing import Optional, Tuple
from pathlib import Path
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from data import process_data_from_files, normalize_data
from model import BatteryDataset, CNN_LSTM
from evaluate import evaluate_model, generate_prediction_plot

def train_pipeline(
    data_dir: Optional[Path] = None,
    epochs: int = config.DEFAULT_EPOCHS,
    seq_len: int = config.SEQ_LEN,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE
) -> Tuple[nn.Module, dict]:
    """
    Executes the end-to-end training pipeline:
    1. Loads and extracts cycles from NASA .mat files
    2. Applies MinMax input normalization and target RUL scaling
    3. Builds sliding window sequences
    4. Trains CNN-LSTM neural network
    5. Saves weights to models/cnn_lstm.pth and scalers to models/scalers.pkl
    6. Evaluates on held-out validation set and saves results
    """
    if data_dir is not None:
        os.environ["BATTERY_DATA_DIR"] = str(data_dir)
        
    print("Locating battery dataset files...")
    files = config.get_battery_files()
    print(f"Found {len(files)} battery files: {[Path(f).name for f in files]}")
    
    print("Ingesting and processing discharge cycles...")
    X_raw, y_raw = process_data_from_files(files, resample_len=config.FEATURE_LEN)
    print(f"Total cycle observations extracted: {len(X_raw)}, Shape: {X_raw.shape}")
    
    print("Applying multi-level signal and target scaling...")
    X_scaled, y_scaled, scaler_X, max_rul = normalize_data(X_raw, y_raw)
    
    # 80/20 train/validation split
    split_idx = int(0.8 * len(X_scaled))
    train_dataset = BatteryDataset(X_scaled[:split_idx], y_scaled[:split_idx], seq_len=seq_len)
    val_dataset = BatteryDataset(X_scaled[split_idx:], y_scaled[split_idx:], seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing CNN-LSTM model on device: {device}")
    
    model = CNN_LSTM(feature_len=config.FEATURE_LEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"Beginning training across {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            
        epoch_loss = running_loss / max(1, len(train_dataset))
        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] - Training Loss (MSE): {epoch_loss:.6f}")
            
    print("Saving model weights and scalers to models/ directory...")
    model_save_path = config.MODELS_DIR / "cnn_lstm.pth"
    torch.save(model.state_dict(), model_save_path)
    
    scalers_save_path = config.MODELS_DIR / "scalers.pkl"
    joblib.dump({"scaler_X": scaler_X, "max_rul": max_rul}, scalers_save_path)
    print(f"Artifacts saved: {model_save_path}, {scalers_save_path}")
    
    print("Running evaluation on validation split...")
    eval_results = evaluate_model(model, val_loader, max_rul=max_rul, device=device, save_metrics=True)
    print(f"Validation Performance -> RMSE: {eval_results['rmse']:.2f} cycles, MAE: {eval_results['mae']:.2f} cycles")
    
    plot_path = generate_prediction_plot(eval_results["actual"], eval_results["predicted"])
    print(f"Evaluation plot saved to: {plot_path}")
    
    return model, eval_results

if __name__ == "__main__":
    train_pipeline(epochs=20)
