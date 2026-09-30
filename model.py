"""
PyTorch Dataset and CNN-LSTM Neural Network Architecture.
"""
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import config

class BatteryDataset(Dataset):
    """
    Sliding window sequence dataset for battery discharge cycles.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = config.SEQ_LEN):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        
    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = torch.tensor(self.X[idx:idx + self.seq_len], dtype=torch.float32)
        y_val = torch.tensor(self.y[idx + self.seq_len], dtype=torch.float32)
        return x_seq, y_val

class CNN_LSTM(nn.Module):
    """
    Hybrid 1D-CNN + 2-layer LSTM architecture:
    - 1D-CNN filters extract local spatial features from voltage/current/temperature signals.
    - LSTM layers capture temporal degradation trajectories across sequential cycles.
    """
    def __init__(self, feature_len: int = config.FEATURE_LEN):
        super().__init__()
        self.feature_len = feature_len
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # After two MaxPool1d(2), feature length is feature_len // 4
        flattened_cnn_dim = 32 * (feature_len // 4)
        self.lstm = nn.LSTM(
            input_size=flattened_cnn_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, channels=3, signal_length=200)
        batch_size, seq_len, channels, signal_length = x.shape
        x_reshaped = x.view(batch_size * seq_len, channels, signal_length)
        
        cnn_features = self.cnn(x_reshaped)
        cnn_out = cnn_features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(cnn_out)
        prediction = self.fc(lstm_out[:, -1, :]).squeeze(-1)
        return prediction
