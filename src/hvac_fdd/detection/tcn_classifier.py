"""
Supervised Temporal Convolutional Network (TCN) classifier for the LBNL SDAHU dataset.

Uses Causal Dilated Convolutions to capture temporal context over a sliding window
of sequence length T.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.detection.base import NUMERIC_FEATURES
from hvac_fdd.exceptions import ModelPersistenceError

logger = logging.getLogger(__name__)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x is (Batch, Features, Seq_Len)
        y1 = self.network(x)
        # We only care about the last timestep in the sequence for classification
        return self.linear(y1[:, :, -1])


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None, seq_len: int):
        # Store data as PyTorch tensors, avoiding duplicates in memory
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        # Slice the contiguous tensor on the fly: shape (seq_len, num_features)
        x_window = self.X[idx : idx + self.seq_len]
        # Transpose to (num_features, seq_len) as expected by Conv1d
        x_window = x_window.transpose(0, 1)
        
        if self.y is not None:
            # Target is the label of the last timestep in the window
            y_label = self.y[idx + self.seq_len - 1]
            return x_window, y_label
        return x_window, torch.tensor(-1)


class TCNClassifierWrapper:
    """
    Scikit-learn style wrapper for TCNModel to act as a backend for FaultClassifier.
    Automatically handles sliding window conversion.
    """
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._scaler = StandardScaler()
        self._model: TCNModel | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TCNClassifierWrapper":
        # X is already filtered to NUMERIC_FEATURES and no NaN
        X_np = self._scaler.fit_transform(X)
        self.classes_ = np.unique(y)
        
        num_features = X_np.shape[1]
        num_classes = len(self.classes_)
        
        logger.info(f"Initializing SlidingWindowDataset (seq_len={self._settings.tcn_seq_len}) from {len(X_np)} rows...")
        dataset = SlidingWindowDataset(X_np, y, self._settings.tcn_seq_len)
        
        loader = DataLoader(
            dataset, 
            batch_size=self._settings.tcn_batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        channel_sizes = [self._settings.tcn_hidden_dim] * self._settings.tcn_levels
        
        self._model = TCNModel(
            num_inputs=num_features,
            num_channels=channel_sizes,
            num_classes=num_classes,
            kernel_size=self._settings.tcn_kernel_size
        ).to(self._device)
        
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._settings.tcn_learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        epochs = self._settings.tcn_epochs
        logger.info(f"Training TCN Classifier on {self._device} with {len(dataset)} windows (classes={num_classes})")
        
        self._model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                
                optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch_x.size(0)
                
            avg_loss = total_loss / len(dataset)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
        logger.info(f"TCN Classifier fitted for classes: {self.classes_}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted yet.")
            
        seq_len = self._settings.tcn_seq_len
        if len(X) < seq_len:
            # Fallback for very small datasets: just predict normal, or pad.
            # For simplicity, pad with the first row.
            pad_size = seq_len - len(X)
            padded_X = np.vstack([np.tile(X[0], (pad_size, 1)), X])
            X_np = self._scaler.transform(padded_X)
        else:
            X_np = self._scaler.transform(X)
            
        dataset = SlidingWindowDataset(X_np, None, seq_len)
        loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)
        
        self._model.eval()
        probs_list = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self._device)
                logits = self._model(batch_x)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs_list.append(probs.cpu().numpy())
                
        window_probs = np.concatenate(probs_list, axis=0)
        
        # We lose the first (seq_len - 1) predictions due to windowing.
        # Pad the beginning with the first window's prediction to maintain original shape.
        pad_size = len(X) - len(window_probs)
        if pad_size > 0:
            first_prob = window_probs[0:1]
            padded_probs = np.concatenate([np.tile(first_prob, (pad_size, 1)), window_probs], axis=0)
            return padded_probs
            
        return window_probs

    def save(self, path: Path | str) -> None:
        if self._model is None:
            raise ModelPersistenceError("Cannot save an unfitted TCNClassifier")
            
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "scaler": self._scaler,
            "classes_": self.classes_,
            "num_features": self._model.network[0].conv1.in_channels,
            "hidden_dim": self._settings.tcn_hidden_dim,
            "kernel_size": self._settings.tcn_kernel_size,
            "levels": self._settings.tcn_levels,
            "num_classes": len(self.classes_),
            "model_state": self._model.state_dict()
        }
        
        try:
            torch.save(state, out)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to save TCN to {out}: {exc}") from exc

    @classmethod
    def load(cls, path: Path | str, settings: Settings | None = None) -> "TCNClassifierWrapper":
        src = Path(path)
        try:
            state = torch.load(src, weights_only=True)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to load TCN from {src}: {exc}") from exc
            
        obj = cls(settings=settings)
        obj._scaler = state["scaler"]
        obj.classes_ = state["classes_"]
        
        num_features = state["num_features"]
        hidden_dim = state.get("hidden_dim", 32)
        kernel_size = state.get("kernel_size", 3)
        levels = state.get("levels", 3)
        num_classes = state["num_classes"]
        
        channel_sizes = [hidden_dim] * levels
        
        obj._model = TCNModel(
            num_inputs=num_features,
            num_channels=channel_sizes,
            num_classes=num_classes,
            kernel_size=kernel_size
        ).to(obj._device)
        
        obj._model.load_state_dict(state["model_state"])
        return obj
