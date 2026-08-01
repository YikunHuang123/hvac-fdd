"""
Supervised Graph Neural Network (GNN) classifier for the LBNL SDAHU dataset.

Uses Graph Structure Learning (GSL) to automatically infer thermodynamic coupling
between sensors/features to improve 6-class diagnosis over flat tabular models (like XGBoost).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.detection.base import NUMERIC_FEATURES
from hvac_fdd.exceptions import ModelPersistenceError

logger = logging.getLogger(__name__)


class GraphConvLayer(nn.Module):
    """
    A native PyTorch implementation of a Graph Convolutional Layer.
    H' = softmax(A) H W
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N, in_features)
        # adj: (N, N)
        adj_norm = F.softmax(adj, dim=-1)
        support = torch.matmul(adj_norm, x) 
        out = self.linear(support)
        return out


class GNNClassifierModel(nn.Module):
    """
    GNN for multivariate time series classification.
    """
    def __init__(self, num_nodes: int, hidden_dim: int, num_classes: int):
        super().__init__()
        # Graph Structure Learning: Learnable adjacency matrix
        self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
        # Encoder (Input feature per node is 1, output is hidden_dim)
        self.gc1 = GraphConvLayer(1, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, hidden_dim)
        
        # Readout & Classification
        self.fc1 = nn.Linear(num_nodes * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, num_nodes, 1)
        h1 = F.relu(self.gc1(x, self.adj))
        h2 = F.relu(self.gc2(h1, self.adj))
        
        # Flatten all node embeddings for classification
        h2_flat = h2.view(h2.size(0), -1)
        
        # Classifier head
        out = F.relu(self.fc1(h2_flat))
        logits = self.fc2(out)
        return logits


class GNNClassifierWrapper:
    """
    Scikit-learn style wrapper for GNNClassifierModel to act as a backend
    for FaultClassifier.
    """
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._scaler = StandardScaler()
        self._model: GNNClassifierModel | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GNNClassifierWrapper":
        # X is already filtered to NUMERIC_FEATURES and no NaN
        X_np = self._scaler.fit_transform(X)
        y_np = y
        self.classes_ = np.unique(y_np)
        
        num_nodes = X_np.shape[1]
        num_classes = len(self.classes_)
        
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y_np, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, 
            batch_size=self._settings.gnn_batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        self._model = GNNClassifierModel(
            num_nodes=num_nodes, 
            hidden_dim=self._settings.gnn_hidden_dim,
            num_classes=num_classes
        ).to(self._device)
        
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._settings.gnn_learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        epochs = self._settings.gnn_epochs
        logger.info(f"Training GNN Classifier on {self._device} with {len(X_tensor)} samples (classes={num_classes})")
        
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
            
        logger.info(f"GNN Classifier fitted for classes: {self.classes_}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted yet.")
            
        X_np = self._scaler.transform(X)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1).to(self._device)
        
        self._model.eval()
        probs_list = []
        with torch.no_grad():
            chunk_size = 4096
            for i in range(0, len(X_tensor), chunk_size):
                batch = X_tensor[i:i+chunk_size]
                logits = self._model(batch)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs.cpu().numpy())
                
        return np.concatenate(probs_list, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self._encoder.inverse_transform(pred_idx)
        
    def save(self, path: Path | str) -> None:
        if self._model is None:
            raise ModelPersistenceError("Cannot save an unfitted GNNClassifier")
            
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "scaler": self._scaler,
            "classes_": self.classes_,
            "num_nodes": self._model.gc1.linear.in_features,
            "actual_num_nodes": self._model.adj.shape[0],
            "hidden_dim": self._settings.gnn_hidden_dim,
            "num_classes": len(self.classes_),
            "model_state": self._model.state_dict()
        }
        
        try:
            torch.save(state, out)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to save GNN to {out}: {exc}") from exc

    @classmethod
    def load(cls, path: Path | str, settings: Settings | None = None) -> "GNNClassifierWrapper":
        src = Path(path)
        try:
            state = torch.load(src, weights_only=True)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to load GNN from {src}: {exc}") from exc
            
        obj = cls(settings=settings)
        obj._scaler = state["scaler"]
        obj.classes_ = state["classes_"]
        
        num_nodes = state["actual_num_nodes"]
        hidden_dim = state.get("hidden_dim", 32)
        num_classes = state["num_classes"]
        
        obj._model = GNNClassifierModel(
            num_nodes=num_nodes, 
            hidden_dim=hidden_dim,
            num_classes=num_classes
        ).to(obj._device)
        
        obj._model.load_state_dict(state["model_state"])
        return obj
