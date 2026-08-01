"""
Kolmogorov-Arnold Network (KAN) anomaly detector for the LBNL SDAHU dataset.

This implements a Deep Learning Autoencoder using KAN layers instead of MLPs.
It learns to reconstruct normal HVAC physics. Anomalies are detected when
the reconstruction error (MSE) significantly exceeds the normal distribution.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# pykan library
from kan import KAN

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.detection.base import DetectorBase, NUMERIC_FEATURES, PREDICT_OUTPUT_COLS
from hvac_fdd.domain import AlertLevel, FaultType
from hvac_fdd.exceptions import DetectorNotFittedError, ModelPersistenceError

logger = logging.getLogger(__name__)

_DETECTOR_SOURCE = "kan_autoencoder"

# Percentile thresholds computed from the normal-only training reconstruction errors.
_THRESHOLD_WARNING_PCT  = 95.0
_THRESHOLD_CRITICAL_PCT = 99.0


class KANDetector(DetectorBase):
    """
    Deep Learning Autoencoder anomaly detector using Kolmogorov-Arnold Networks.

    Fit: Train KAN to reconstruct normal data.
    Predict: Return rows flagged as anomalies with anomaly_index = MSE_error.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self._settings = settings or get_settings()
        self._scaler: StandardScaler | None = None
        
        self._input_dim: int = 0
        self._model: KAN | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._threshold_warning:  float = 0.0
        self._threshold_critical: float = 0.0

    # ── DetectorBase interface ────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "KANDetector":
        """
        Fit the KAN Autoencoder on FaultType.NORMAL rows only.

        Args:
            df: Feature-engineered DataFrame.
        """
        normal_df = df[df["fault_type"] == FaultType.NORMAL.value]
        # Downsample for KAN to speed up training, since B-splines are computationally heavy
        MAX_KAN_SAMPLES = 500000
        if len(normal_df) > MAX_KAN_SAMPLES:
            normal_df = normal_df.sample(n=MAX_KAN_SAMPLES, random_state=self._settings.random_state)
            
        X_np = _clean_features(normal_df)
        self._input_dim = X_np.shape[1]

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        
        # Build a robust Autoencoder: [input_dim, input_dim, input_dim]
        # using pykan. We preserve the dimensionality to let KAN non-linear edges filter features.
        hidden_dim = self._input_dim
        self._model = KAN(
            width=[self._input_dim, hidden_dim, self._input_dim], 
            grid=10, 
            k=3, 
            seed=self._settings.random_state
        ).to(self._device)

        logger.info(f"Training KAN-AD on {self._device} with {len(X_np)} samples (dim={self._input_dim}->{hidden_dim}->{self._input_dim})")
        
        # Training loop
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        criterion = torch.nn.MSELoss()
        
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)
        
        epochs = 5
        self._model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred_y = self._model(batch_x)
                loss = criterion(pred_y, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
            scheduler.step()
            avg_loss = total_loss / len(dataset)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

        # Compute thresholds based on reconstruction error of training data
        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_tensor)
            # MSE per sample
            errors = torch.mean((X_tensor - preds) ** 2, dim=1).cpu().numpy()

        self._threshold_warning  = float(np.percentile(errors, _THRESHOLD_WARNING_PCT))
        self._threshold_critical = float(np.percentile(errors, _THRESHOLD_CRITICAL_PCT))

        self._is_fitted = True
        logger.info(
            "KAN-AD fitted. Thresholds: Warning >= %.4f, Critical >= %.4f",
            self._threshold_warning, self._threshold_critical,
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all rows; return only rows flagged as anomalies.
        """
        self._require_fitted()

        valid_mask = df[NUMERIC_FEATURES].notna().all(axis=1)
        valid_df   = df[valid_mask]
        if valid_df.empty:
            return pd.DataFrame(columns=PREDICT_OUTPUT_COLS)

        X_np = self._scaler.transform(valid_df[NUMERIC_FEATURES].values)  # type: ignore
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self._device)

        self._model.eval()
        with torch.no_grad():
            # chunk inference to avoid OOM on huge datasets
            preds_list = []
            chunk_size = 4096
            for i in range(0, len(X_tensor), chunk_size):
                batch = X_tensor[i:i+chunk_size]
                preds_list.append(self._model(batch))
            
            preds = torch.cat(preds_list, dim=0)
            errors = torch.mean((X_tensor - preds) ** 2, dim=1).cpu().numpy()

        anomaly_mask = errors >= self._threshold_warning
        if not anomaly_mask.any():
            return pd.DataFrame(columns=PREDICT_OUTPUT_COLS)

        sub = valid_df.iloc[np.where(anomaly_mask)[0]].copy()
        sub["anomaly_index"]   = errors[anomaly_mask]
        sub["violated_policy"] = "RECONSTRUCTION_ERROR"
        sub["trigger_signal"]  = "kan_mse"
        sub["detector_source"] = _DETECTOR_SOURCE
        sub["alert_level"]     = sub["anomaly_index"].map(self._map_alert_level)

        logger.debug("KAN-AD predict: %d anomalies / %d valid rows", len(sub), len(valid_df))
        return _select_output(sub)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        if not self._is_fitted:
            raise DetectorNotFittedError("Cannot save an unfitted KANDetector")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        # PyTorch model state dict
        model_state = self._model.state_dict() if self._model else None
        
        try:
            joblib.dump(
                {
                    "scaler":             self._scaler,
                    "input_dim":          self._input_dim,
                    "model_state":        model_state,
                    "threshold_warning":  self._threshold_warning,
                    "threshold_critical": self._threshold_critical,
                },
                out,
            )
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to save model to {out}: {exc}") from exc
        logger.info("KANDetector saved to %s", out)

    @classmethod
    def load(
        cls,
        path: Path | str,
        settings: Settings | None = None,
    ) -> "KANDetector":
        src = Path(path)
        try:
            data = joblib.load(src)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to load model from {src}: {exc}") from exc
        
        obj = cls(settings=settings)
        obj._scaler             = data["scaler"]
        obj._input_dim          = data["input_dim"]
        obj._threshold_warning  = data["threshold_warning"]
        obj._threshold_critical = data["threshold_critical"]
        
        hidden_dim = obj._input_dim
        obj._model = KAN(
            width=[obj._input_dim, hidden_dim, obj._input_dim], 
            grid=10, 
            k=3, 
            seed=obj._settings.random_state
        ).to(obj._device)
        
        if data["model_state"]:
            obj._model.load_state_dict(data["model_state"])
            
        obj._is_fitted = True
        logger.info("KANDetector loaded from %s", src)
        return obj

    def _map_alert_level(self, score: float) -> str:
        if score >= self._threshold_critical:
            return AlertLevel.CRITICAL.value
        if score >= self._threshold_warning:
            return AlertLevel.WARNING.value
        return AlertLevel.INFO.value


def _clean_features(df: pd.DataFrame) -> np.ndarray:
    return df[NUMERIC_FEATURES].dropna().values


def _select_output(df: pd.DataFrame) -> pd.DataFrame:
    if "ground_truth" not in df.columns:
        df = df.copy()
        df["ground_truth"] = None
    return df[PREDICT_OUTPUT_COLS].reset_index(drop=True)
