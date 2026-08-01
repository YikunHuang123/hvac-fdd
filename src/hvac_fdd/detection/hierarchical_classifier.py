"""
Hierarchical XGBoost Classifier backend for FaultClassifier.
Implements a 3-stage cascade architecture:
Stage 1: Normal vs Fault
Stage 2: Sensor Bias vs Actuator Stuck vs Actuator Leakage
Stage 3: Fine-grained fault isolation (e.g. coi_bias vs oa_bias)
"""
from __future__ import annotations

import logging
from pathlib import Path
import joblib

import numpy as np
from xgboost import XGBClassifier

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.exceptions import ModelPersistenceError

logger = logging.getLogger(__name__)


class HierarchicalXGBWrapper:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.classes_ = None
        self.class_to_idx = {}
        
        # We will hold multiple XGBoost models
        self.stage1_model = None
        self.stage2_model = None
        self.stage3_bias_model = None
        self.stage3_stuck_model = None

    def _get_stage2_label(self, label: str) -> str:
        if "bias" in label:
            return "sensor_bias"
        elif "stuck" in label:
            return "actuator_stuck"
        elif "leakage" in label:
            return "actuator_leakage"
        return "unknown"

    def fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray = None) -> "HierarchicalXGBWrapper":
        if classes is not None:
            self.classes_ = classes
            y_str = np.array([classes[idx] for idx in y])
        else:
            self.classes_ = np.unique(y)
            y_str = y
            
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        
        logger.info(f"Hierarchical XGB: Fitting on {len(X)} rows...")
        
        # ── Stage 1: Normal vs Fault ──
        logger.info("Training Stage 1 (Normal vs Fault)...")
        y_s1 = np.where(y_str == "normal", 0, 1)
        self.stage1_model = XGBClassifier(
            n_estimators=self._settings.clf_n_estimators,
            random_state=self._settings.random_state,
            n_jobs=-1,
            eval_metric="logloss"
        )
        self.stage1_model.fit(X, y_s1)
        
        # Isolate fault data for downstream stages
        fault_mask = (y_str != "normal")
        X_fault = X[fault_mask]
        y_fault = y_str[fault_mask]
        
        if len(X_fault) == 0:
            logger.warning("No fault data found. Skipping Stage 2 & 3.")
            return self
            
        # ── Stage 2: Fault Categories ──
        logger.info("Training Stage 2 (Fault Categories)...")
        y_s2_str = np.array([self._get_stage2_label(lbl) for lbl in y_fault])
        
        # map to int
        s2_mapping = {"sensor_bias": 0, "actuator_stuck": 1, "actuator_leakage": 2}
        y_s2 = np.array([s2_mapping[lbl] for lbl in y_s2_str])
        
        self.stage2_model = XGBClassifier(
            n_estimators=self._settings.clf_n_estimators,
            random_state=self._settings.random_state,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
        self.stage2_model.fit(X_fault, y_s2)
        
        # ── Stage 3: Fine-grained models ──
        logger.info("Training Stage 3 (Fine-grained isolation)...")
        
        # Stage 3 - Sensor Bias (coi_bias vs oa_bias)
        mask_bias = (y_s2_str == "sensor_bias")
        if np.any(mask_bias):
            X_bias = X_fault[mask_bias]
            y_bias = y_fault[mask_bias]
            # coi_bias -> 0, oa_bias -> 1
            y_bias_int = np.where(y_bias == "coi_bias", 0, 1)
            self.stage3_bias_model = XGBClassifier(
                n_estimators=self._settings.clf_n_estimators,
                random_state=self._settings.random_state,
                n_jobs=-1,
                eval_metric="logloss"
            )
            self.stage3_bias_model.fit(X_bias, y_bias_int)
            
        # Stage 3 - Actuator Stuck (coi_stuck vs damper_stuck)
        mask_stuck = (y_s2_str == "actuator_stuck")
        if np.any(mask_stuck):
            X_stuck = X_fault[mask_stuck]
            y_stuck = y_fault[mask_stuck]
            # coi_stuck -> 0, damper_stuck -> 1
            y_stuck_int = np.where(y_stuck == "coi_stuck", 0, 1)
            self.stage3_stuck_model = XGBClassifier(
                n_estimators=self._settings.clf_n_estimators,
                random_state=self._settings.random_state,
                n_jobs=-1,
                eval_metric="logloss"
            )
            self.stage3_stuck_model.fit(X_stuck, y_stuck_int)

        logger.info("Hierarchical XGB models fitted successfully.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.stage1_model is None:
            raise RuntimeError("Model is not fitted yet.")
            
        N = len(X)
        probs = np.zeros((N, len(self.classes_)))
        
        # S1: P(Normal), P(Fault)
        p_s1 = self.stage1_model.predict_proba(X)
        p_normal = p_s1[:, 0]
        p_fault = p_s1[:, 1]
        
        # Set normal prob
        if "normal" in self.class_to_idx:
            probs[:, self.class_to_idx["normal"]] = p_normal
            
        if self.stage2_model is not None:
            # S2: P(Cat | Fault)
            p_s2 = self.stage2_model.predict_proba(X)
            # p_s2 cols: 0: sensor_bias, 1: actuator_stuck, 2: actuator_leakage
            
            p_bias = p_fault * p_s2[:, 0]
            p_stuck = p_fault * p_s2[:, 1]
            p_leakage = p_fault * p_s2[:, 2]
            
            # S3: Leakage directly maps to coi_leakage
            if "coi_leakage" in self.class_to_idx:
                probs[:, self.class_to_idx["coi_leakage"]] = p_leakage
                
            # S3: Bias
            if self.stage3_bias_model is not None:
                p_s3_bias = self.stage3_bias_model.predict_proba(X)
                if "coi_bias" in self.class_to_idx:
                    probs[:, self.class_to_idx["coi_bias"]] = p_bias * p_s3_bias[:, 0]
                if "oa_bias" in self.class_to_idx:
                    probs[:, self.class_to_idx["oa_bias"]] = p_bias * p_s3_bias[:, 1]
                    
            # S3: Stuck
            if self.stage3_stuck_model is not None:
                p_s3_stuck = self.stage3_stuck_model.predict_proba(X)
                if "coi_stuck" in self.class_to_idx:
                    probs[:, self.class_to_idx["coi_stuck"]] = p_stuck * p_s3_stuck[:, 0]
                if "damper_stuck" in self.class_to_idx:
                    probs[:, self.class_to_idx["damper_stuck"]] = p_stuck * p_s3_stuck[:, 1]
                    
        # Normalize just in case
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def save(self, path: Path | str) -> None:
        if self.stage1_model is None:
            raise ModelPersistenceError("Cannot save unfitted HierarchicalXGBWrapper")
            
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "classes_": self.classes_,
            "class_to_idx": self.class_to_idx,
            "stage1_model": self.stage1_model,
            "stage2_model": self.stage2_model,
            "stage3_bias_model": self.stage3_bias_model,
            "stage3_stuck_model": self.stage3_stuck_model
        }
        
        try:
            joblib.dump(state, out)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to save to {out}: {exc}") from exc

    @classmethod
    def load(cls, path: Path | str, settings: Settings | None = None) -> "HierarchicalXGBWrapper":
        src = Path(path)
        try:
            state = joblib.load(src)
        except Exception as exc:
            raise ModelPersistenceError(f"Failed to load from {src}: {exc}") from exc
            
        obj = cls(settings=settings)
        obj.classes_ = state["classes_"]
        obj.class_to_idx = state["class_to_idx"]
        obj.stage1_model = state["stage1_model"]
        obj.stage2_model = state["stage2_model"]
        obj.stage3_bias_model = state["stage3_bias_model"]
        obj.stage3_stuck_model = state["stage3_stuck_model"]
        
        return obj
