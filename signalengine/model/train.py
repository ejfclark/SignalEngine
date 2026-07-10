"""LightGBM training over the labeled feature panel.

Output of training is twofold:
  1. Out-of-sample predictions for every walk-forward test block — the ONLY
     predictions the backtest is allowed to see.
  2. A final model fitted on all labeled data — used for live signal generation.

Metrics to care about, in order: AUC vs 0.5 (any edge at all?), then
precision at the trading threshold vs the base rate (is the top of the
probability distribution actually enriched?). Accuracy is meaningless here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from ..config import Config
from ..features.pipeline import FEATURE_COLUMNS
from .splits import purged_walk_forward

MODEL_FILE = "model.joblib"
PREDICTIONS_FILE = "oos_predictions.parquet"
IMPORTANCE_FILE = "feature_importance.csv"


class CalibratedModel:
    """LightGBM + isotonic mapping, exposing the same predict_proba/
    feature_importances_ surface the rest of the engine expects."""

    def __init__(self, model, iso):
        self.model = model
        self.iso = iso

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


@dataclass
class TrainResult:
    fold_metrics: pd.DataFrame
    oos: pd.DataFrame  # meta + label + predicted probability, out-of-sample only
    model: object  # LGBMClassifier or CalibratedModel
    importance: pd.DataFrame = field(repr=False, default=None)


def _make_model(cfg: Config, y_train: np.ndarray) -> LGBMClassifier:
    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    return LGBMClassifier(
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        num_leaves=cfg.model.num_leaves,
        min_child_samples=cfg.model.min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        random_state=42,
        verbosity=-1,
    )


def train_walk_forward(labeled: pd.DataFrame, cfg: Config) -> TrainResult:
    data = labeled.dropna(subset=["label"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No labeled rows — is the price history long enough for the horizon?")

    X = data[FEATURE_COLUMNS]
    y = data["label"].to_numpy()

    fold_rows = []
    oos_parts = []
    threshold = cfg.backtest.probability_threshold

    def fit_predict(train_mask: np.ndarray, predict_X) -> tuple:
        """Fit (optionally with out-of-time isotonic calibration) and score."""
        if not cfg.model.calibrate:
            model = _make_model(cfg, y[train_mask])
            model.fit(X[train_mask], y[train_mask])
            return model, None, model.predict_proba(predict_X)[:, 1]

        from sklearn.isotonic import IsotonicRegression

        # Chronological split of the train window: model on the head,
        # calibration on the tail the model never saw.
        train_idx = np.flatnonzero(train_mask)
        order = np.argsort(data.loc[train_idx, "date"].to_numpy())
        cut = int(len(train_idx) * (1.0 - cfg.model.calibration_frac))
        head, tail = train_idx[order[:cut]], train_idx[order[cut:]]

        model = _make_model(cfg, y[head])
        model.fit(X.iloc[head], y[head])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(model.predict_proba(X.iloc[tail])[:, 1], y[tail])
        return model, iso, iso.predict(model.predict_proba(predict_X)[:, 1])

    for fold, (train_mask, test_mask) in enumerate(
        purged_walk_forward(data["date"], cfg.cv.n_folds, cfg.cv.purge_days)
    ):
        if train_mask.sum() < 500 or test_mask.sum() < 100:
            continue
        _, _, prob = fit_predict(train_mask, X[test_mask])

        y_test = y[test_mask]
        flagged = prob >= threshold
        fold_rows.append({
            "fold": fold,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "test_start": data.loc[test_mask, "date"].min().date(),
            "test_end": data.loc[test_mask, "date"].max().date(),
            "base_rate": float(y_test.mean()),
            "auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else np.nan,
            "flagged": int(flagged.sum()),
            f"precision@{threshold:.2f}": float(y_test[flagged].mean()) if flagged.any() else np.nan,
        })

        # Regime columns ride along so the backtest can test entry gates.
        regime_cols = [c for c in ("breadth_20d", "vix", "btc_ret_20d") if c in data.columns]
        part = data.loc[test_mask, ["ticker", "date", "label", "outcome", "entry_price",
                                    "stop_price", "target_price", "exit_price",
                                    "exit_date", "trade_return", "close", *regime_cols]].copy()
        part["probability"] = prob
        part["fold"] = fold
        oos_parts.append(part)

    if not oos_parts:
        raise ValueError("No usable folds — not enough data for the configured n_folds.")

    all_mask = np.ones(len(data), dtype=bool)
    fitted, iso, _ = fit_predict(all_mask, X.iloc[:1])
    final_model = CalibratedModel(fitted, iso) if iso is not None else fitted

    importance = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": final_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return TrainResult(
        fold_metrics=pd.DataFrame(fold_rows),
        oos=pd.concat(oos_parts, ignore_index=True),
        model=final_model,
        importance=importance,
    )


def save_artifacts(result: TrainResult, directory: Path, asset: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, directory / f"{asset}_{MODEL_FILE}")
    result.oos.to_parquet(directory / f"{asset}_{PREDICTIONS_FILE}", index=False)
    result.importance.to_csv(directory / f"{asset}_{IMPORTANCE_FILE}", index=False)


def load_model(directory: Path, asset: str) -> LGBMClassifier:
    return joblib.load(directory / f"{asset}_{MODEL_FILE}")
