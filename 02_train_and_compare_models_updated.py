"""
02_train_and_compare_models.py

Train and compare several calibrated tree-based models on the processed data.
The best model is selected using validation performance and then retrained on
train + validation for the final artifact.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

warnings.filterwarnings("ignore")

# ======================================================================
# CONFIGURATION
# ======================================================================
PROCESSED_DIR = Path("processed")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")

RANDOM_STATE = 42
CV_FOLDS = 4
N_ITER_SEARCH = 3
PRIMARY_SEARCH_SCORING = "f1_macro"

FINAL_MODEL_NAME = "Best_Calibrated_Model.pkl"
MODEL_INFO_NAME = "best_model_info.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    name: str
    best_estimator: object
    best_params: dict
    validation_metrics: dict
    search_time_sec: float


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    X_val = pd.read_csv(PROCESSED_DIR / "val_features.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "test_features.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "train_labels.csv").squeeze("columns")
    y_val = pd.read_csv(PROCESSED_DIR / "val_labels.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "test_labels.csv").squeeze("columns")

    label_enc = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
    return X_train, X_val, X_test, y_train, y_val, y_test, label_enc


def _min_class_count(y: pd.Series) -> int:
    counts = y.value_counts()
    return int(counts.min()) if not counts.empty else 2


def make_cv(y: pd.Series) -> StratifiedKFold:
    n_splits = max(2, min(CV_FOLDS, _min_class_count(y)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def build_candidate_searches():
    return [
        (
            "RandomForest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [300, 500, 800],
                "max_depth": [None, 12, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.5, 0.7],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [400, 600, 900],
                "max_depth": [None, 12, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.5, 0.7],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
        ),
    ]


def tune_model(name: str, estimator, param_grid: dict, X_train, y_train):
    cv = make_cv(y_train)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        scoring=PRIMARY_SEARCH_SCORING,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )

    log.info("Tuning %s ...", name)
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    log.info(
        "%s tuning done in %.1fs | best CV %s = %.4f",
        name,
        elapsed,
        PRIMARY_SEARCH_SCORING,
        search.best_score_,
    )
    return search.best_estimator_, search.best_params_, elapsed


def evaluate_metrics(model, X, y, label_enc, split_name: str) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, y_prob, labels=model.classes_)),
    }
    try:
        metrics["top2"] = float(top_k_accuracy_score(y, y_prob, k=2, labels=model.classes_))
    except Exception:
        metrics["top2"] = float("nan")
    try:
        metrics["top3"] = float(top_k_accuracy_score(y, y_prob, k=3, labels=model.classes_))
    except Exception:
        metrics["top3"] = float("nan")

    log.info(
        "%s | Acc=%.4f  MacroF1=%.4f  Top-2=%.4f  Top-3=%.4f  LogLoss=%.4f",
        split_name,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["top2"],
        metrics["top3"],
        metrics["log_loss"],
    )

    if split_name.lower() == "test":
        log.info("\n%s", classification_report(y, y_pred, target_names=label_enc.classes_, zero_division=0))

    return metrics


def calibration_folds(y_train: pd.Series) -> int:
    return max(2, min(CV_FOLDS, _min_class_count(y_train)))


def calibrate_model(best_estimator, X, y):
    cv_folds = calibration_folds(y)
    log.info("Calibrating probabilities with %d-fold sigmoid calibration...", cv_folds)
    calibrated = CalibratedClassifierCV(estimator=best_estimator, method="sigmoid", cv=cv_folds)
    calibrated.fit(X, y)
    return calibrated


def compare_candidates(X_train, X_val, y_train, y_val, label_enc):
    results: list[CandidateResult] = []

    for name, estimator, grid in build_candidate_searches():
        best_estimator, best_params, elapsed = tune_model(name, estimator, grid, X_train, y_train)
        validation_metrics = evaluate_metrics(best_estimator, X_val, y_val, label_enc, f"{name} (val)")
        results.append(
            CandidateResult(
                name=name,
                best_estimator=best_estimator,
                best_params=best_params,
                validation_metrics=validation_metrics,
                search_time_sec=elapsed,
            )
        )

    # Prefer higher macro F1, then higher accuracy, then lower log loss.
    results.sort(
        key=lambda r: (
            r.validation_metrics["macro_f1"],
            r.validation_metrics["accuracy"],
            -r.validation_metrics["log_loss"],
        ),
        reverse=True,
    )
    return results


def save_model_info(info: dict):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / MODEL_INFO_NAME, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, label_enc = load_processed_data()

    log.info("Loaded processed data: train=%s, val=%s, test=%s", X_train.shape, X_val.shape, X_test.shape)

    ranked = compare_candidates(X_train, X_val, y_train, y_val, label_enc)
    best = ranked[0]

    log.info("Best candidate on validation: %s", best.name)
    log.info("Best params: %s", best.best_params)

    # Retrain the selected model on train + validation, then calibrate on the combined set.
    X_train_full = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], axis=0, ignore_index=True)

    final_estimator = clone(best.best_estimator)
    final_calibrated = calibrate_model(final_estimator, X_train_full, y_train_full)

    test_metrics = evaluate_metrics(final_calibrated, X_test, y_test, label_enc, "Test")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / FINAL_MODEL_NAME
    joblib.dump(final_calibrated, model_path)
    log.info("Final model saved to %s", model_path)

    info = {
        "best_model_name": best.name,
        "best_params": best.best_params,
        "validation_metrics": best.validation_metrics,
        "test_metrics": test_metrics,
        "selected_features_count": int(X_train.shape[1]),
        "model_path": str(model_path),
    }
    save_model_info(info)
    log.info("Saved model metadata to artifacts/%s", MODEL_INFO_NAME)


if __name__ == "__main__":
    main()
