"""
01_preprocess_and_select_features.py

Preprocess the student specialization dataset, engineer course features, split the
data, and save the fitted artifacts needed by the training and UI scripts.

This version supports the expanded per-course schema:
- grade
- points
- termWork
- examWork
- result
- registered (engineered)
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ======================================================================
# CONFIGURATION
# ======================================================================
INPUT_CANDIDATES = [
    Path("data/students_pre_specialization.csv"),
    Path("data/students_sample.csv"),
    Path("students_pre_specialization.csv"),
    Path("students_sample.csv"),
]

OUTPUT_DIR = Path("processed")
ARTIFACT_DIR = Path("artifacts")

TARGET_COL = "program"
DROP_COLS = ["id", "min_gpa"]  # keep gpa because it exists in the JSON sample

TEST_RATIO = 0.20
VAL_RATIO = 0.15
RANDOM_STATE = 42

VAR_THRESHOLD = 0.001
N_ET_ESTIMATORS = 300
MAX_FEATURES = 120
MIN_FEATURES = 40

COURSE_SUFFIXES = ("grade", "points", "termWork", "examWork", "result")

GRADE_ORDER = {
    "Not_Registered": 0,
    "F": 1,
    "Abs": 2,
    "Con": 3,
    "D": 4,
    "D+": 5,
    "C": 6,
    "C+": 7,
    "P": 8,
    "B": 9,
    "B+": 10,
    "A": 11,
    "A+": 12,
}

GRADE_TO_POINTS = {
    "Not_Registered": 0.0,
    "F": 0.0,
    "Abs": 0.0,
    "Con": 0.0,
    "D": 2.0,
    "D+": 2.2,
    "C": 2.4,
    "C+": 2.7,
    "P": 0.0,
    "B": 3.0,
    "B+": 3.3,
    "A": 3.7,
    "A+": 4.0,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def resolve_input_file() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the input CSV. Checked: "
        + ", ".join(str(p) for p in INPUT_CANDIDATES)
    )


def load_and_clean_data(path: Path) -> pd.DataFrame:
    log.info("Loading %s", path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _course_prefixes_from_columns(columns: Iterable[str]) -> list[str]:
    prefixes = set()
    for col in columns:
        for suffix in COURSE_SUFFIXES:
            marker = f"_{suffix}"
            if col.endswith(marker):
                prefixes.add(col[: -len(marker)])
                break
    return sorted(prefixes)


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def engineer_course_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Ensure every course has the full feature set:
    grade, points, termWork, examWork, result, registered.
    """
    prefixes = _course_prefixes_from_columns(df.columns)

    for prefix in prefixes:
        grade_col = f"{prefix}_grade"
        points_col = f"{prefix}_points"
        term_col = f"{prefix}_termWork"
        exam_col = f"{prefix}_examWork"
        result_col = f"{prefix}_result"
        reg_col = f"{prefix}_registered"

        # Create missing columns so old files can still run through this pipeline.
        if grade_col not in df.columns:
            df[grade_col] = np.nan
        if points_col not in df.columns:
            df[points_col] = np.nan
        if term_col not in df.columns:
            df[term_col] = np.nan
        if exam_col not in df.columns:
            df[exam_col] = np.nan
        if result_col not in df.columns:
            df[result_col] = np.nan

        # Registered flag is based on the original presence of any course value.
        original_presence = pd.concat(
            [
                df[grade_col].notna(),
                df[points_col].notna(),
                df[term_col].notna(),
                df[exam_col].notna(),
                df[result_col].notna(),
            ],
            axis=1,
        ).any(axis=1)
        df[reg_col] = original_presence.astype(np.int8)

        # Normalize grade text.
        df[grade_col] = (
            df[grade_col]
            .fillna("Not_Registered")
            .astype(str)
            .str.strip()
            .replace({"nan": "Not_Registered", "None": "Not_Registered", "": "Not_Registered"})
        )

        # Fill points from grade when missing.
        df[points_col] = _safe_numeric(df[points_col], default=np.nan)
        missing_points = df[points_col].isna()
        if missing_points.any():
            df.loc[missing_points, points_col] = df.loc[missing_points, grade_col].map(GRADE_TO_POINTS)

        # Fill term/exam/result numerics.
        df[term_col] = _safe_numeric(df[term_col], default=0.0)
        df[exam_col] = _safe_numeric(df[exam_col], default=0.0)
        df[result_col] = _safe_numeric(df[result_col], default=np.nan)
        missing_result = df[result_col].isna()
        if missing_result.any():
            df.loc[missing_result, result_col] = df.loc[missing_result, term_col] + df.loc[missing_result, exam_col]
        df[result_col] = df[result_col].fillna(0.0)

    return df, prefixes


def handle_missing_values(df: pd.DataFrame, grade_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    # Keep grades as text until encoding.
    df[grade_cols] = df[grade_cols].fillna("Not_Registered")

    # Force all numeric columns to be numeric.
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Keep GPA if available because it exists in the source JSON and usually helps.
    if "gpa" in df.columns:
        df["gpa"] = pd.to_numeric(df["gpa"], errors="coerce")
        if df["gpa"].isna().any():
            df["gpa"] = df["gpa"].fillna(df["gpa"].median())

    return df


def encode_grades(df: pd.DataFrame, grade_cols: list[str]) -> pd.DataFrame:
    for col in grade_cols:
        df[col] = df[col].map(lambda g: GRADE_ORDER.get(str(g).strip(), 0)).astype(np.int16)
    return df


def split_data(X: pd.DataFrame, y: pd.Series):
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    val_fraction = VAL_RATIO / (1.0 - TEST_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=val_fraction,
        stratify=y_tmp,
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def remove_low_variance(X_train, X_val, X_test):
    selector = VarianceThreshold(threshold=VAR_THRESHOLD)
    selector.fit(X_train)
    kept = X_train.columns[selector.get_support()].tolist()
    if not kept:
        raise RuntimeError("VarianceThreshold removed every feature. Check the input data.")
    return X_train[kept], X_val[kept], X_test[kept], selector


def select_by_importance(X_train, X_val, X_test, y_train):
    et = ExtraTreesClassifier(
        n_estimators=N_ET_ESTIMATORS,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    et.fit(X_train, y_train)
    importances = et.feature_importances_

    if len(importances) == 0:
        raise RuntimeError("No features available for importance selection.")

    above_mean = np.where(importances >= importances.mean())[0]
    ordered_idx = above_mean[np.argsort(importances[above_mean])[::-1]]

    # Fallback to the full ranking if the mean-based filter is too aggressive.
    if len(ordered_idx) < MIN_FEATURES:
        ordered_idx = np.argsort(importances)[::-1]

    selected_idx = ordered_idx[: min(MAX_FEATURES, len(ordered_idx))]
    selected = X_train.columns[selected_idx].tolist()
    return X_train[selected], X_val[selected], X_test[selected], et, selected


def save_datasets(X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        X.to_csv(OUTPUT_DIR / f"{name}_features.csv", index=False)
        y.to_frame(name=TARGET_COL).to_csv(OUTPUT_DIR / f"{name}_labels.csv", index=False)


def save_artifacts(label_enc, var_selector, et_model, feature_names) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "label_encoder.pkl": label_enc,
        "variance_selector.pkl": var_selector,
        "et_feature_model.pkl": et_model,
        "selected_features.pkl": feature_names,
    }
    for fname, obj in artifacts.items():
        joblib.dump(obj, ARTIFACT_DIR / fname)


def main():
    log.info("Starting preprocessing pipeline...")
    input_file = resolve_input_file()
    df = load_and_clean_data(input_file)

    df, course_prefixes = engineer_course_features(df)
    grade_cols = [f"{p}_grade" for p in course_prefixes]
    numeric_cols = []
    for p in course_prefixes:
        numeric_cols.extend(
            [f"{p}_points", f"{p}_termWork", f"{p}_examWork", f"{p}_result", f"{p}_registered"]
        )

    df = handle_missing_values(df, grade_cols, numeric_cols)
    df = encode_grades(df, grade_cols)

    # Remove fields that are not meant to be learned directly.
    df = df.drop(columns=DROP_COLS, errors="ignore")

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' was not found in the input data.")

    # Extract target and encode labels.
    y_raw = df.pop(TARGET_COL)
    label_enc = LabelEncoder()
    y = pd.Series(label_enc.fit_transform(y_raw), index=y_raw.index, name=TARGET_COL)

    # Drop any remaining non-numeric columns to keep the model input stable.
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        log.warning("Dropping unresolved text columns: %s", object_cols)
        df = df.drop(columns=object_cols)

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    log.info("Detected %d course prefixes and %d total feature columns.", len(course_prefixes), df.shape[1])

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, y)
    X_train, X_val, X_test, var_sel = remove_low_variance(X_train, X_val, X_test)
    X_train, X_val, X_test, et_model, selected_feats = select_by_importance(X_train, X_val, X_test, y_train)

    save_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
    save_artifacts(label_enc, var_sel, et_model, selected_feats)

    log.info("Preprocessing complete.")
    log.info("Saved %d selected features.", len(selected_feats))


if __name__ == "__main__":
    main()
