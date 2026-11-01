"""Failure-risk baseline for Tawseel COD parcels.

You write this file. Keep every name and signature exactly as it is: the grader imports them.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "failed"

# The raw columns of make_parcels() that the model may use at decision time (when the parcel
# is created). Nothing known only after the delivery attempt, not the label, not the row id.
# TODO: this is every column of the table. Keep only the ones known at decision time.
FEATURE_COLUMNS: list[str] = ["customer_id", "merchant_id", "area", "promised_slot",
                              "cod_amount", "n_items", "address_text",]

NUMERIC_COLUMNS = ["cod_amount", "n_items",
                   "customer_prior_parcels", "customer_prior_fail_rate",]

CATEGORICAL_COLUMNS = ["customer_id", "merchant_id",
                       "area", "promised_slot", "address_text",]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    history = df[["customer_id", "created_at", "failed"]].copy()
    history["_time_bucket"] = history["created_at"].dt.floor("min")

    grouped = history.groupby(["customer_id", "_time_bucket"], dropna=False, sort=False)[
        "failed"].agg(group_count="size", group_failures="sum").reset_index()
    grouped = grouped.sort_values(
        ["customer_id", "_time_bucket"], kind="mergesort")

    by_customer = grouped.groupby("customer_id", dropna=False, sort=False)

    grouped["customer_prior_parcels"] = by_customer["group_count"].cumsum() - \
        grouped["group_count"]
    grouped["customer_prior_failures"] = by_customer["group_failures"].cumsum(
    ) - grouped["group_failures"]
    grouped["customer_prior_fail_rate"] = grouped["customer_prior_failures"] / \
        grouped["customer_prior_parcels"]

    lookup = grouped.set_index(["customer_id", "_time_bucket"])[
        ["customer_prior_parcels", "customer_prior_fail_rate"]]
    row_keys = pd.MultiIndex.from_arrays(
        [df["customer_id"], df["created_at"].dt.floor("min")])
    mapped = lookup.reindex(row_keys)

    out["customer_prior_parcels"] = mapped["customer_prior_parcels"].to_numpy(
        dtype=np.int64)
    out["customer_prior_fail_rate"] = mapped["customer_prior_fail_rate"].to_numpy(
        dtype=float)

    return out


def time_split(df: pd.DataFrame, time_col: str = "created_at", test_fraction: float = 0.2,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically without separating equal timestamps."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    if len(df) == 0:
        return df.copy(), df.copy()

    n_test = math.ceil(len(df) * test_fraction)

    sorted_times = df[time_col].sort_values(kind="mergesort")
    boundary = sorted_times.iloc[-n_test]

    train = df.loc[df[time_col] < boundary].copy()
    test = df.loc[df[time_col] >= boundary].copy()

    return train, test


def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an unfitted preprocessing and Logistic Regression pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True,),),
            ("scaler", StandardScaler()),])

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__",),),
            ("onehot", OneHotEncoder(handle_unknown="ignore"),),])

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS,),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS,),],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs",),),])

    return model


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Evaluate the model using classification and probability metrics."""

    probabilities = model.predict_proba(test_df)[:, 1]
    y_true = test_df[TARGET]

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
        "base_rate": float(y_true.mean()),
        "n": int(len(test_df)),
    }


def compare_to_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 0,
) -> dict:
    """Compare Logistic Regression against a prior-probability baseline."""

    model = build_pipeline(seed)

    # Fit both models using training data only.
    model.fit(train_df, train_df[TARGET])

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_df, train_df[TARGET])

    model_metrics = evaluate(model, test_df)
    dummy_metrics = evaluate(dummy, test_df)

    return {
        "model": model_metrics,
        "dummy": dummy_metrics,
        "lift_pr_auc": (model_metrics["pr_auc"] - dummy_metrics["pr_auc"]), }
