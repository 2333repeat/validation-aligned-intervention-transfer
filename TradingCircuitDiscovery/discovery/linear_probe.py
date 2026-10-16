from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearProbeResult:
    accuracy: float
    coefficients: np.ndarray
    intercept: float
    predictions: np.ndarray


@dataclass
class TemporalProbeResult:
    coefficients: np.ndarray
    intercept: float
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray


def _validate_binary(features, labels) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of rows")
    if len(np.unique(y)) != 2:
        raise ValueError("linear probe expects binary labels")
    return x, y


def _metrics(y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    auroc = float("nan")
    if len(np.unique(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, probabilities))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "auroc": auroc,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "prevalence": float(np.mean(y_true)),
    }


def _fit_model(x: np.ndarray, y: np.ndarray, random_state: int):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
    )
    model.fit(x, y)
    return model


def train_linear_probe(x, y, test_fraction: float = 0.2, random_state: int = 1) -> LinearProbeResult:
    features, labels = _validate_binary(x, y)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_fraction,
        random_state=random_state,
        stratify=labels,
    )
    model = _fit_model(x_train, y_train, random_state)
    predictions = model.predict(x_test)
    classifier = model[-1]
    return LinearProbeResult(
        accuracy=float(balanced_accuracy_score(y_test, predictions)),
        coefficients=classifier.coef_.copy(),
        intercept=float(classifier.intercept_[0]),
        predictions=predictions,
    )


def train_temporal_probe(x, y, partitions, random_state: int = 1) -> TemporalProbeResult:
    features, labels = _validate_binary(x, y)
    split = np.asarray(partitions).astype(str)
    if len(split) != len(labels):
        raise ValueError("partitions must have one value per sample")
    fit_indices = np.flatnonzero(split == "discovery")
    validation_indices = np.flatnonzero(split == "validation")
    test_indices = np.flatnonzero(split == "test")
    if min(len(fit_indices), len(validation_indices), len(test_indices)) == 0:
        raise ValueError("discovery, validation and test partitions are all required")
    if len(np.unique(labels[fit_indices])) != 2:
        raise ValueError("discovery partition must contain both classes")

    model = _fit_model(features[fit_indices], labels[fit_indices], random_state)
    validation_predictions = model.predict(features[validation_indices])
    validation_probabilities = model.predict_proba(features[validation_indices])[:, 1]
    test_predictions = model.predict(features[test_indices])
    test_probabilities = model.predict_proba(features[test_indices])[:, 1]
    classifier = model[-1]
    scaler = model[0]
    return TemporalProbeResult(
        coefficients=classifier.coef_.copy(),
        intercept=float(classifier.intercept_[0]),
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        validation_metrics=_metrics(labels[validation_indices], validation_predictions, validation_probabilities),
        test_metrics=_metrics(labels[test_indices], test_predictions, test_probabilities),
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        feature_mean=scaler.mean_.copy(),
        feature_scale=scaler.scale_.copy(),
    )
