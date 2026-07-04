"""Module for training and evaluating machine learning models for stress detection.

This module provides functions to train and evaluate RandomForest, SVM, and KNN
classifiers on the physiological features dataset. It uses chronological splitting
and TimeSeriesSplit cross-validation to prevent data leakage from temporal overlap.
"""

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def evaluate_model_cv(
    model, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5
) -> list[float]:
    """Perform TimeSeriesSplit cross-validation on a model.

    Parameters
    ----------
    model : estimator
        Scikit-learn estimator.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    cv_splits : int, default 5
        Number of splits for cross-validation.

    Returns
    -------
    list[float]
        Accuracy scores for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train.to_numpy().ravel())
        score = model.score(X_test, y_test.to_numpy().ravel())
        scores.append(score)

    return scores


def train_and_evaluate_all(
    features: pd.DataFrame, labels: pd.DataFrame, test_size_ratio: float = 0.33
) -> dict[str, object]:
    """Train and evaluate RF, KNN, and SVM models using a chronological split.

    Parameters
    ----------
    features : pd.DataFrame
        The feature matrix (first 48 columns).
    labels : pd.DataFrame
        The target label (stress column).
    test_size_ratio : float, default 0.33
        The ratio of data to use for chronological testing.

    Returns
    -------
    dict[str, object]
        Dictionary of trained models: {'RF': rf_model, 'KNN': knn_model, 'SVM': svm_model}.
    """
    total_len = len(features)
    split_idx = int(round(total_len * (1 - test_size_ratio)))

    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = labels.iloc[:split_idx], labels.iloc[split_idx:]

    y_train_flat = y_train.to_numpy().ravel()
    y_test_flat = y_test.to_numpy().ravel()

    models = {
        "RF": RandomForestClassifier(n_estimators=150, max_depth=15, class_weight="balanced", random_state=30),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(class_weight="balanced", random_state=30),
    }

    trained_models = {}

    for name, model in models.items():
        print(f"=== Evaluating {name} ===")
        cv_scores = evaluate_model_cv(model, features, labels)
        print(f"TimeSeriesSplit CV Accuracy: {sum(cv_scores)/len(cv_scores):.4f} {cv_scores}")

        model.fit(X_train, y_train_flat)
        y_pred = model.predict(X_test)

        print("Chronological Test Set Classification Report:")
        print(classification_report(y_test_flat, y_pred, zero_division=0))
        print("\n")

        trained_models[name] = model

    return trained_models
