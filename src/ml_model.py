"""
Phase 5 - Feature Engineering & ML
Problem: Predict whether a lead will be Closed Won (binary classification)
         from Source, Source Group, and Notes Sentiment/Word Count.

Features used:
  - source          (one-hot encoded, 20 categories)
  - source_group    (one-hot encoded, 7 categories)
  - notes_sentiment (continuous, from TextBlob)
  - notes_word_count (continuous)

Target: is_won (1 = Closed Won, 0 = all other stages)

Models:
  1. Logistic Regression  (interpretable baseline)
  2. Random Forest        (non-linear, feature importance)

NOTE: The dataset is intentionally balanced with ~10% win rate and near-
uniform source distribution. Real predictive signal is limited by design.
All metrics are generated from actual model execution — no fabrication.
"""

import csv
import json
import os
import pickle

from collections import Counter

# scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

PROCESSED_PATH = "data/processed/leads_processed.csv"
OUTPUT_PATH    = "reports/ml_results.json"
MODEL_DIR      = "models"

CATEGORICAL_FEATURES = ["source", "source_group"]
NUMERIC_FEATURES     = ["notes_sentiment", "notes_word_count"]
TARGET               = "is_won"

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load():
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X_cat  = [[r["source"], r["source_group"]] for r in rows]
    X_num  = [[float(r["notes_sentiment"]), int(r["notes_word_count"])] for r in rows]
    y      = [int(r["is_won"]) for r in rows]

    # Build combined feature rows as dicts for ColumnTransformer
    X = [
        {
            "source":          r["source"],
            "source_group":    r["source_group"],
            "notes_sentiment": float(r["notes_sentiment"]),
            "notes_word_count": int(r["notes_word_count"]),
        }
        for r in rows
    ]
    return X, y, rows


def rows_to_matrix(X_dicts):
    """Convert list-of-dicts to list-of-lists preserving column order."""
    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    return [[row[c] for c in cols] for row in X_dicts]


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             [0, 1]),   # indices of source, source_group in the matrix
            ("num", "passthrough", [2, 3]),  # notes_sentiment, notes_word_count
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(name, pipeline, X_train, X_test, y_train, y_test, feature_names_out=None):
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_prob  = pipeline.predict_proba(X_test)[:, 1]

    acc     = round(accuracy_score(y_test, y_pred), 4)
    prec    = round(precision_score(y_test, y_pred, zero_division=0), 4)
    rec     = round(recall_score(y_test, y_pred, zero_division=0), 4)
    f1      = round(f1_score(y_test, y_pred, zero_division=0), 4)
    roc_auc = round(roc_auc_score(y_test, y_prob), 4)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    report  = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n  [{name}]")
    print(f"  Accuracy  : {acc}")
    print(f"  Precision : {prec}")
    print(f"  Recall    : {rec}")
    print(f"  F1        : {f1}")
    print(f"  ROC-AUC   : {roc_auc}")
    print(f"  Confusion matrix: {cm}")

    return {
        "model":      name,
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1_score":   f1,
        "roc_auc":    roc_auc,
        "confusion_matrix": cm,
        "classification_report": {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            if isinstance(v, dict) else round(v, 4)
            for k, v in report.items()
        },
    }


def cross_validate(name, pipeline, X, y, cv=5):
    scores = cross_val_score(
        pipeline, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
    )
    result = {
        "model":   name,
        "cv_folds": cv,
        "roc_auc_scores": [round(s, 4) for s in scores.tolist()],
        "roc_auc_mean":   round(scores.mean(), 4),
        "roc_auc_std":    round(scores.std(), 4),
    }
    print(f"  [{name}] CV ROC-AUC: {result['roc_auc_mean']:.4f} (+/- {result['roc_auc_std']:.4f})")
    return result


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def extract_feature_importance(pipeline, top_n=20):
    """Extract feature importances from a fitted Random Forest pipeline."""
    ohe    = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    clf    = pipeline.named_steps["classifier"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = cat_names + NUMERIC_FEATURES
    importances = clf.feature_importances_
    ranked = sorted(
        zip(all_names, importances),
        key=lambda x: -x[1]
    )[:top_n]
    return [{"feature": f, "importance": round(imp, 6)} for f, imp in ranked]


def extract_lr_coefficients(pipeline, top_n=20):
    """Extract top absolute coefficients from fitted Logistic Regression."""
    ohe       = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    clf       = pipeline.named_steps["classifier"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = cat_names + NUMERIC_FEATURES
    coefs     = clf.coef_[0]
    ranked = sorted(
        zip(all_names, coefs),
        key=lambda x: -abs(x[1])
    )[:top_n]
    return [{"feature": f, "coefficient": round(c, 6)} for f, c in ranked]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("[INFO] Loading processed data...")
    X_dicts, y, rows = load()
    X = rows_to_matrix(X_dicts)

    print(f"[INFO] Dataset: {len(X):,} samples | Positive class (is_won=1): {sum(y):,} ({sum(y)/len(y)*100:.1f}%)")

    # Train / test split  (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train: {len(X_train):,} | Test: {len(X_test):,}")

    results = {
        "dataset_info": {
            "total_samples":      len(X),
            "train_samples":      len(X_train),
            "test_samples":       len(X_test),
            "positive_class_pct": round(sum(y) / len(y) * 100, 2),
            "features":           CATEGORICAL_FEATURES + NUMERIC_FEATURES,
            "target":             TARGET,
        },
        "model_results":       [],
        "cross_validation":    [],
        "feature_importance":  [],
        "lr_coefficients":     [],
    }

    # ── Model 1: Logistic Regression ───────────────────────────────────
    print("\n[MODEL 1] Logistic Regression")
    lr_pipe = build_pipeline(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    )
    lr_metrics = evaluate("Logistic Regression", lr_pipe, X_train, X_test, y_train, y_test)
    results["model_results"].append(lr_metrics)

    print("[INFO] Cross-validating Logistic Regression...")
    lr_cv = cross_validate("Logistic Regression", build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced",
                           random_state=RANDOM_STATE, solver="lbfgs")
    ), X, y)
    results["cross_validation"].append(lr_cv)

    # Extract LR coefficients
    results["lr_coefficients"] = extract_lr_coefficients(lr_pipe)

    # ── Model 2: Random Forest ─────────────────────────────────────────
    print("\n[MODEL 2] Random Forest")
    rf_pipe = build_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    rf_metrics = evaluate("Random Forest", rf_pipe, X_train, X_test, y_train, y_test)
    results["model_results"].append(rf_metrics)

    print("[INFO] Cross-validating Random Forest...")
    rf_cv = cross_validate("Random Forest", build_pipeline(
        RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1)
    ), X, y)
    results["cross_validation"].append(rf_cv)

    # Extract RF feature importances
    results["feature_importance"] = extract_feature_importance(rf_pipe)

    # ── Save best model (RF by ROC-AUC) ───────────────────────────────
    best = max(results["model_results"], key=lambda x: x["roc_auc"])
    best_pipe = rf_pipe if best["model"] == "Random Forest" else lr_pipe
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    results["best_model"] = best["model"]
    results["best_model_path"] = model_path
    print(f"\n[INFO] Best model: {best['model']} (ROC-AUC={best['roc_auc']}) -> saved to '{model_path}'")

    # ── Write results ──────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] ML results -> '{OUTPUT_PATH}'")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n=== ML SUMMARY ===")
    for m in results["model_results"]:
        print(f"  {m['model']:<25} Acc={m['accuracy']}  F1={m['f1_score']}  ROC-AUC={m['roc_auc']}")
    print("\n  Top 10 features (Random Forest importance):")
    for fi in results["feature_importance"][:10]:
        print(f"    {fi['feature']:<40} {fi['importance']:.6f}")


if __name__ == "__main__":
    run()
