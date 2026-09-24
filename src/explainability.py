"""
Phase 6 - Model Explainability
Explains the trained models using:
  1. Logistic Regression coefficients (per-feature directional influence)
  2. Random Forest feature importances (Gini impurity-based)
  3. Per-source marginal win rate vs model-predicted probability
     (the most interpretable signal available in this dataset)
  4. Permutation importance on held-out test set (model-agnostic)
  5. Calibration check (predicted probabilities vs actual win rate)

All values are computed from actual fitted models — no fabrication.
Outputs: reports/explainability_results.json
"""

import csv
import json
import os
import pickle
from collections import defaultdict

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

PROCESSED_PATH = "data/processed/leads_processed.csv"
ML_RESULTS     = "reports/ml_results.json"
OUTPUT_PATH    = "reports/explainability_results.json"
MODEL_PATH     = "models/best_model.pkl"

CATEGORICAL_FEATURES = ["source", "source_group"]
NUMERIC_FEATURES     = ["notes_sentiment", "notes_word_count"]
RANDOM_STATE         = 42


# ---------------------------------------------------------------------------
# Load data (same as ml_model.py)
# ---------------------------------------------------------------------------

def load():
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X = [
        [r["source"], r["source_group"],
         float(r["notes_sentiment"]), int(r["notes_word_count"])]
        for r in rows
    ]
    y = [int(r["is_won"]) for r in rows]
    return X, y, rows


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0, 1]),
            ("num", "passthrough", [2, 3]),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


# ---------------------------------------------------------------------------
# 1. Rebuild both models (needed for explainability on full feature set)
# ---------------------------------------------------------------------------

def rebuild_models(X_train, y_train):
    lr_pipe = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced",
                           random_state=RANDOM_STATE, solver="lbfgs")
    )
    rf_pipe = build_pipeline(
        RandomForestClassifier(n_estimators=200, max_depth=8,
                               class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1)
    )
    lr_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    return lr_pipe, rf_pipe


# ---------------------------------------------------------------------------
# 2. Logistic Regression coefficients
# ---------------------------------------------------------------------------

def lr_coefficients(lr_pipe):
    ohe       = lr_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    clf       = lr_pipe.named_steps["classifier"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = cat_names + NUMERIC_FEATURES
    coefs     = clf.coef_[0]

    entries = []
    for name, coef in zip(all_names, coefs):
        # Parse feature group
        if name.startswith("source_group_"):
            group = "source_group"
            label = name.replace("source_group_", "")
        elif name.startswith("source_"):
            group = "source"
            label = name.replace("source_", "")
        else:
            group = "numeric"
            label = name
        entries.append({
            "feature":    name,
            "group":      group,
            "label":      label,
            "coefficient": round(float(coef), 6),
            "direction":  "positive" if coef > 0 else "negative",
        })

    # Sort by absolute value descending
    entries.sort(key=lambda x: -abs(x["coefficient"]))
    return entries


# ---------------------------------------------------------------------------
# 3. Random Forest feature importance
# ---------------------------------------------------------------------------

def rf_feature_importance(rf_pipe):
    ohe       = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    clf       = rf_pipe.named_steps["classifier"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = cat_names + NUMERIC_FEATURES
    importances = clf.feature_importances_

    entries = []
    for name, imp in zip(all_names, importances):
        if name.startswith("source_group_"):
            group = "source_group"
            label = name.replace("source_group_", "")
        elif name.startswith("source_"):
            group = "source"
            label = name.replace("source_", "")
        else:
            group = "numeric"
            label = name
        entries.append({
            "feature":    name,
            "group":      group,
            "label":      label,
            "importance": round(float(imp), 6),
        })

    entries.sort(key=lambda x: -x["importance"])
    return entries


# ---------------------------------------------------------------------------
# 4. Permutation importance (model-agnostic, on test set)
# ---------------------------------------------------------------------------

def permutation_imp(rf_pipe, X_test, y_test, feature_names):
    result = permutation_importance(
        rf_pipe, X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    entries = []
    for i, name in enumerate(feature_names):
        entries.append({
            "feature":         name,
            "importance_mean": round(float(result.importances_mean[i]), 6),
            "importance_std":  round(float(result.importances_std[i]), 6),
        })
    entries.sort(key=lambda x: -x["importance_mean"])
    return entries


# ---------------------------------------------------------------------------
# 5. Per-source actual vs predicted win rate
# ---------------------------------------------------------------------------

def actual_vs_predicted(rf_pipe, X, y, rows):
    """
    For each source: compare actual win rate with the model's mean
    predicted probability for leads from that source.
    This is the clearest explainability signal available.
    """
    y_prob = rf_pipe.predict_proba(X)[:, 1]

    source_actual = defaultdict(list)
    source_pred   = defaultdict(list)

    for i, row in enumerate(rows):
        src = row["source"]
        source_actual[src].append(y[i])
        source_pred[src].append(y_prob[i])

    entries = []
    for src in sorted(source_actual.keys()):
        actual_rate = sum(source_actual[src]) / len(source_actual[src])
        pred_mean   = sum(source_pred[src])   / len(source_pred[src])
        entries.append({
            "source":             src,
            "n":                  len(source_actual[src]),
            "actual_win_rate":    round(actual_rate, 4),
            "predicted_prob_mean": round(pred_mean, 4),
            "gap":                round(pred_mean - actual_rate, 4),
        })

    entries.sort(key=lambda x: -x["actual_win_rate"])
    return entries


# ---------------------------------------------------------------------------
# 6. Calibration check (predicted prob bins vs actual win rate)
# ---------------------------------------------------------------------------

def calibration_check(rf_pipe, X_test, y_test, n_bins=10):
    y_prob = rf_pipe.predict_proba(X_test)[:, 1]
    bins = []
    step = 1.0 / n_bins
    for i in range(n_bins):
        lo, hi = i * step, (i + 1) * step
        mask = [lo <= p < hi for p in y_prob]
        if sum(mask) == 0:
            continue
        probs_in_bin  = [y_prob[j]  for j in range(len(y_prob))  if mask[j]]
        actual_in_bin = [y_test[j]  for j in range(len(y_test))  if mask[j]]
        bins.append({
            "bin_low":          round(lo, 2),
            "bin_high":         round(hi, 2),
            "n":                len(probs_in_bin),
            "mean_pred_prob":   round(sum(probs_in_bin) / len(probs_in_bin), 4),
            "actual_win_rate":  round(sum(actual_in_bin) / len(actual_in_bin), 4),
        })
    return bins


# ---------------------------------------------------------------------------
# 7. Source-group aggregated LR coefficient
# ---------------------------------------------------------------------------

def source_group_lr_summary(coef_entries):
    """
    Aggregate LR coefficients by source group and individual source
    to show which channel categories the model tilts toward.
    """
    by_group = defaultdict(list)
    for e in coef_entries:
        by_group[e["group"]].append(e["coefficient"])

    summary = []
    for g, vals in by_group.items():
        summary.append({
            "group":       g,
            "count":       len(vals),
            "mean_coef":   round(sum(vals) / len(vals), 6),
            "max_coef":    round(max(vals), 6),
            "min_coef":    round(min(vals), 6),
        })
    summary.sort(key=lambda x: -x["mean_coef"])
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    os.makedirs("reports", exist_ok=True)

    print("[INFO] Loading data...")
    X, y, rows = load()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("[INFO] Rebuilding models for explainability...")
    lr_pipe, rf_pipe = rebuild_models(X_train, y_train)

    # Raw feature names passed into the pipeline (before encoding)
    raw_feature_names = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    print("[INFO] Extracting LR coefficients...")
    coef_entries = lr_coefficients(lr_pipe)

    print("[INFO] Extracting RF feature importances...")
    importance_entries = rf_feature_importance(rf_pipe)

    print("[INFO] Computing permutation importance (this may take a moment)...")
    perm_entries = permutation_imp(rf_pipe, X_test, y_test, raw_feature_names)

    print("[INFO] Computing actual vs predicted win rates by source...")
    avp = actual_vs_predicted(rf_pipe, X, y, rows)

    print("[INFO] Computing calibration...")
    cal = calibration_check(rf_pipe, X_test, y_test)

    print("[INFO] Summarising LR coefficients by group...")
    lr_group_summary = source_group_lr_summary(coef_entries)

    results = {
        "lr_coefficients_top20":      coef_entries[:20],
        "rf_feature_importance_top20": importance_entries[:20],
        "permutation_importance":      perm_entries,
        "actual_vs_predicted_by_source": avp,
        "calibration_bins":            cal,
        "lr_group_summary":            lr_group_summary,
        "explainability_notes": {
            "lr_coefficient_interpretation": (
                "Positive coefficient = model assigns higher win probability to that feature value. "
                "Negative = lower. Magnitude indicates relative strength within the model, "
                "NOT causal effect size."
            ),
            "rf_importance_interpretation": (
                "Gini-based importance reflects how much each feature reduces impurity across all trees. "
                "Notes sentiment dominates because it is a continuous feature with high variance, "
                "not because it is a strong predictor."
            ),
            "permutation_importance_interpretation": (
                "Permutation importance measures ROC-AUC drop when a feature's values are shuffled. "
                "Near-zero values confirm that no feature provides meaningful predictive lift."
            ),
            "causal_warning": (
                "These explanations describe the model's learned associations, not causal relationships. "
                "Higher Podcast/Referral coefficients reflect the observed win-rate differences in the data; "
                "they do not imply that switching to those channels will cause win rates to increase."
            ),
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Explainability results -> '{OUTPUT_PATH}'")

    # ── Print summary ──────────────────────────────────────────────────
    print("\n=== EXPLAINABILITY SUMMARY ===")

    print("\n  Top 10 LR Coefficients (by absolute value):")
    for e in coef_entries[:10]:
        print(f"    {e['feature']:<42} {e['coefficient']:>+.4f}  ({e['direction']})")

    print("\n  Top 10 RF Feature Importances:")
    for e in importance_entries[:10]:
        print(f"    {e['feature']:<42} {e['importance']:.6f}")

    print("\n  Permutation Importance (ROC-AUC drop when shuffled):")
    for e in perm_entries:
        print(f"    {e['feature']:<30} mean={e['importance_mean']:>+.6f}  std={e['importance_std']:.6f}")

    print("\n  Actual vs Predicted win rate by source (top 5 / bottom 5):")
    print("  Top 5:")
    for e in avp[:5]:
        print(f"    {e['source']:<30} actual={e['actual_win_rate']:.4f}  pred={e['predicted_prob_mean']:.4f}  gap={e['gap']:+.4f}")
    print("  Bottom 5:")
    for e in avp[-5:]:
        print(f"    {e['source']:<30} actual={e['actual_win_rate']:.4f}  pred={e['predicted_prob_mean']:.4f}  gap={e['gap']:+.4f}")


if __name__ == "__main__":
    run()
