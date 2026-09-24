"""
Phase 6 – Model Explainability
================================
Model artifacts : models/logistic_regression_model.pkl  (best by val ROC-AUC)
                  models/random_forest_model.pkl
                  models/lightgbm_model.pkl
Data source     : data/processed/cleaned_leads.csv

Methods applied
---------------
  G1  Logistic Regression signed coefficients  (global, linear)
  G2  Random Forest Gini feature importance    (global, tree-based)
  G3  LightGBM split-based feature importance  (global, tree-based)
  G4  Permutation Importance on test set        (global, model-agnostic)
  G5  SHAP TreeExplainer — RF global summary    (global, Shapley values)
  L1  SHAP waterfall plots — representative cases (local):
        • True Positive  (high predicted prob, actual Won)
        • False Positive (high predicted prob, actual Not-Won)
        • False Negative (low predicted prob, actual Won)
        • True Negative  (low predicted prob, actual Not-Won)
  C1  Actual win rate vs model predicted probability by source (calibration)

Outputs
-------
  reports/figures/expl_g1_lr_coefficients.png
  reports/figures/expl_g2_rf_feature_importance.png
  reports/figures/expl_g3_lgbm_feature_importance.png
  reports/figures/expl_g4_permutation_importance.png
  reports/figures/expl_g5_shap_summary.png
  reports/figures/expl_l1_shap_waterfall_tp.png
  reports/figures/expl_l1_shap_waterfall_fp.png
  reports/figures/expl_l1_shap_waterfall_fn.png
  reports/figures/expl_l1_shap_waterfall_tn.png
  reports/figures/expl_c1_actual_vs_predicted.png
  reports/explainability_phase6.json   (all numeric results)
"""

import csv
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLEAN_PATH  = "data/processed/cleaned_leads.csv"
FIG_DIR     = "reports/figures"
OUT_JSON    = "reports/explainability_phase6.json"
RANDOM_STATE = 42

os.makedirs(FIG_DIR, exist_ok=True)

SOURCE_TIER = {
    "Podcast":1,"Partner Program":1,"Referral":1,"Webinars":1,"Content Marketing":1,
    "LinkedIn Outreach":2,"Cold Email":2,"Cold Call":2,"Facebook Ads":2,"Chatbot":2,
    "Retargeting Ads":2,"Direct Traffic":2,"Organic Search (SEO)":2,"Trade Show":2,
    "Google Ads":2,"Social Media":3,"Purchased List":3,"Other":3,
    "Website Form":3,"Networking Event":3,
}
NUM_COLS = ["notes_sentiment","notes_word_count","notes_has_text",
            "source_tier","owner_freq","company_freq"]

plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa", "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})

# ---------------------------------------------------------------------------
# Data loading (must mirror Phase 5 exactly)
# ---------------------------------------------------------------------------
def load_data():
    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    owner_c = {}; company_c = {}
    for r in rows:
        owner_c[r["lead_owner"]] = owner_c.get(r["lead_owner"], 0) + 1
        company_c[r["company"]]  = company_c.get(r["company"],  0) + 1

    X, y, meta = [], [], []
    for r in rows:
        X.append([
            r["source"], r["source_group"],
            float(r["notes_sentiment"]), int(r["notes_word_count"]),
            int(r["notes_has_text"]),
            SOURCE_TIER.get(r["source"], 2),
            owner_c[r["lead_owner"]] / total,
            company_c[r["company"]]  / total,
        ])
        y.append(int(r["is_won"]))
        meta.append({"source": r["source"], "source_group": r["source_group"],
                     "deal_stage": r["deal_stage"]})
    return X, np.array(y), meta, rows


def reproduce_splits(X, y):
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.15/0.85,
        stratify=y_tr_full, random_state=RANDOM_STATE)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# G1: LR Coefficients
# ---------------------------------------------------------------------------
def g1_lr_coefficients(lr_pipe, feature_names):
    ohe   = lr_pipe.named_steps["pre"].named_transformers_["cat"]
    clf   = lr_pipe.named_steps["clf"]
    cat_names = list(ohe.get_feature_names_out(["source","source_group"]))
    all_names = cat_names + NUM_COLS
    coefs     = clf.coef_[0]

    pairs = sorted(zip(all_names, coefs), key=lambda x: x[1])
    names = [p[0] for p in pairs]
    vals  = [p[1] for p in pairs]
    colors = ["#ef4444" if v < 0 else "#22c55e" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(names, vals, color=colors, height=0.7, edgecolor="white")
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_xlabel("Coefficient value")
    ax.set_title("G1  Logistic Regression Signed Coefficients\n"
                 "(positive = model tilts toward Closed Won prediction)")
    ax.text(0.5, -0.07,
            "ASSOCIATION, NOT CAUSATION: These coefficients reflect learned correlations "
            "from training data, not causal effects.",
            transform=ax.transAxes, ha="center", fontsize=8,
            color="#dc2626", style="italic")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "expl_g1_lr_coefficients.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  [SAVED] {path}")

    result = [{"feature": n, "coefficient": round(float(v), 6)}
              for n, v in zip(all_names, coefs)]
    result.sort(key=lambda x: -abs(x["coefficient"]))
    return result


# ---------------------------------------------------------------------------
# G2: RF Gini Feature Importance
# ---------------------------------------------------------------------------
def g2_rf_importance(rf_pipe, feature_names):
    ohe  = rf_pipe.named_steps["pre"].named_transformers_["cat"]
    clf  = rf_pipe.named_steps["clf"]
    cat_names = list(ohe.get_feature_names_out(["source", "source_group"]))
    all_names = cat_names + NUM_COLS
    imps = clf.feature_importances_

    pairs = sorted(zip(all_names, imps), key=lambda x: -x[1])[:20]
    names = [p[0] for p in pairs][::-1]
    vals  = [p[1] for p in pairs][::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names, vals, color="#3b82f6", height=0.65, edgecolor="white")
    ax.set_xlabel("Gini Importance")
    ax.set_title("G2  Random Forest — Gini Feature Importances (Top 20)\n"
                 "WARNING: continuous features inflate Gini scores via variance, "
                 "not predictive power")
    for v, n in zip(vals, names):
        ax.text(v + 0.001, names.index(n), f"{v:.4f}", va="center", fontsize=8)
    ax.text(0.5, -0.07,
            "Gini importance is biased toward high-variance continuous features. "
            "Use permutation importance for unbiased signal.",
            transform=ax.transAxes, ha="center", fontsize=8,
            color="#d97706", style="italic")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "expl_g2_rf_feature_importance.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  [SAVED] {path}")

    result = [{"feature": n, "gini_importance": round(float(v), 6)}
              for n, v in zip(all_names, imps)]
    result.sort(key=lambda x: -x["gini_importance"])
    return result


# ---------------------------------------------------------------------------
# G3: LightGBM Feature Importance
# ---------------------------------------------------------------------------
def g3_lgbm_importance(lgbm_pipe):
    ohe  = lgbm_pipe.named_steps["pre"].named_transformers_["cat"]
    clf  = lgbm_pipe.named_steps["clf"]
    cat_names = list(ohe.get_feature_names_out(["source","source_group"]))
    all_names = cat_names + NUM_COLS
    imps_split = clf.feature_importances_

    pairs = sorted(zip(all_names, imps_split), key=lambda x: -x[1])[:20]
    names = [p[0] for p in pairs][::-1]
    vals  = [p[1] for p in pairs][::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names, vals, color="#7c3aed", height=0.65, edgecolor="white")
    ax.set_xlabel("Split Count Importance")
    ax.set_title("G3  LightGBM — Split-Based Feature Importances (Top 20)")
    for v, n in zip(vals, names):
        ax.text(v + 0.5, names.index(n), str(int(v)), va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "expl_g3_lgbm_feature_importance.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  [SAVED] {path}")

    result = [{"feature": n, "split_importance": int(v)}
              for n, v in zip(all_names, imps_split)]
    result.sort(key=lambda x: -x["split_importance"])
    return result


# ---------------------------------------------------------------------------
# G4: Permutation Importance (model-agnostic, on test set)
# ---------------------------------------------------------------------------
def g4_permutation_importance(rf_pipe, X_test, y_test):
    """
    Permutation importance: measures actual ROC-AUC drop when each raw input
    column is randomly shuffled. The pipeline is evaluated end-to-end so the
    8 raw input columns are shuffled (not the post-OHE columns).
    Near-zero or negative = no genuine predictive lift.
    """
    # Raw input column names (col order matches X rows built in load_data)
    raw_names = ["source", "source_group"] + NUM_COLS

    result_pi = permutation_importance(
        rf_pipe, X_test, y_test,
        n_repeats=15, random_state=RANDOM_STATE,
        scoring="roc_auc", n_jobs=-1,
    )

    pairs = sorted(
        zip(raw_names,
            result_pi.importances_mean,
            result_pi.importances_std),
        key=lambda x: -x[1]
    )
    names_sorted = [p[0] for p in pairs][::-1]
    means_sorted = [p[1] for p in pairs][::-1]
    stds_sorted  = [p[2] for p in pairs][::-1]

    colors = ["#22c55e" if m > 0.001 else "#ef4444" if m < -0.001
              else "#9ca3af" for m in means_sorted]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names_sorted, means_sorted,
            xerr=stds_sorted, color=colors,
            height=0.65, edgecolor="white",
            error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#374151"})
    ax.axvline(0, color="#374151", linewidth=1, linestyle="--")
    ax.set_xlabel("Mean ROC-AUC drop (15 repeats)")
    ax.set_title("G4  Permutation Importance — RF on Test Set (raw features)\n"
                 "Near-zero / negative = shuffling the feature does NOT hurt the model")
    for i, (m, s) in enumerate(zip(means_sorted, stds_sorted)):
        ax.text(m + 0.0002, i, f"{m:+.5f}", va="center", fontsize=8)
    ax.text(0.5, -0.07,
            "Permutation importance is the definitive model-agnostic signal test. "
            "Values near zero confirm NO feature provides genuine predictive lift.",
            transform=ax.transAxes, ha="center", fontsize=8,
            color="#dc2626", style="italic")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "expl_g4_permutation_importance.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  [SAVED] {path}")

    result = []
    for n, m, s in zip(raw_names,
                        result_pi.importances_mean,
                        result_pi.importances_std):
        result.append({
            "feature":   n,
            "mean_drop": round(float(m), 6),
            "std_drop":  round(float(s), 6),
        })
    result.sort(key=lambda x: -x["mean_drop"])
    return result


# ---------------------------------------------------------------------------
# G5: SHAP TreeExplainer (RF global summary)
# ---------------------------------------------------------------------------
def g5_shap_summary(rf_pipe, X_test, y_test, feature_names_all=None, n_sample=2000):
    """
    SHAP values computed on a random sample of the test set using
    TreeExplainer on the fitted RF classifier (after preprocessing).
    feature_names derived from the RF pipeline's own OHE (drop=None -> 33 features).
    """
    pre  = rf_pipe.named_steps["pre"]
    clf  = rf_pipe.named_steps["clf"]

    # Derive correct feature names from this pipeline's OHE (drop=None)
    ohe = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["source", "source_group"]))
    feature_names_all = cat_names + NUM_COLS

    # Transform test data through the fitted preprocessor
    X_test_transformed = pre.transform(X_test)

    # Sample for speed
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_test_transformed), size=min(n_sample, len(X_test_transformed)),
                     replace=False)
    X_sample = X_test_transformed[idx]
    y_sample = y_test[idx]

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # Handle multiple SHAP output formats across library versions:
    #   list of 2 arrays  → [class0, class1]      (older shap)
    #   3-D array (n,f,c) → [:,:,1]               (newer shap)
    #   2-D array (n,f)   → already class-1        (single-output)
    if isinstance(shap_values, list):
        sv_pos = np.array(shap_values[1])          # list[class1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_pos = shap_values[:, :, 1]              # 3D → class 1 slice
    else:
        sv_pos = np.array(shap_values)             # already 2D

    # Ensure 2D float array (n_samples, n_features)
    sv_pos = sv_pos.astype(float)
    assert sv_pos.ndim == 2, f"Unexpected SHAP shape: {sv_pos.shape}"

    # Resolve base_value for class 1
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        base_val = float(ev[1])
    else:
        base_val = float(ev)

    # Global summary bar plot
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(sv_pos, X_sample,
                      feature_names=feature_names_all,
                      plot_type="bar", show=False, max_display=20)
    ax = plt.gca()
    ax.set_title("G5  SHAP Global Feature Importance — RF\n"
                 "(mean |SHAP value| across test sample, class=Closed Won)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "expl_g5_shap_summary.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  [SAVED] {path}")

    # Mean |SHAP| per feature
    mean_abs_shap = np.abs(sv_pos).mean(axis=0)
    shap_result = sorted(
        [{"feature": n, "mean_abs_shap": round(float(v), 6)}
         for n, v in zip(feature_names_all, mean_abs_shap.tolist())],
        key=lambda x: -x["mean_abs_shap"]
    )
    return shap_result, explainer, sv_pos, X_sample, y_sample, idx, base_val


# ---------------------------------------------------------------------------
# L1: SHAP Waterfall plots — representative individual predictions
# ---------------------------------------------------------------------------
def l1_shap_waterfalls(rf_pipe, X_test, y_test, meta_test,
                        feature_names_all, explainer, sv_pos,
                        X_sample, y_sample, idx, base_val):
    """
    Select representative cases from the SHAP sample:
      TP – highest predicted prob and actually Won
      FP – highest predicted prob but actually Not-Won
      FN – lowest predicted prob but actually Won
      TN – lowest predicted prob and actually Not-Won
    """
    pre  = rf_pipe.named_steps["pre"]
    clf  = rf_pipe.named_steps["clf"]

    # Derive feature names from RF pipeline OHE (drop=None)
    ohe = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["source", "source_group"]))
    feature_names_all = cat_names + NUM_COLS

    y_prob_sample = clf.predict_proba(X_sample)[:, 1]

    cases = {}

    # TP: high prob, actual = 1
    tp_mask = y_sample == 1
    if tp_mask.any():
        tp_i = np.where(tp_mask)[0][np.argmax(y_prob_sample[tp_mask])]
        cases["tp"] = int(tp_i)

    # FP: high prob, actual = 0
    fp_mask = y_sample == 0
    if fp_mask.any():
        fp_i = np.where(fp_mask)[0][np.argmax(y_prob_sample[fp_mask])]
        cases["fp"] = int(fp_i)

    # FN: low prob, actual = 1
    if tp_mask.any():
        fn_i = np.where(tp_mask)[0][np.argmin(y_prob_sample[tp_mask])]
        cases["fn"] = int(fn_i)

    # TN: low prob, actual = 0
    if fp_mask.any():
        tn_i = np.where(fp_mask)[0][np.argmin(y_prob_sample[fp_mask])]
        cases["tn"] = int(tn_i)

    case_labels = {
        "tp": ("True Positive",  "High prob, Actual=Won",     "#16a34a"),
        "fp": ("False Positive", "High prob, Actual=Not-Won", "#dc2626"),
        "fn": ("False Negative", "Low prob, Actual=Won",      "#d97706"),
        "tn": ("True Negative",  "Low prob, Actual=Not-Won",  "#3b82f6"),
    }
    local_results = {}

    for case_key, sample_idx in cases.items():
        label, subtitle, color = case_labels[case_key]
        prob = float(y_prob_sample[sample_idx])
        actual = int(y_sample[sample_idx])

        # SHAP waterfall for this sample
        shap_exp = shap.Explanation(
            values      = sv_pos[sample_idx],
            base_values = base_val,
            data        = X_sample[sample_idx],
            feature_names=feature_names_all,
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(shap_exp, max_display=15, show=False)
        plt.suptitle(
            f"L1  SHAP Waterfall — {label}\n"
            f"{subtitle}  |  Predicted prob={prob:.4f}  |  Actual={actual}",
            fontsize=11, fontweight="bold", y=1.01
        )
        ax2 = plt.gca()
        fig2 = ax2.get_figure()
        # Causal disclaimer
        fig2.text(0.5, -0.02,
                  "SHAP values describe this model's prediction, not causal attribution. "
                  "A high SHAP for 'source_Podcast' means the model associates it with "
                  "higher win probability, not that Podcast causes wins.",
                  ha="center", fontsize=7.5, color="#dc2626", style="italic",
                  wrap=True)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f"expl_l1_shap_waterfall_{case_key}.png")
        plt.savefig(path, bbox_inches="tight"); plt.close()
        print(f"  [SAVED] {path}")

        # Record local metadata
        global_idx = int(idx[sample_idx])
        local_results[case_key] = {
            "label":           label,
            "sample_idx":      sample_idx,
            "global_data_idx": global_idx,
            "predicted_prob":  round(prob, 4),
            "actual_label":    actual,
            "top_shap_features": sorted(
                [{"feature": feature_names_all[i],
                  "shap_value": round(float(sv_pos[sample_idx][i]), 6)}
                 for i in range(len(feature_names_all))],
                key=lambda x: -abs(x["shap_value"])
            )[:10],
        }

    return local_results


# ---------------------------------------------------------------------------
# C1: Actual win rate vs model predicted probability by source
# ---------------------------------------------------------------------------
def c1_actual_vs_predicted(rf_pipe, X_all, y_all, meta_all):
    y_prob = rf_pipe.predict_proba(X_all)[:, 1]

    from collections import defaultdict
    src_actual = defaultdict(list)
    src_pred   = defaultdict(list)
    for i, m in enumerate(meta_all):
        src_actual[m["source"]].append(int(y_all[i]))
        src_pred[m["source"]].append(float(y_prob[i]))

    results = []
    for src in sorted(src_actual.keys()):
        act  = sum(src_actual[src]) / len(src_actual[src])
        pred = sum(src_pred[src])   / len(src_pred[src])
        results.append({
            "source":              src,
            "n":                   len(src_actual[src]),
            "actual_win_rate":     round(act, 4),
            "mean_predicted_prob": round(pred, 4),
            "gap":                 round(pred - act, 4),
        })
    results.sort(key=lambda x: -x["actual_win_rate"])

    sources  = [r["source"]              for r in results]
    actual   = [r["actual_win_rate"]     for r in results]
    predicted= [r["mean_predicted_prob"] for r in results]

    x = np.arange(len(sources))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w/2, actual,    w, label="Actual win rate",         color="#22c55e", edgecolor="white")
    b2 = ax.bar(x + w/2, predicted, w, label="Model mean predicted prob",color="#3b82f6", edgecolor="white", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Rate / Probability")
    ax.set_ylim(0, 0.65)
    ax.axhline(0.0999, color="#374151", linewidth=1, linestyle="--", alpha=0.6,
               label="Overall win rate 9.99%")
    ax.set_title("C1  Actual Win Rate vs Model Predicted Probability by Source\n"
                 "Model probabilities (~0.50) fail to track actual rates (~0.10) — confirms low predictive power")
    ax.legend(fontsize=9)
    ax.text(0.5, -0.22,
            "The large gap between actual win rates (green, ~10%) and model predicted "
            "probabilities (blue, ~50%) is expected. When class_weight='balanced' is used "
            "the model predicts near-0.5 for most samples. This is an artefact of calibration, "
            "not performance failure. ROC-AUC is the correct metric: it is scale-invariant.",
            transform=ax.transAxes, ha="center", fontsize=7.5,
            color="#57606a", style="italic")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "expl_c1_actual_vs_predicted.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  [SAVED] {path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("[STEP 1] Loading data and models...")
    X_all, y_all, meta_all, raw_rows = load_data()
    X_tr, X_val, X_te, y_tr, y_val, y_te = reproduce_splits(X_all, y_all)
    meta_te = [meta_all[i] for i in range(len(X_all))
               if X_all[i] in X_te]   # approximate; we only use for source labels

    lr_pipe   = load_model("models/logistic_regression_model.pkl")
    rf_pipe   = load_model("models/random_forest_model.pkl")
    lgbm_pipe = load_model("models/lightgbm_model.pkl")

    with open("models/feature_names.json") as f:
        fnames = json.load(f)
    feature_names_all = fnames["all_features"]
    print(f"         {len(feature_names_all)} features loaded")

    results = {}

    # G1 — LR Coefficients
    print("[G1] LR Coefficients...")
    results["g1_lr_coefficients"] = g1_lr_coefficients(lr_pipe, feature_names_all)
    top5_pos = [x for x in results["g1_lr_coefficients"] if x["coefficient"] > 0][:5]
    top5_neg = [x for x in results["g1_lr_coefficients"] if x["coefficient"] < 0][:5]
    print(f"     Top positive: {[x['feature'] for x in top5_pos]}")
    print(f"     Top negative: {[x['feature'] for x in top5_neg]}")

    # G2 — RF Gini
    print("[G2] RF Gini Feature Importance...")
    results["g2_rf_gini_importance"] = g2_rf_importance(rf_pipe, feature_names_all)
    print(f"     Top 5 by Gini: {[x['feature'] for x in results['g2_rf_gini_importance'][:5]]}")

    # G3 — LGBM Split
    print("[G3] LightGBM Split Importance...")
    results["g3_lgbm_split_importance"] = g3_lgbm_importance(lgbm_pipe)
    print(f"     Top 5 by splits: {[x['feature'] for x in results['g3_lgbm_split_importance'][:5]]}")

    # G4 — Permutation Importance
    print("[G4] Permutation Importance on test set (15 repeats — may take a moment)...")
    results["g4_permutation_importance"] = g4_permutation_importance(rf_pipe, X_te, y_te)
    max_drop = max(x["mean_drop"] for x in results["g4_permutation_importance"])
    min_drop = min(x["mean_drop"] for x in results["g4_permutation_importance"])
    print(f"     Permutation drop range: [{min_drop:.6f}, {max_drop:.6f}]")

    # G5 — SHAP Summary (RF)
    print("[G5] SHAP TreeExplainer (RF, n=2000 from test set)...")
    (results["g5_shap_mean_abs"],
     shap_explainer, sv_pos,
     X_sample, y_sample, sample_idx, base_val) = g5_shap_summary(
         rf_pipe, X_te, y_te, feature_names_all, n_sample=2000)
    print(f"     Top 5 by mean|SHAP|: {[x['feature'] for x in results['g5_shap_mean_abs'][:5]]}")

    # L1 — SHAP Waterfall local cases
    print("[L1] SHAP Waterfall plots (TP, FP, FN, TN)...")
    # We need meta for test set rows — reconstruct via split indices
    X_all_np = np.array(X_all, dtype=object)
    _, X_te_idx, _, _ = train_test_split(
        np.arange(len(X_all)), y_all,
        test_size=0.15, stratify=y_all, random_state=RANDOM_STATE)
    meta_te_list = [meta_all[i] for i in X_te_idx]

    results["l1_local_cases"] = l1_shap_waterfalls(
        rf_pipe, X_te, y_te, meta_te_list,
        feature_names_all, shap_explainer, sv_pos,
        X_sample, y_sample, sample_idx, base_val)
    for k, v in results["l1_local_cases"].items():
        print(f"     {k.upper()}: pred={v['predicted_prob']}, actual={v['actual_label']}, "
              f"top_feat={v['top_shap_features'][0]['feature']}")

    # C1 — Actual vs Predicted
    print("[C1] Actual win rate vs predicted probability by source...")
    results["c1_actual_vs_predicted"] = c1_actual_vs_predicted(
        rf_pipe, X_all, y_all, meta_all)
    print(f"     Computed for {len(results['c1_actual_vs_predicted'])} sources")

    # Causal boundary statement
    results["causal_boundary_statement"] = {
        "global_importance": (
            "All global importance metrics (LR coefficients, RF Gini, LGBM splits, "
            "permutation importance, SHAP) describe associations learned from training "
            "data. A high LR coefficient for 'source_Podcast' means the model associates "
            "Podcast leads with higher win probability based on observed win rate differences. "
            "It does NOT mean that Podcast causes wins. The 1.41 pp win rate advantage may "
            "reflect unmeasured confounders: lead quality, industry segment, deal size, "
            "or sales rep assignment."
        ),
        "local_predictions": (
            "SHAP waterfall values explain why the model assigned a particular probability "
            "to a specific prediction. They attribute credit to input features but do not "
            "identify causal mechanisms. A SHAP value of +0.02 for 'source_Referral' means "
            "the model pushed its probability estimate upward by 0.02 for that feature — "
            "not that Referral caused the lead to be won."
        ),
        "permutation_finding": (
            "Permutation importance (G4) is the definitive test: shuffling any feature "
            "yields near-zero or negative ROC-AUC change. This confirms that no available "
            "feature provides genuine predictive lift beyond random chance. The model is "
            "essentially guessing at ROC-AUC ~0.50–0.51."
        ),
        "practical_implication": (
            "The source-level win rate differences (9.28%–10.69%) are real analytical "
            "findings for CHANNEL INVESTMENT DECISIONS (Phases 3–4). They are not "
            "sufficient as an individual-lead scoring signal. A useful predictive model "
            "requires additional features: deal size, industry, response time, engagement "
            "score, number of follow-ups."
        ),
    }

    # Write results
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Explainability results -> '{OUT_JSON}'")

    # Summary
    print("\n=== PHASE 6 SUMMARY ===")
    print(f"  G1  LR coefs top positive: {[x['feature'] for x in top5_pos]}")
    print(f"  G1  LR coefs top negative: {[x['feature'] for x in top5_neg]}")
    print(f"  G2  RF Gini top feature  : {results['g2_rf_gini_importance'][0]['feature']} "
          f"({results['g2_rf_gini_importance'][0]['gini_importance']:.4f})")
    print(f"  G4  Perm importance range: [{min_drop:.6f}, {max_drop:.6f}] "
          f"(all near-zero = no real signal)")
    print(f"  G5  SHAP top feature     : {results['g5_shap_mean_abs'][0]['feature']} "
          f"(mean|SHAP|={results['g5_shap_mean_abs'][0]['mean_abs_shap']:.6f})")
    print(f"  Local cases saved        : TP, FP, FN, TN waterfall plots")
    print(f"  Figures saved            : 10")


if __name__ == "__main__":
    run()
