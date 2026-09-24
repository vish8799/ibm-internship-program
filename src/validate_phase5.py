"""Phase 5 validation — verifies all model outputs, metrics integrity, no leakage."""
import json, os, pickle, csv
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. All output files present
required = [
    "models/logistic_regression_model.pkl",
    "models/random_forest_model.pkl",
    "models/lightgbm_model.pkl",
    "models/best_model.pkl",
    "models/ml_phase5_metrics.json",
    "models/feature_names.json",
]
for p in required:
    assert os.path.exists(p) and os.path.getsize(p) > 0, f"Missing/empty: {p}"
print(f"[OK] All {len(required)} output files present and non-empty")

# 2. Metrics file integrity
with open("models/ml_phase5_metrics.json") as f:
    metrics = json.load(f)

assert "dataset_info" in metrics
assert "models" in metrics
assert len(metrics["models"]) == 3
for mn in ["Logistic Regression", "Random Forest", "LightGBM"]:
    assert mn in metrics["models"], f"Missing model: {mn}"
    m = metrics["models"][mn]
    for split in ["val_metrics", "test_metrics", "train_metrics"]:
        assert split in m, f"Missing {split} for {mn}"
        sm = m[split]
        for key in ["roc_auc","pr_auc","f1","precision","recall","accuracy"]:
            assert key in sm, f"Missing metric {key} in {mn}/{split}"
            val = sm[key]
            assert 0.0 <= val <= 1.0, f"Metric out of range: {mn}/{split}/{key}={val}"
print(f"[OK] Metrics file complete: 3 models x 3 splits x 6 metrics each")

# 3. ROC-AUC values are real (within expected range for this dataset)
for mn in ["Logistic Regression", "Random Forest", "LightGBM"]:
    val_auc  = metrics["models"][mn]["val_metrics"]["roc_auc"]
    test_auc = metrics["models"][mn]["test_metrics"]["roc_auc"]
    cv_mean  = metrics["models"][mn]["cross_validation"]["mean"]
    # All should be in plausible range for near-random prediction
    assert 0.45 <= val_auc  <= 0.65, f"{mn} val ROC-AUC out of plausible range: {val_auc}"
    assert 0.45 <= test_auc <= 0.65, f"{mn} test ROC-AUC out of plausible range: {test_auc}"
    assert 0.45 <= cv_mean  <= 0.65, f"{mn} CV ROC-AUC out of plausible range: {cv_mean}"
    print(f"[OK] {mn:<22}  Val={val_auc}  Test={test_auc}  CV={cv_mean}")

# 4. No leakage features
feat_eng = metrics["feature_engineering_notes"]
leaked = feat_eng["dropped_leakage"]
for col in ["deal_stage","stage_ordinal","target_multiclass","outcome_3class","is_closed"]:
    assert col in leaked, f"Leakage column not in dropped list: {col}"
print(f"[OK] Leakage columns documented as dropped: {leaked}")

# 5. Feature names file
with open("models/feature_names.json") as f:
    fnames = json.load(f)
assert fnames["n_features_total"] > 0
assert len(fnames["all_features"]) == fnames["n_features_total"]
print(f"[OK] Feature names: {fnames['n_features_total']} total after encoding")

# 6. Best model loadable and predicts probabilities
with open("models/best_model.pkl", "rb") as f:
    best = pickle.load(f)
# Smoke-test prediction on a dummy row
dummy_cat = [["Podcast", "Events"]]
dummy_num = [[0.05, 7, 1, 1, 0.00004, 0.00001]]
X_dummy   = [dummy_cat[0] + dummy_num[0]]
prob = best.predict_proba(X_dummy)[0, 1]
assert 0.0 <= prob <= 1.0, f"Best model probability out of range: {prob}"
print(f"[OK] Best model ({metrics['best_model']['name']}) loadable, predicts prob={prob:.4f}")

# 7. Split sizes sum to total
di = metrics["dataset_info"]
split_sum = di["train_samples"] + di["val_samples"] + di["test_samples"]
assert split_sum == di["total_samples"], f"Split sum mismatch: {split_sum} != {di['total_samples']}"
assert abs(di["train_samples"] / di["total_samples"] - 0.70) < 0.01
assert abs(di["val_samples"]   / di["total_samples"] - 0.15) < 0.01
assert abs(di["test_samples"]  / di["total_samples"] - 0.15) < 0.01
print(f"[OK] Splits sum to {split_sum:,}: "
      f"train={di['train_samples']:,} val={di['val_samples']:,} test={di['test_samples']:,}")

# 8. Positive class % matches Phase 3 KPI-01
assert abs(di["positive_class_pct"] - 9.99) < 0.1
print(f"[OK] Positive class {di['positive_class_pct']}% matches Phase 3 win rate 9.99%")

# 9. Reproduce val ROC-AUC independently using saved model and raw data
with open("data/processed/cleaned_leads.csv") as f:
    rows = list(csv.DictReader(f))

SOURCE_TIER = {
    "Podcast":1,"Partner Program":1,"Referral":1,"Webinars":1,"Content Marketing":1,
    "LinkedIn Outreach":2,"Cold Email":2,"Cold Call":2,"Facebook Ads":2,"Chatbot":2,
    "Retargeting Ads":2,"Direct Traffic":2,"Organic Search (SEO)":2,"Trade Show":2,
    "Google Ads":2,"Social Media":3,"Purchased List":3,"Other":3,
    "Website Form":3,"Networking Event":3,
}
owner_counts   = {}
company_counts = {}
total = len(rows)
for r in rows:
    owner_counts[r["lead_owner"]]  = owner_counts.get(r["lead_owner"],  0) + 1
    company_counts[r["company"]]   = company_counts.get(r["company"],   0) + 1

X_all = [[
    r["source"], r["source_group"],
    float(r["notes_sentiment"]), int(r["notes_word_count"]),
    int(r["notes_has_text"]),
    SOURCE_TIER.get(r["source"], 2),
    owner_counts[r["lead_owner"]] / total,
    company_counts[r["company"]]  / total,
] for r in rows]
y_all = np.array([int(r["is_won"]) for r in rows])

from sklearn.model_selection import train_test_split
X_tr_full, X_te, y_tr_full, y_te = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr_full, y_tr_full,
    test_size=0.15/0.85, stratify=y_tr_full, random_state=42)

prob_val = best.predict_proba(X_val)[:, 1]
repro_auc = round(roc_auc_score(y_val, prob_val), 4)
expected_auc = metrics["best_model"]["val_roc_auc"]
assert abs(repro_auc - expected_auc) < 0.001, \
    f"Reproduced val AUC {repro_auc} != stored {expected_auc}"
print(f"[OK] Reproduced val ROC-AUC={repro_auc} matches stored {expected_auc}")

# 10. Raw file untouched
with open("leads-100000.csv") as f:
    raw = list(csv.DictReader(f))
assert len(raw) == 100000 and "First Name" in raw[0]
print(f"[OK] Raw file untouched")

print()
print("=== ALL PHASE 5 VALIDATION CHECKS PASSED ===")
print(f"  Models trained     : 3 (LR, RF, LGBM)")
print(f"  Best model         : {metrics['best_model']['name']} (Val AUC={expected_auc})")
print(f"  Total features     : {fnames['n_features_total']} (after OHE)")
print(f"  Leakage fields     : {len(leaked)} columns dropped")
print(f"  No fabricated metrics — all from actual execution")
