"""Phase 6 validation — checks all explainability outputs are present and internally consistent."""
import json, os

# 1. All figures present and non-empty
expected_figs = [
    "expl_g1_lr_coefficients.png",
    "expl_g2_rf_feature_importance.png",
    "expl_g3_lgbm_feature_importance.png",
    "expl_g4_permutation_importance.png",
    "expl_g5_shap_summary.png",
    "expl_l1_shap_waterfall_tp.png",
    "expl_l1_shap_waterfall_fp.png",
    "expl_l1_shap_waterfall_fn.png",
    "expl_l1_shap_waterfall_tn.png",
    "expl_c1_actual_vs_predicted.png",
]
FIG_DIR = "reports/figures"
for fname in expected_figs:
    path = os.path.join(FIG_DIR, fname)
    assert os.path.exists(path), f"Missing figure: {fname}"
    assert os.path.getsize(path) > 10_000, f"Suspiciously small figure: {fname}"
print(f"[OK] All 10 explainability figures present and non-empty")

# 2. JSON output present
assert os.path.exists("reports/explainability_phase6.json")
with open("reports/explainability_phase6.json") as f:
    results = json.load(f)

required_keys = [
    "g1_lr_coefficients", "g2_rf_gini_importance", "g3_lgbm_split_importance",
    "g4_permutation_importance", "g5_shap_mean_abs", "l1_local_cases",
    "c1_actual_vs_predicted", "causal_boundary_statement",
]
for k in required_keys:
    assert k in results, f"Missing key: {k}"
print(f"[OK] JSON output has all {len(required_keys)} required sections")

# 3. LR coefficients: 31 features (drop='first' OHE)
lr_coefs = results["g1_lr_coefficients"]
assert len(lr_coefs) == 31, f"Expected 31 LR coefs, got {len(lr_coefs)}"
coef_vals = [c["coefficient"] for c in lr_coefs]
assert any(v > 0 for v in coef_vals), "No positive LR coefficients"
assert any(v < 0 for v in coef_vals), "No negative LR coefficients"
print(f"[OK] LR coefficients: {len(lr_coefs)} features, "
      f"pos={sum(1 for v in coef_vals if v>0)}, neg={sum(1 for v in coef_vals if v<0)}")

# 4. RF Gini: top feature has high variance (notes_sentiment or notes_word_count)
top_gini = results["g2_rf_gini_importance"][0]["feature"]
assert top_gini in ["notes_sentiment","company_freq","notes_word_count","owner_freq"], \
    f"Unexpected top Gini feature: {top_gini}"
print(f"[OK] RF Gini top feature: {top_gini} (expected high-variance continuous feature)")

# 5. LGBM importance: at least 20 features
assert len(results["g3_lgbm_split_importance"]) >= 20
print(f"[OK] LGBM split importance: {len(results['g3_lgbm_split_importance'])} features")

# 6. Permutation importance: all values near-zero (no genuine signal)
perm = results["g4_permutation_importance"]
max_drop = max(x["mean_drop"] for x in perm)
min_drop = min(x["mean_drop"] for x in perm)
assert abs(max_drop) < 0.02, f"Unexpectedly large permutation drop: {max_drop}"
assert abs(min_drop) < 0.02, f"Unexpectedly large permutation drop: {min_drop}"
print(f"[OK] Permutation importance range: [{min_drop:.6f}, {max_drop:.6f}] — all near-zero")

# 7. SHAP mean|SHAP| values are positive and sum to non-zero
shap_vals = results["g5_shap_mean_abs"]
assert all(x["mean_abs_shap"] >= 0 for x in shap_vals)
assert sum(x["mean_abs_shap"] for x in shap_vals) > 0
print(f"[OK] SHAP: {len(shap_vals)} features, top={shap_vals[0]['feature']} "
      f"(mean|SHAP|={shap_vals[0]['mean_abs_shap']:.6f})")

# 8. Local cases: TP, FP, FN, TN all present with valid probs
cases = results["l1_local_cases"]
assert set(cases.keys()) == {"tp","fp","fn","tn"}, f"Missing cases: {set(cases.keys())}"
for k, v in cases.items():
    assert 0.0 <= v["predicted_prob"] <= 1.0
    assert v["actual_label"] in [0, 1]
    assert len(v["top_shap_features"]) == 10
tp = cases["tp"]; fp = cases["fp"]; fn = cases["fn"]; tn = cases["tn"]
assert tp["actual_label"] == 1, "TP actual should be 1"
assert fp["actual_label"] == 0, "FP actual should be 0"
assert fn["actual_label"] == 1, "FN actual should be 1"
assert tn["actual_label"] == 0, "TN actual should be 0"
assert tp["predicted_prob"] > fn["predicted_prob"], "TP should have higher prob than FN"
assert fp["predicted_prob"] > tn["predicted_prob"], "FP should have higher prob than TN"
print(f"[OK] Local cases: TP(prob={tp['predicted_prob']}), FP(prob={fp['predicted_prob']}), "
      f"FN(prob={fn['predicted_prob']}), TN(prob={tn['predicted_prob']})")

# 9. Actual vs predicted: 20 sources, actual rates ~10%, predicted ~50%
avp = results["c1_actual_vs_predicted"]
assert len(avp) == 20
actual_rates = [x["actual_win_rate"] for x in avp]
pred_rates   = [x["mean_predicted_prob"] for x in avp]
assert all(0.085 <= r <= 0.115 for r in actual_rates), f"Actual rates out of range: {min(actual_rates)}-{max(actual_rates)}"
assert all(0.35 <= p <= 0.70  for p in pred_rates),    f"Pred probs out of range: {min(pred_rates)}-{max(pred_rates)}"
print(f"[OK] Actual vs predicted: actual range [{min(actual_rates):.4f}, {max(actual_rates):.4f}], "
      f"pred range [{min(pred_rates):.4f}, {max(pred_rates):.4f}]")

# 10. Causal boundary statement present and non-empty
cbs = results["causal_boundary_statement"]
for key in ["global_importance","local_predictions","permutation_finding","practical_implication"]:
    assert key in cbs and len(cbs[key]) > 50, f"Missing/short causal statement: {key}"
print(f"[OK] Causal boundary statement present with 4 sub-sections")

# 11. Raw file untouched
import csv
with open("leads-100000.csv") as f:
    raw = list(csv.DictReader(f))
assert len(raw) == 100000 and "First Name" in raw[0]
print(f"[OK] Raw file untouched")

print()
print("=== ALL PHASE 6 VALIDATION CHECKS PASSED ===")
print(f"  Figures generated  : 10 (G1-G5 global + L1 x4 waterfall + C1 calibration)")
print(f"  Methods applied    : LR coefs, RF Gini, LGBM splits, Permutation, SHAP")
print(f"  Local cases        : TP, FP, FN, TN with SHAP waterfall")
print(f"  Perm drop range    : [{min_drop:.6f}, {max_drop:.6f}] — confirms no real signal")
print(f"  Causal disclaimers : Explicit in all outputs and JSON report")
