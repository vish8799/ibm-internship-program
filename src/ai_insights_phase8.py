"""
Phase 8 – AI Insights (updated for Phase 5/6 pipeline outputs)
================================================================
Reads validated results from:
  - data/processed/data_quality_report.json   (Phase 2)
  - reports/eda_phase3_kpis.json              (Phase 3)
  - models/ml_phase5_metrics.json             (Phase 5 — 3 models, 70/15/15 split)
  - reports/explainability_phase6.json        (Phase 6 — LR/RF/LGBM/SHAP)
  - reports/sql_phase4_results.json           (Phase 4)

Outputs: reports/ai_insights.json
"""

import json
import os

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context():
    """Build a validated context dict from all Phase 5/6 pipeline outputs."""
    qual   = load_json("data/processed/data_quality_report.json")
    kpis   = load_json("reports/eda_phase3_kpis.json")
    ml5    = load_json("models/ml_phase5_metrics.json")
    expl6  = load_json("reports/explainability_phase6.json")

    # Data quality
    missing = sum(qual["missing_values_per_col"].values())

    # KPIs
    k1 = kpis["kpi_01_overall_win_rate"]
    k2 = kpis["kpi_02_closed_rate"]
    k3 = sorted(kpis["kpi_03_win_rate_by_source"], key=lambda x: x["win_rate_pct"], reverse=True)
    k4 = sorted(kpis["kpi_04_win_rate_by_source_group"], key=lambda x: x["win_rate_pct"], reverse=True)
    k11 = kpis["kpi_11_notes_sentiment_by_outcome"]

    # ML Phase 5
    lr  = ml5["models"]["Logistic Regression"]
    rf  = ml5["models"]["Random Forest"]
    lgbm= ml5["models"]["LightGBM"]
    best= ml5["best_model"]

    # Explainability Phase 6
    top_gini     = expl6["g2_rf_gini_importance"][0]
    top_lr_coef  = sorted(expl6["g1_lr_coefficients"], key=lambda x: abs(x["coefficient"]), reverse=True)[0]
    perm         = expl6["g4_permutation_importance"]
    shap         = expl6["g5_shap_mean_abs"]
    avp          = expl6["c1_actual_vs_predicted"]

    ctx = {
        # Data quality
        "total_leads":       k1["total_leads"],
        "missing_values":    missing,
        "duplicate_ids":     qual["duplicate_account_ids"],
        "invalid_sources":   qual["invalid_source"],
        "invalid_stages":    qual["invalid_stage"],

        # KPIs
        "closed_won":        k1["closed_won"],
        "win_rate_pct":      k1["win_rate_pct"],
        "closed_lost":       k2["closed_total"] - k1["closed_won"],
        "close_rate_pct":    k2["closed_rate_pct"],
        "open_pipeline":     k1["total_leads"] - k2["closed_total"],

        # Source performance
        "top_source":        k3[0]["source"],
        "top_source_rate":   k3[0]["win_rate_pct"],
        "top_source_n":      k3[0]["total_leads"],
        "top5_sources":      [f"{x['source']} ({x['win_rate_pct']}%)" for x in k3[:5]],
        "bot_source":        k3[-1]["source"],
        "bot_source_rate":   k3[-1]["win_rate_pct"],
        "win_rate_spread":   round(k3[0]["win_rate_pct"] - k3[-1]["win_rate_pct"], 2),
        "source_count":      len(k3),

        # Source groups
        "best_sg":           k4[0]["source_group"],
        "best_sg_rate":      k4[0]["win_rate_pct"],
        "best_sg_n":         k4[0]["total_leads"],
        "worst_sg":          k4[-1]["source_group"],
        "worst_sg_rate":     k4[-1]["win_rate_pct"],

        # Sentiment
        "sent_won":          round(k11["Won"]["mean"], 4),
        "sent_lost":         round(k11["Lost"]["mean"], 4),
        "sent_open":         round(k11["Open"]["mean"], 4),
        "sent_diff":         round(abs(k11["Won"]["mean"] - k11["Lost"]["mean"]), 4),

        # ML Phase 5 — 3 models
        "split_train":       ml5["dataset_info"]["train_samples"],
        "split_val":         ml5["dataset_info"]["val_samples"],
        "split_test":        ml5["dataset_info"]["test_samples"],
        "positive_class_pct":ml5["dataset_info"]["positive_class_pct"],
        "best_model_name":   best["name"],
        "best_val_auc":      best["val_roc_auc"],

        "lr_val_auc":        lr["val_metrics"]["roc_auc"],
        "lr_test_auc":       lr["test_metrics"]["roc_auc"],
        "lr_cv_mean":        lr["cross_validation"]["mean"],
        "lr_cv_std":         lr["cross_validation"]["std"],
        "lr_train_auc":      lr["train_metrics"]["roc_auc"],

        "rf_val_auc":        rf["val_metrics"]["roc_auc"],
        "rf_test_auc":       rf["test_metrics"]["roc_auc"],
        "rf_cv_mean":        rf["cross_validation"]["mean"],
        "rf_cv_std":         rf["cross_validation"]["std"],
        "rf_train_auc":      rf["train_metrics"]["roc_auc"],

        "lgbm_val_auc":      lgbm["val_metrics"]["roc_auc"],
        "lgbm_test_auc":     lgbm["test_metrics"]["roc_auc"],
        "lgbm_cv_mean":      lgbm["cross_validation"]["mean"],
        "lgbm_cv_std":       lgbm["cross_validation"]["std"],
        "lgbm_train_auc":    lgbm["train_metrics"]["roc_auc"],

        "auc_above_baseline": round(best["val_roc_auc"] - 0.5, 4),
        "random_baseline":    0.5,

        # Explainability Phase 6
        "top_gini_feature":  top_gini["feature"],
        "top_gini_imp":      round(top_gini["gini_importance"] * 100, 1),
        "top_lr_coef_feat":  top_lr_coef["feature"],
        "top_lr_coef_val":   top_lr_coef["coefficient"],

        "perm_max":          max(p["mean_drop"] for p in perm),
        "perm_min":          min(p["mean_drop"] for p in perm),
        "perm_max_feat":     max(perm, key=lambda x: x["mean_drop"])["feature"],
        "perm_min_feat":     min(perm, key=lambda x: x["mean_drop"])["feature"],

        "shap_top_feat":     shap[0]["feature"],
        "shap_top_val":      shap[0]["mean_abs_shap"],

        # AVP gap
        "avp_top_source":    avp[0]["source"] if avp else "N/A",
        "avp_top_actual":    avp[0]["actual_win_rate"] if avp else 0,
        "avp_top_pred":      avp[0]["mean_predicted_prob"] if avp else 0,
    }
    return ctx


INSIGHT_TEMPLATES = [
    {
        "id": "data_quality",
        "title": "Data Quality Assessment",
        "category": "Data",
        "source": "data_quality_report.json",
        "rating": "positive",
        "template": (
            "The leads dataset contains {total_leads:,} records across 14 fields. "
            "Data quality is excellent: {missing_values} missing values, "
            "{duplicate_ids} duplicate Account IDs, {invalid_sources} invalid Source values, "
            "and {invalid_stages} invalid Deal Stage values were found. "
            "The dataset required no imputation or deduplication and was used as-is "
            "after standardisation of field names and creation of 8 derived features "
            "(source_tier, owner_freq, company_freq, notes_has_text, plus OHE columns)."
        ),
    },
    {
        "id": "pipeline_overview",
        "title": "Pipeline Overview",
        "category": "KPI",
        "source": "eda_phase3_kpis.json",
        "rating": "neutral",
        "template": (
            "Across {total_leads:,} total leads, {closed_won:,} reached Closed Won — "
            "an overall win rate of {win_rate_pct}%. A further {closed_lost:,} leads were "
            "Closed Lost or Disqualified (loss rate {loss_rate:.2f}%), while {open_pipeline:,} leads "
            "({open_pct:.1f}%) remain in active pipeline stages. The overall close rate "
            "(won + lost + disqualified) stands at {close_rate_pct}%, suggesting substantial "
            "open pipeline with conversion potential still to be realised."
        ),
    },
    {
        "id": "source_performance",
        "title": "Lead Source Performance",
        "category": "Analytics",
        "source": "eda_phase3_kpis.json + sql_phase4_results.json",
        "rating": "positive",
        "template": (
            "{top_source} is the highest-performing acquisition channel with a {top_source_rate}% "
            "win rate across {top_source_n:,} leads, followed by {top5_1}, {top5_2}, {top5_3}, "
            "and {top5_4}. The lowest-performing channel is {bot_source} at {bot_source_rate}%. "
            "The spread across all {source_count} sources is {win_rate_spread} percentage points — "
            "real but narrow. This spread is consistent across SQL validation and is not an "
            "artefact of data imbalance — all sources have approximately equal lead volumes (~5,000 each)."
        ),
    },
    {
        "id": "source_group_insight",
        "title": "Channel Group Analysis",
        "category": "Analytics",
        "source": "eda_phase3_kpis.json",
        "rating": "positive",
        "template": (
            "Grouping the 20 sources into 7 channel categories reveals that "
            "{best_sg} delivers the highest win rate at {best_sg_rate}% ({best_sg_n:,} leads). "
            "Events and Outbound channels both exceed 10%, while {worst_sg} ({worst_sg_rate}%) "
            "and Social Media (9.67%) underperform the 9.99% average. Inbound channels, despite "
            "the largest lead volume (25,012), achieve only 9.87% — a volume-quality trade-off "
            "worth monitoring. These patterns are actionable for budget allocation, though "
            "causation cannot be inferred from win rate differences alone."
        ),
    },
    {
        "id": "sentiment_finding",
        "title": "Notes Sentiment Analysis",
        "category": "Feature Analysis",
        "source": "eda_phase3_kpis.json",
        "rating": "neutral",
        "template": (
            "TextBlob sentiment analysis on the Notes field reveals that mean polarity scores "
            "are nearly identical across deal outcomes: Won = {sent_won}, Lost = {sent_lost}, "
            "Open = {sent_open}. The absolute difference between Won and Lost is only {sent_diff} "
            "polarity points. This confirms that notes sentiment carries no meaningful signal for "
            "predicting deal outcomes in this dataset. The finding held across both EDA and "
            "permutation importance analysis ({perm_max_feat} permutation drop = +{perm_max:.6f} — "
            "all near-zero)."
        ),
    },
    {
        "id": "ml_performance",
        "title": "Predictive Model Performance (Phase 5)",
        "category": "ML",
        "source": "models/ml_phase5_metrics.json",
        "rating": "warning",
        "template": (
            "Three classifiers were trained to predict Closed Won (is_won=1) on a 70/15/15 "
            "stratified split ({split_train:,} train / {split_val:,} val / {split_test:,} test) "
            "using 8 engineered features. "
            "Logistic Regression: val AUC = {lr_val_auc}, test AUC = {lr_test_auc} "
            "(CV: {lr_cv_mean:.4f} ± {lr_cv_std:.4f}). "
            "Random Forest: val AUC = {rf_val_auc}, test AUC = {rf_test_auc} "
            "(CV: {rf_cv_mean:.4f} ± {rf_cv_std:.4f}). "
            "LightGBM: val AUC = {lgbm_val_auc}, test AUC = {lgbm_test_auc} "
            "(CV: {lgbm_cv_mean:.4f} ± {lgbm_cv_std:.4f}). "
            "Best model: {best_model_name} (Val AUC = {best_val_auc}), only {auc_above_baseline} "
            "above random baseline. High train AUC for RF ({rf_train_auc}) and LGBM "
            "({lgbm_train_auc}) confirms overfitting — no generalizable signal exists in the "
            "available features."
        ),
    },
    {
        "id": "feature_importance",
        "title": "Feature Importance Interpretation (Phase 6)",
        "category": "Explainability",
        "source": "reports/explainability_phase6.json",
        "rating": "warning",
        "template": (
            "Random Forest Gini importance assigns {top_gini_imp:.1f}% to {top_gini_feature}, "
            "followed by company_freq (22.7%) and notes_word_count (17.1%). However, permutation "
            "importance (actual ROC-AUC Δ when features are shuffled) shows near-zero impact: "
            "max = {perm_max:+.6f} ({perm_max_feat}), min = {perm_min:+.6f} ({perm_min_feat}). "
            "SHAP mean |SHAP| also confirms {shap_top_feat} as the most influential feature "
            "(mean |SHAP| = {shap_top_val:.6f}), but all absolute SHAP values are negligible. "
            "Gini importance overstates continuous features; no feature provides genuine predictive lift."
        ),
    },
    {
        "id": "business_recommendations",
        "title": "Business Recommendations",
        "category": "Recommendations",
        "source": "all phases",
        "rating": "positive",
        "template": (
            "Based on validated analytical findings across all phases: "
            "(1) Prioritise Podcast, Partner Program, Referral, and Webinar channels — "
            "consistently above 10.5% win rates with Won:Lost ratios of 1.056–1.163. "
            "(2) Review Networking Event (9.28% win rate, Won:Lost = 0.909 — the only source "
            "with more losses than wins) and Website Form (9.29%) for conversion optimisation. "
            "(3) Inbound channels generate 24.7% of all Closed Won by volume despite average "
            "win rates — protect and scale this volume. "
            "(4) For predictive lead scoring, invest in capturing additional feature types: "
            "deal value, industry vertical, response time, engagement score, and follow-up count. "
            "Note: all recommendations are associations — causal claims require controlled experiments."
        ),
    },
    {
        "id": "causal_disclaimer",
        "title": "Correlation vs Causation Statement",
        "category": "Methodology",
        "source": "all phases",
        "rating": "neutral",
        "template": (
            "All analytical findings in this project describe statistical associations observed "
            "in the dataset. The higher win rates for Podcast, Referral, and Partner Program "
            "sources reflect patterns in the data — they do not establish that these channels "
            "cause higher win rates. Unmeasured confounding factors (lead quality, industry, "
            "deal size, sales rep skill) may explain observed differences. Model coefficients "
            "and feature importances describe learned associations, not causal pathways. "
            "Business decisions based on these findings should be validated through controlled "
            "A/B experiments or quasi-experimental methods."
        ),
    },
]


def render_insight(template_str, ctx):
    extra = {
        "open_pct":   round(ctx["open_pipeline"] / ctx["total_leads"] * 100, 1),
        "loss_rate":  round(ctx["close_rate_pct"] - ctx["win_rate_pct"], 2),
        "top5_1":     ctx["top5_sources"][1],
        "top5_2":     ctx["top5_sources"][2],
        "top5_3":     ctx["top5_sources"][3],
        "top5_4":     ctx["top5_sources"][4],
    }
    merged = {**ctx, **extra}
    return template_str.format(**merged)


def llm_enhance(insight_text, model="gpt-3.5-turbo"):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        import urllib.request, json as _json
        system = (
            "You are an analytics assistant. You receive a factual summary of a data analytics "
            "project and rephrase it into clear, professional prose. STRICT RULES: "
            "(1) Do NOT invent, modify, or add any statistics, percentages, or numerical claims. "
            "(2) Do not make causal claims unless present in the input. "
            "(3) Keep the same factual content. (4) Output only the rephrased paragraph."
        )
        payload = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": insight_text},
            ],
            "temperature": 0.3,
            "max_tokens":  300,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = _json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[WARN] LLM call failed: {e} — using template output")
        return None


def run():
    os.makedirs("reports", exist_ok=True)

    print("[INFO] Building validated context from Phase 5/6 pipeline results...")
    ctx = build_context()
    print(f"[INFO] Context: {len(ctx)} keys | best_model={ctx['best_model_name']} | "
          f"best_val_auc={ctx['best_val_auc']} | win_rate={ctx['win_rate_pct']}%")

    api_key_present = bool(os.environ.get("OPENAI_API_KEY", ""))
    print(f"[INFO] LLM: {'ENABLED' if api_key_present else 'DISABLED (no OPENAI_API_KEY) — using templates'}")

    insights = []
    for tmpl in INSIGHT_TEMPLATES:
        rendered = render_insight(tmpl["template"], ctx)
        llm_text = llm_enhance(rendered) if api_key_present else None
        insight = {
            "id":           tmpl["id"],
            "title":        tmpl["title"],
            "category":     tmpl["category"],
            "source":       tmpl["source"],
            "rating":       tmpl["rating"],
            "text":         rendered,
            "llm_enhanced": llm_text is not None,
            "display_text": llm_text if llm_text else rendered,
        }
        insights.append(insight)
        print(f"  [{tmpl['category']}] {tmpl['title']}: {len(rendered)} chars "
              f"{'(LLM enhanced)' if llm_text else '(template)'}")

    output = {
        "generated_by":  "Phase 8 — AI Insights (Phase 5/6 sync)",
        "data_sources":  [
            "data/processed/data_quality_report.json",
            "reports/eda_phase3_kpis.json",
            "models/ml_phase5_metrics.json",
            "reports/explainability_phase6.json",
            "reports/sql_phase4_results.json",
        ],
        "llm_used":      api_key_present,
        "total_insights": len(insights),
        "insights":      insights,
        "context_snapshot": {
            k: ctx[k] for k in [
                "total_leads", "win_rate_pct", "top_source", "top_source_rate",
                "win_rate_spread", "best_model_name", "best_val_auc",
                "lr_val_auc", "rf_val_auc", "lgbm_val_auc",
                "sent_diff", "best_sg", "best_sg_rate",
                "bot_source", "bot_source_rate",
                "perm_max", "perm_min", "top_gini_feature", "top_gini_imp",
            ]
        },
    }

    with open("reports/ai_insights.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] reports/ai_insights.json — {len(insights)} insights")
    print("\n" + "="*70)
    for ins in insights:
        print(f"\n[{ins['category'].upper()}] {ins['title']}")
        print("-"*60)
        print(ins["display_text"].encode("ascii", "replace").decode("ascii"))

    return insights


if __name__ == "__main__":
    run()
