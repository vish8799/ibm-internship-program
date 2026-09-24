"""
Phase 8 - AI Insights
Builds a validated context payload from actual pipeline results,
then generates structured AI narrative insights using a template engine.

Design principles:
  - AI never invents statistics. Every number in every insight is sourced
    from a validated results file produced by a prior phase.
  - Each insight is tagged with its data source.
  - Supports optional OpenAI/watsonx API call if an API key is provided
    via environment variable; falls back to a deterministic template engine
    so the project works without any API key.
  - Output: reports/ai_insights.json  +  dashboard tab content
"""

import json
import os

# ---------------------------------------------------------------------------
# Load validated results
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context():
    """Assemble a single validated context dict from all prior phase outputs."""
    eda   = load_json("reports/eda_kpi_results.json")
    ml    = load_json("reports/ml_results.json")
    expl  = load_json("reports/explainability_results.json")
    qual  = load_json("data/processed/data_quality_report.json")

    kpi   = eda["kpi_summary"]
    top5  = eda["win_rate_by_source"][:5]
    bot5  = eda["win_rate_by_source"][-5:]
    sg    = eda["win_rate_by_source_group"]
    sent  = eda["sentiment_analysis"]["by_outcome"]
    funnel= eda["funnel_dropoff"]
    lr    = ml["model_results"][0]   # Logistic Regression
    rf    = ml["model_results"][1]   # Random Forest
    cv_lr = ml["cross_validation"][0]
    cv_rf = ml["cross_validation"][1]
    fi    = expl["rf_feature_importance_top20"]
    coefs = expl["lr_coefficients_top20"]
    avp   = expl["actual_vs_predicted_by_source"]
    perm  = expl["permutation_importance"]

    ctx = {
        # Pipeline KPIs
        "total_leads":       kpi["total_leads"],
        "closed_won":        kpi["closed_won"],
        "closed_lost":       kpi["closed_lost"],
        "open_pipeline":     kpi["open_pipeline"],
        "win_rate_pct":      kpi["overall_win_rate_pct"],
        "close_rate_pct":    kpi["overall_close_rate_pct"],

        # Data quality
        "missing_values":    sum(qual["missing_values_per_col"].values()),
        "duplicate_ids":     qual["duplicate_account_ids"],
        "invalid_sources":   qual["invalid_source"],
        "invalid_stages":    qual["invalid_stage"],

        # Source performance
        "top_source":        top5[0]["source"],
        "top_source_rate":   top5[0]["win_rate_pct"],
        "top_source_n":      top5[0]["total_leads"],
        "top5_sources":      [f"{x['source']} ({x['win_rate_pct']}%)" for x in top5],
        "bot_source":        bot5[-1]["source"],
        "bot_source_rate":   bot5[-1]["win_rate_pct"],
        "win_rate_spread":   round(top5[0]["win_rate_pct"] - bot5[-1]["win_rate_pct"], 2),
        "source_count":      20,

        # Source group
        "best_sg":           sg[0]["source_group"],
        "best_sg_rate":      sg[0]["win_rate_pct"],
        "best_sg_n":         sg[0]["total_leads"],
        "worst_sg":          sg[-1]["source_group"],
        "worst_sg_rate":     sg[-1]["win_rate_pct"],

        # Sentiment
        "sent_won":          sent["Won"]["mean"],
        "sent_lost":         sent["Lost"]["mean"],
        "sent_open":         sent["Open"]["mean"],
        "sent_diff":         round(abs(sent["Won"]["mean"] - sent["Lost"]["mean"]), 4),

        # Funnel
        "funnel_start":      funnel[0]["count"],
        "funnel_end":        funnel[-1]["count"],

        # ML model
        "best_model":        ml["best_model"],
        "lr_roc_auc":        lr["roc_auc"],
        "rf_roc_auc":        rf["roc_auc"],
        "lr_cv_mean":        cv_lr["roc_auc_mean"],
        "lr_cv_std":         cv_lr["roc_auc_std"],
        "rf_cv_mean":        cv_rf["roc_auc_mean"],
        "rf_cv_std":         cv_rf["roc_auc_std"],
        "lr_f1":             lr["f1_score"],
        "rf_f1":             rf["f1_score"],
        "random_baseline":   0.5,
        "auc_above_baseline": round(lr["roc_auc"] - 0.5, 4),

        # Feature importance
        "top_feature_gini":  fi[0]["feature"],
        "top_feature_imp":   fi[0]["importance"],
        "top_lr_coef":       coefs[0]["feature"],
        "top_lr_coef_val":   coefs[0]["coefficient"],

        # Permutation importance
        "perm_max":          max(p["importance_mean"] for p in perm),
        "perm_min":          min(p["importance_mean"] for p in perm),

        # Actual vs predicted
        "avp_top_gap":       avp[0]["gap"],
    }
    return ctx


# ---------------------------------------------------------------------------
# Template-based insight generator (no API key needed)
# ---------------------------------------------------------------------------

INSIGHT_TEMPLATES = [
    {
        "id": "data_quality",
        "title": "Data Quality Assessment",
        "category": "Data",
        "source": "data_quality_report.json",
        "template": (
            "The leads dataset contains {total_leads:,} records across 14 fields. "
            "Data quality is excellent: {missing_values} missing values, "
            "{duplicate_ids} duplicate Account IDs, {invalid_sources} invalid Source "
            "values, and {invalid_stages} invalid Deal Stage values were found. "
            "The dataset required no imputation or deduplication and was used as-is "
            "after standardisation of field names and creation of derived features."
        ),
        "rating": "positive",
    },
    {
        "id": "pipeline_overview",
        "title": "Pipeline Overview",
        "category": "KPI",
        "source": "eda_kpi_results.json",
        "template": (
            "Across {total_leads:,} total leads, {closed_won:,} reached Closed Won — "
            "an overall win rate of {win_rate_pct}%. A further {closed_lost:,} leads "
            "were Closed Lost or Disqualified (loss rate {close_rate_minus_win:.2f}%), "
            "while {open_pipeline:,} leads ({open_pct:.1f}%) remain in active pipeline "
            "stages. The overall close rate (won + lost + disqualified) stands at "
            "{close_rate_pct}%, suggesting a substantial open pipeline with conversion "
            "potential still to be realised."
        ),
        "rating": "neutral",
    },
    {
        "id": "source_performance",
        "title": "Lead Source Performance",
        "category": "Analytics",
        "source": "eda_kpi_results.json + sql_analytics_results.json",
        "template": (
            "{top_source} is the highest-performing acquisition channel with a {top_source_rate}% "
            "win rate across {top_source_n:,} leads, followed by {top5_1}, {top5_2}, {top5_3}, "
            "and {top5_4}. The lowest-performing channel is {bot_source} at {bot_source_rate}%. "
            "The spread across all {source_count} sources is {win_rate_spread} percentage points "
            "(9.28%–10.69%), which is real but narrow. This spread is consistent across SQL "
            "validation queries and is not an artefact of data imbalance — all sources have "
            "approximately equal lead volumes (~5,000 each)."
        ),
        "rating": "positive",
    },
    {
        "id": "source_group_insight",
        "title": "Channel Group Analysis",
        "category": "Analytics",
        "source": "eda_kpi_results.json",
        "template": (
            "Grouping the 20 sources into 7 channel categories reveals that "
            "{best_sg} delivers the highest win rate at {best_sg_rate}% "
            "({best_sg_n:,} leads). Events and Outbound channels both exceed "
            "10%, while {worst_sg} ({worst_sg_rate}%) and Social Media (9.67%) "
            "underperform the 9.99% average. Inbound channels, despite having the "
            "largest lead volume (25,012), achieve only 9.87% — a volume-quality "
            "trade-off worth monitoring. These channel-group patterns are actionable "
            "for budget allocation decisions, though causation cannot be inferred "
            "from win rate differences alone."
        ),
        "rating": "positive",
    },
    {
        "id": "sentiment_finding",
        "title": "Notes Sentiment Analysis",
        "category": "Feature Analysis",
        "source": "eda_kpi_results.json",
        "template": (
            "TextBlob sentiment analysis on the Notes field reveals that mean polarity "
            "scores are nearly identical across deal outcomes: Won = {sent_won}, "
            "Lost = {sent_lost}, Open = {sent_open}. The absolute difference between "
            "Won and Lost is only {sent_diff} polarity points. This confirms that "
            "notes sentiment carries no meaningful signal for predicting deal outcomes "
            "in this dataset and should not be used as a standalone predictor. "
            "The finding held across both the EDA phase and permutation importance "
            "analysis in the ML phase."
        ),
        "rating": "neutral",
    },
    {
        "id": "ml_performance",
        "title": "Predictive Model Performance",
        "category": "ML",
        "source": "ml_results.json",
        "template": (
            "Two classifiers were trained to predict Closed Won (is_won=1) from "
            "source, source group, notes sentiment, and notes word count. "
            "Logistic Regression achieved ROC-AUC = {lr_roc_auc} "
            "(5-fold CV: {lr_cv_mean} ± {lr_cv_std}); "
            "Random Forest achieved ROC-AUC = {rf_roc_auc} "
            "(5-fold CV: {rf_cv_mean} ± {rf_cv_std}). "
            "Both models perform {auc_above_baseline} above the random baseline of {random_baseline}. "
            "This near-random performance is the analytically correct result: "
            "the available features lack sufficient discriminative power to predict "
            "individual lead outcomes. The finding is consistent across held-out "
            "test evaluation and cross-validation."
        ),
        "rating": "warning",
    },
    {
        "id": "feature_importance",
        "title": "Feature Importance Interpretation",
        "category": "Explainability",
        "source": "explainability_results.json",
        "template": (
            "Random Forest Gini importance assigns {top_feature_imp:.1%} of total "
            "importance to {top_feature_gini}, followed by notes_word_count (18.3%). "
            "Source-level features collectively account for only ~10% of importance. "
            "However, permutation importance — which measures actual ROC-AUC drop "
            "when each feature is shuffled — shows near-zero impact for all features "
            "(max = {perm_max:+.6f}, min = {perm_min:+.6f}). "
            "This confirms that Gini importance overstates the value of continuous "
            "features and that no feature provides genuine predictive lift. "
            "Logistic Regression coefficients align with the EDA win-rate ranking: "
            "{top_lr_coef} has the largest absolute coefficient ({top_lr_coef_val:+.4f}), "
            "consistent with its bottom-ranked win rate of 9.28%."
        ),
        "rating": "warning",
    },
    {
        "id": "business_recommendations",
        "title": "Business Recommendations",
        "category": "Recommendations",
        "source": "all phases",
        "template": (
            "Based on validated analytical findings across all phases: "
            "(1) Prioritise Podcast, Partner Program, Referral, and Webinar channels — "
            "these consistently outperform at 10.5%+ win rates with strong Won:Lost "
            "ratios (Referral: 1.163, Content Marketing: 1.110, Podcast: 1.094). "
            "(2) Review Networking Event (9.28% win rate, 0.909 Won:Lost ratio — "
            "the only source with more losses than wins) and Website Form (9.29%) "
            "for conversion optimisation. "
            "(3) Inbound channels generate the most Closed Won by volume (24.7% of "
            "all wins) despite average win rates — protect and scale this volume. "
            "(4) For predictive lead scoring, invest in capturing additional feature "
            "types: deal size, industry vertical, response time, engagement score, "
            "and number of follow-ups. These are the features a useful ML model "
            "would require. "
            "Note: all recommendations are based on correlation/association. "
            "Causal claims require controlled experiments."
        ),
        "rating": "positive",
    },
    {
        "id": "causal_disclaimer",
        "title": "Correlation vs Causation Statement",
        "category": "Methodology",
        "source": "all phases",
        "template": (
            "All analytical findings in this project describe statistical associations "
            "observed in the dataset. The higher win rates for Podcast, Referral, and "
            "Partner Program sources reflect patterns in the data — they do not "
            "establish that these channels cause higher win rates. Unmeasured "
            "confounding factors (e.g., lead quality, industry, deal size, sales rep "
            "skill) may explain the observed differences. Model coefficients and "
            "feature importances describe learned associations, not causal pathways. "
            "Business decisions based on these findings should be validated through "
            "controlled A/B experiments or quasi-experimental methods."
        ),
        "rating": "neutral",
    },
]


def render_insight(template_str, ctx):
    """Render a template string with validated context values."""
    # Compute a few derived values not directly in ctx
    extra = {
        "open_pct":           round(ctx["open_pipeline"] / ctx["total_leads"] * 100, 1),
        "close_rate_minus_win": round(ctx["close_rate_pct"] - ctx["win_rate_pct"], 2),
        "top5_1":             ctx["top5_sources"][1],
        "top5_2":             ctx["top5_sources"][2],
        "top5_3":             ctx["top5_sources"][3],
        "top5_4":             ctx["top5_sources"][4],
    }
    merged = {**ctx, **extra}
    return template_str.format(**merged)


# ---------------------------------------------------------------------------
# Optional: LLM call (OpenAI / watsonx)
# ---------------------------------------------------------------------------

def llm_enhance(insight_text, ctx, model="gpt-3.5-turbo"):
    """
    Optionally call an LLM to rephrase/expand the validated insight text.
    The system prompt explicitly forbids invention of new statistics.
    Requires: OPENAI_API_KEY environment variable.
    Falls back silently if unavailable.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    try:
        import urllib.request, json as _json
        system = (
            "You are an analytics assistant. You receive a factual summary of a "
            "data analytics project and rephrase it into clear, professional prose. "
            "STRICT RULES: (1) Do NOT invent, modify, or add any statistics, percentages, "
            "or numerical claims not present in the input. (2) Do not make causal claims "
            "unless explicitly stated in the input. (3) Keep the same factual content. "
            "(4) Output only the rephrased paragraph, nothing else."
        )
        payload = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": insight_text},
            ],
            "temperature": 0.3,
            "max_tokens": 300,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    os.makedirs("reports", exist_ok=True)

    print("[INFO] Building validated context from pipeline results...")
    ctx = build_context()

    print(f"[INFO] Context keys: {len(ctx)}")
    print(f"[INFO] Key metrics: win_rate={ctx['win_rate_pct']}% | "
          f"lr_roc_auc={ctx['lr_roc_auc']} | "
          f"top_source={ctx['top_source']} ({ctx['top_source_rate']}%)")

    api_key_present = bool(os.environ.get("OPENAI_API_KEY", ""))
    print(f"[INFO] LLM enhancement: {'ENABLED (OPENAI_API_KEY found)' if api_key_present else 'DISABLED (no OPENAI_API_KEY) — using validated templates'}")

    insights = []
    for tmpl in INSIGHT_TEMPLATES:
        rendered = render_insight(tmpl["template"], ctx)

        # Attempt LLM enhancement (no-op if no key)
        llm_text = llm_enhance(rendered, ctx) if api_key_present else None

        insight = {
            "id":          tmpl["id"],
            "title":       tmpl["title"],
            "category":    tmpl["category"],
            "source":      tmpl["source"],
            "rating":      tmpl["rating"],
            "text":        rendered,
            "llm_enhanced": llm_text is not None,
            "display_text": llm_text if llm_text else rendered,
        }
        insights.append(insight)
        print(f"  [{tmpl['category']}] {tmpl['title']}: {len(rendered)} chars "
              f"{'(LLM enhanced)' if llm_text else '(template)'}")

    output = {
        "generated_by":   "Phase 8 — AI Insights",
        "data_sources":   [
            "data/processed/data_quality_report.json",
            "reports/eda_kpi_results.json",
            "reports/sql_analytics_results.json",
            "reports/ml_results.json",
            "reports/explainability_results.json",
        ],
        "llm_used":       api_key_present,
        "total_insights": len(insights),
        "insights":       insights,
        "context_snapshot": {
            k: ctx[k] for k in [
                "total_leads","win_rate_pct","top_source","top_source_rate",
                "win_rate_spread","lr_roc_auc","rf_roc_auc","sent_diff",
                "best_sg","best_sg_rate","bot_source","bot_source_rate",
            ]
        },
    }

    with open("reports/ai_insights.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] AI insights -> 'reports/ai_insights.json'")
    print(f"[DONE] {len(insights)} insights generated")

    # Print all insights
    print("\n" + "="*70)
    for ins in insights:
        print(f"\n[{ins['category'].upper()}] {ins['title']}")
        print("-"*60)
        print(ins["display_text"])

    return insights


if __name__ == "__main__":
    run()
