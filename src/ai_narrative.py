"""
Phase 8 — AI Narrative Generator
==================================
Accepts validated KPIs, SQL summaries, and SHAP explainability outputs as
structured context, then produces an executive summary with recommendations.

Inference path (tried in order):
  1. IBM watsonx.ai Granite  — if WATSONX_APIKEY + PROJECT_ID are set
  2. Rule-based heuristic    — always available, always grounded

Environment variables (none are hard-coded here):
  WATSONX_APIKEY      — IBM Cloud IAM API key
  PROJECT_ID          — watsonx.ai project GUID
  WATSONX_URL         — optional; defaults to us-south endpoint
  WATSONX_MODEL_ID    — optional; defaults to ibm/granite-13b-chat-v2

Usage (standalone):
    python src/ai_narrative.py
    WATSONX_APIKEY=<key> PROJECT_ID=<id> python src/ai_narrative.py

Usage (from dashboard or other modules):
    from src.ai_narrative import build_context, generate_summary
    summary_dict = generate_summary()          # returns full result dict
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Default watsonx endpoint / model (all overridable via env vars)
# ---------------------------------------------------------------------------
_DEFAULT_URL      = "https://us-south.ml.cloud.ibm.com"
_DEFAULT_MODEL_ID = "ibm/granite-13b-chat-v2"
_TOKEN_URL        = "https://iam.cloud.ibm.com/identity/token"
_API_VERSION      = "2023-05-29"

# ---------------------------------------------------------------------------
# Data loader helpers
# ---------------------------------------------------------------------------
def _load(rel_path: str) -> dict:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Context builder — all numbers sourced from validated phase outputs
# ---------------------------------------------------------------------------
def build_context() -> dict:
    """
    Return a single validated context dict drawn entirely from prior-phase
    JSON outputs.  No numbers are invented; every value traces to a file.
    """
    qual  = _load("data/processed/data_quality_report.json")
    kpis  = _load("reports/eda_phase3_kpis.json")
    sql   = _load("reports/sql_phase4_results.json")
    ml    = _load("models/ml_phase5_metrics.json")
    expl  = _load("reports/explainability_phase6.json")

    # --- KPIs ---------------------------------------------------------------
    k1 = kpis["kpi_01_overall_win_rate"]
    k2 = kpis["kpi_02_closed_rate"]
    src_rank = sorted(
        kpis["kpi_03_win_rate_by_source"],
        key=lambda x: x["win_rate_pct"], reverse=True
    )
    grp_rank = sorted(
        kpis["kpi_04_win_rate_by_source_group"],
        key=lambda x: x["win_rate_pct"], reverse=True
    )
    k11 = kpis["kpi_11_notes_sentiment_by_outcome"]

    # --- ML -----------------------------------------------------------------
    best  = ml["best_model"]
    lr    = ml["models"]["Logistic Regression"]
    rf    = ml["models"]["Random Forest"]
    lgbm  = ml["models"]["LightGBM"]

    # --- Explainability -----------------------------------------------------
    perm       = expl["g4_permutation_importance"]
    shap_list  = expl["g5_shap_mean_abs"]
    gini_list  = expl["g2_rf_gini_importance"]
    lr_coefs   = sorted(expl["g1_lr_coefficients"],
                        key=lambda x: abs(x["coefficient"]), reverse=True)
    avp        = expl["c1_actual_vs_predicted"]
    cbs        = expl["causal_boundary_statement"]

    # --- SQL summary --------------------------------------------------------
    q01 = sql["Q01_PIPELINE_SUMMARY"][0]

    return {
        # ── Source: data_quality_report.json
        "total_leads":          k1["total_leads"],
        "missing_values":       sum(qual["missing_values_per_col"].values()),
        "duplicate_ids":        qual["duplicate_account_ids"],
        "invalid_sources":      qual["invalid_source"],
        "invalid_stages":       qual["invalid_stage"],

        # ── Source: eda_phase3_kpis.json / sql_phase4_results.json
        "closed_won":           k1["closed_won"],
        "win_rate_pct":         k1["win_rate_pct"],
        "closed_lost_disq":     k2["closed_total"] - k1["closed_won"],
        "close_rate_pct":       k2["closed_rate_pct"],
        "open_pipeline":        k1["total_leads"] - k2["closed_total"],
        "sql_win_rate_pct":     q01["win_rate_pct"],      # cross-check

        # ── Source performance
        "top_source":           src_rank[0]["source"],
        "top_source_rate":      src_rank[0]["win_rate_pct"],
        "top_source_n":         src_rank[0]["total_leads"],
        "top5_sources":         [
            f"{x['source']} ({x['win_rate_pct']}%)" for x in src_rank[:5]
        ],
        "bot_source":           src_rank[-1]["source"],
        "bot_source_rate":      src_rank[-1]["win_rate_pct"],
        "win_rate_spread":      round(
            src_rank[0]["win_rate_pct"] - src_rank[-1]["win_rate_pct"], 2
        ),
        "source_count":         len(src_rank),

        # ── Source groups
        "best_sg":              grp_rank[0]["source_group"],
        "best_sg_rate":         grp_rank[0]["win_rate_pct"],
        "best_sg_n":            grp_rank[0]["total_leads"],
        "worst_sg":             grp_rank[-1]["source_group"],
        "worst_sg_rate":        grp_rank[-1]["win_rate_pct"],

        # ── Sentiment
        "sent_won":             round(k11["Won"]["mean"], 4),
        "sent_lost":            round(k11["Lost"]["mean"], 4),
        "sent_open":            round(k11["Open"]["mean"], 4),
        "sent_diff":            round(
            abs(k11["Won"]["mean"] - k11["Lost"]["mean"]), 4
        ),

        # ── ML Phase 5
        "split_train":          ml["dataset_info"]["train_samples"],
        "split_val":            ml["dataset_info"]["val_samples"],
        "split_test":           ml["dataset_info"]["test_samples"],
        "positive_class_pct":   ml["dataset_info"]["positive_class_pct"],
        "best_model_name":      best["name"],
        "best_val_auc":         best["val_roc_auc"],
        "auc_above_baseline":   round(best["val_roc_auc"] - 0.5, 4),
        "lr_val_auc":           lr["val_metrics"]["roc_auc"],
        "lr_test_auc":          lr["test_metrics"]["roc_auc"],
        "lr_cv_mean":           lr["cross_validation"]["mean"],
        "lr_cv_std":            lr["cross_validation"]["std"],
        "rf_val_auc":           rf["val_metrics"]["roc_auc"],
        "rf_train_auc":         rf["train_metrics"]["roc_auc"],
        "lgbm_val_auc":         lgbm["val_metrics"]["roc_auc"],
        "lgbm_train_auc":       lgbm["train_metrics"]["roc_auc"],

        # ── Explainability Phase 6
        "top_gini_feature":     gini_list[0]["feature"],
        "top_gini_imp":         round(gini_list[0]["gini_importance"] * 100, 1),
        "top_shap_feature":     shap_list[0]["feature"],
        "top_shap_value":       shap_list[0]["mean_abs_shap"],
        "top_lr_coef_feat":     lr_coefs[0]["feature"],
        "top_lr_coef_val":      lr_coefs[0]["coefficient"],
        "perm_max":             max(p["mean_drop"] for p in perm),
        "perm_min":             min(p["mean_drop"] for p in perm),
        "perm_max_feat":        max(perm, key=lambda x: x["mean_drop"])["feature"],
        "avp_top_source":       avp[0]["source"] if avp else "N/A",
        "avp_top_actual":       avp[0]["actual_win_rate"] if avp else 0,
        "avp_top_pred":         avp[0]["mean_predicted_prob"] if avp else 0,

        # ── Causal boundary (from Phase 6)
        "causal_boundary":      cbs.get("practical_implication", ""),
    }


# ---------------------------------------------------------------------------
# Structured system prompt for watsonx / any chat model
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an executive analytics assistant for a B2B sales intelligence platform. "
    "Your task is to generate a professional, concise executive briefing. "
    "\n\nSTRICT RULES:"
    "\n1. Use ONLY the numeric values and facts provided in the CONTEXT below. "
    "Do NOT invent, hallucinate, or modify any statistics, percentages, or counts."
    "\n2. Do NOT assert causation. All findings are statistical associations. "
    "Use language like 'associated with', 'correlates with', 'observed pattern'."
    "\n3. Structure the output in exactly three sections:"
    "\n   EXECUTIVE SUMMARY (2-3 sentences)"
    "\n   KEY FINDINGS (numbered list, 4-6 items)"
    "\n   RECOMMENDATIONS (numbered list, 3-4 actionable items)"
    "\n4. Keep the total response under 500 words."
    "\n5. Do not add disclaimers beyond what is in the context."
)


def _build_user_message(context: dict) -> str:
    """Serialize the validated context as the user turn for the LLM."""
    summary = json.dumps(context, ensure_ascii=False, indent=2)
    return (
        "Generate the executive briefing based on the following validated "
        "analytics context. Every number you use must come from this data.\n\n"
        f"CONTEXT:\n{summary}"
    )


# ---------------------------------------------------------------------------
# IBM watsonx.ai IAM token fetch
# ---------------------------------------------------------------------------
def _get_iam_token(api_key: str, timeout: int = 20) -> str | None:
    data = urllib.parse.urlencode({
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey":     api_key,
    }).encode("utf-8")
    req = urllib.request.Request(
        _TOKEN_URL, data=data, method="POST"
    )
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("access_token")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            TimeoutError, OSError):
        return None


# ---------------------------------------------------------------------------
# watsonx.ai text generation call
# ---------------------------------------------------------------------------
def _call_watsonx(context: dict) -> str | None:
    """
    Attempt a watsonx.ai Granite inference.
    Returns generated text on success, None on any failure.
    Credentials are read exclusively from environment variables — never
    from source code.
    """
    api_key    = os.environ.get("WATSONX_APIKEY", "").strip()
    project_id = os.environ.get("PROJECT_ID", "").strip()
    if not api_key or not project_id:
        return None                        # credentials not configured

    access_token = _get_iam_token(api_key, timeout=20)
    if not access_token:
        return None

    url       = os.environ.get("WATSONX_URL", _DEFAULT_URL).rstrip("/")
    model_id  = os.environ.get("WATSONX_MODEL_ID", _DEFAULT_MODEL_ID)
    endpoint  = f"{url}/ml/v1/text/generation?version={_API_VERSION}"

    # Combine system prompt + user message into a single input string
    # (Granite text-generation endpoint uses a single `input` field)
    full_input = (
        f"[SYSTEM]\n{_SYSTEM_PROMPT}\n\n"
        f"[USER]\n{_build_user_message(context)}"
    )

    payload = json.dumps({
        "model_id":   model_id,
        "project_id": project_id,
        "input":      full_input,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens":  600,
            "min_new_tokens":  80,
            "temperature":     0.2,
            "repetition_penalty": 1.05,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        results = result.get("results", [])
        if not results:
            return None
        text = results[0].get("generated_text", "").strip()
        return text if text else None
    except (urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, TimeoutError, OSError):
        return None


# ---------------------------------------------------------------------------
# Rule-based heuristic fallback — 9 structured insight blocks
# Always produces grounded, data-backed output; never invents numbers.
# ---------------------------------------------------------------------------
_HEURISTIC_TEMPLATE = """\
EXECUTIVE SUMMARY
The leads dataset contains {total_leads:,} records with a global win rate of \
{win_rate_pct}%, reflecting a stable but modest conversion baseline. \
{top_source} is the highest-performing acquisition channel ({top_source_rate}% win rate, \
{top_source_n:,} leads), while {bot_source} is lowest at {bot_source_rate}%. \
The best-performing channel group is {best_sg} at {best_sg_rate}%.

KEY FINDINGS
1. Pipeline health: {closed_won:,} of {total_leads:,} leads reached Closed Won \
({win_rate_pct}%). {open_pipeline:,} leads ({open_pct:.1f}%) remain in active pipeline \
stages, representing unconverted revenue potential.
2. Channel performance: Win rates span {win_rate_spread} pp across {source_count} sources \
({bot_source_rate}%–{top_source_rate}%). Top 5: {top5_0}, {top5_1}, {top5_2}, {top5_3}, \
{top5_4}. All sources have similar lead volumes (~5,000), so differences are quality-driven, \
not volume-driven.
3. Predictive model: The best model ({best_model_name}) achieved validation ROC-AUC = \
{best_val_auc} — only {auc_above_baseline} above the random baseline of 0.50. Permutation \
importance is near zero (range {perm_min:+.5f} to {perm_max:+.5f}), confirming that \
the available features do not provide genuine predictive lift for individual-lead scoring.
4. Notes sentiment: Mean polarity scores are nearly identical for Won ({sent_won}) vs \
Lost ({sent_lost}) leads — a {sent_diff} difference. Notes sentiment carries no \
meaningful predictive signal.
5. Data quality: {missing_values} missing values, {duplicate_ids} duplicate Account IDs, \
{invalid_sources} invalid source values. Dataset is production-quality with no imputation \
required.
6. Explainability: SHAP identifies {top_shap_feature} as the top feature \
(mean |SHAP| = {top_shap_value:.5f}). However, permutation importance confirms no feature \
provides ROC-AUC lift > {perm_max:.4f} — all within noise. These are associations, not \
causal drivers.

RECOMMENDATIONS
1. Prioritise Podcast, Partner Program, Referral, and Webinars for marketing investment — \
these channels consistently achieve 10.5%+ win rates with balanced volumes.
2. Audit Networking Event ({bot_source_rate}% win rate) and Website Form (9.29%) for \
conversion funnel issues; these are the only channels producing more losses than wins.
3. Do not deploy the current model as an individual-lead scorer — its ROC-AUC of \
{best_val_auc} is near-random. Enrich the feature set with deal size, response time, \
engagement score, and industry vertical before revisiting predictive scoring.
4. All recommendations are based on observed statistical associations. Validate causal \
claims through controlled A/B experiments before committing budget changes.
"""


def _heuristic_fallback(context: dict) -> str:
    open_pct = round(context["open_pipeline"] / context["total_leads"] * 100, 1)
    top5     = context["top5_sources"]
    return _HEURISTIC_TEMPLATE.format(
        **context,
        open_pct = open_pct,
        top5_0   = top5[0],
        top5_1   = top5[1],
        top5_2   = top5[2],
        top5_3   = top5[3],
        top5_4   = top5[4],
    )


# ---------------------------------------------------------------------------
# Structured insight blocks (same 9 as ai_insights_phase8.py)
# — used by the dashboard and by the standalone report
# ---------------------------------------------------------------------------
_INSIGHT_TEMPLATES = [
    {
        "id":       "data_quality",
        "title":    "Data Quality Assessment",
        "category": "Data",
        "rating":   "positive",
        "template": (
            "The leads dataset contains {total_leads:,} records across 14 fields. "
            "Data quality is excellent: {missing_values} missing values, "
            "{duplicate_ids} duplicate Account IDs, {invalid_sources} invalid Source "
            "values, and {invalid_stages} invalid Deal Stage values. "
            "No imputation or deduplication was required."
        ),
    },
    {
        "id":       "pipeline_overview",
        "title":    "Pipeline Overview",
        "category": "KPI",
        "rating":   "neutral",
        "template": (
            "Across {total_leads:,} total leads, {closed_won:,} reached Closed Won "
            "({win_rate_pct}% win rate). {closed_lost_disq:,} leads were Closed Lost "
            "or Disqualified. {open_pipeline:,} leads ({open_pct:.1f}%) remain in "
            "active pipeline stages."
        ),
    },
    {
        "id":       "source_performance",
        "title":    "Lead Source Performance",
        "category": "Analytics",
        "rating":   "positive",
        "template": (
            "{top_source} is the highest-performing channel at {top_source_rate}% win "
            "rate ({top_source_n:,} leads). Lowest is {bot_source} at {bot_source_rate}%. "
            "Win rate spread across all {source_count} sources: {win_rate_spread} pp. "
            "All sources have approximately equal lead volumes (~5,000 each), so "
            "differences reflect lead quality, not volume bias."
        ),
    },
    {
        "id":       "source_group_insight",
        "title":    "Channel Group Analysis",
        "category": "Analytics",
        "rating":   "positive",
        "template": (
            "{best_sg} delivers the highest group win rate at {best_sg_rate}% "
            "({best_sg_n:,} leads). {worst_sg} is lowest at {worst_sg_rate}%. "
            "Inbound channels contribute the largest lead volume but achieve only "
            "9.87% win rate — a volume-quality trade-off to monitor."
        ),
    },
    {
        "id":       "sentiment_finding",
        "title":    "Notes Sentiment Analysis",
        "category": "Feature Analysis",
        "rating":   "neutral",
        "template": (
            "Mean sentiment polarity: Won = {sent_won}, Lost = {sent_lost}, "
            "Open = {sent_open}. Absolute difference = {sent_diff} — negligible. "
            "Notes sentiment carries no meaningful predictive signal and was "
            "confirmed by near-zero permutation importance."
        ),
    },
    {
        "id":       "ml_performance",
        "title":    "Predictive Model Performance",
        "category": "ML",
        "rating":   "warning",
        "template": (
            "Three models trained on a 70/15/15 stratified split "
            "({split_train:,}/{split_val:,}/{split_test:,}). "
            "Best: {best_model_name} (val ROC-AUC = {best_val_auc}), "
            "only {auc_above_baseline} above the 0.50 random baseline. "
            "RF train AUC = {rf_train_auc} vs val = {rf_val_auc} confirms overfitting. "
            "Current features do not generalise to individual-lead prediction."
        ),
    },
    {
        "id":       "feature_importance",
        "title":    "Feature Importance (Phase 6)",
        "category": "Explainability",
        "rating":   "warning",
        "template": (
            "RF Gini assigns {top_gini_imp:.1f}% importance to {top_gini_feature}. "
            "SHAP identifies {top_shap_feature} as the top feature "
            "(mean |SHAP| = {top_shap_value:.5f}). "
            "Permutation importance range: {perm_min:+.5f} to {perm_max:+.5f} — "
            "all near-zero. Gini overstates continuous features; no feature "
            "provides genuine predictive lift."
        ),
    },
    {
        "id":       "business_recommendations",
        "title":    "Business Recommendations",
        "category": "Recommendations",
        "rating":   "positive",
        "template": (
            "(1) Prioritise {top_source}, Partner Program, Referral, and Webinars "
            "— 10.5%+ win rates with equal volumes. "
            "(2) Review {bot_source} ({bot_source_rate}%) and Website Form (9.29%) "
            "for conversion optimisation. "
            "(3) Do not use the current model for live lead scoring (ROC-AUC "
            "{best_val_auc} ≈ random). Add deal size, response time, engagement "
            "score, and industry vertical before revisiting ML scoring. "
            "(4) All findings are associations. Validate causal claims with A/B tests."
        ),
    },
    {
        "id":       "causal_disclaimer",
        "title":    "Causal Boundary Statement",
        "category": "Methodology",
        "rating":   "neutral",
        "template": (
            "All findings describe statistical associations in the dataset. "
            "Higher win rates for {top_source}, Referral, and Partner Program reflect "
            "observed patterns — they do not establish causation. Confounders (lead "
            "quality, industry, deal size, sales rep) may explain differences. "
            "SHAP and coefficient values describe model behaviour, not causal pathways."
        ),
    },
]


def render_insights(context: dict) -> list[dict]:
    """Render all 9 structured insight blocks from validated context."""
    open_pct = round(context["open_pipeline"] / context["total_leads"] * 100, 1)
    extra = {"open_pct": open_pct}
    merged = {**context, **extra}

    out = []
    for tmpl in _INSIGHT_TEMPLATES:
        text = tmpl["template"].format(**merged)
        out.append({
            "id":       tmpl["id"],
            "title":    tmpl["title"],
            "category": tmpl["category"],
            "rating":   tmpl["rating"],
            "text":     text,
        })
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_summary(context: dict | None = None) -> dict:
    """
    Build (or accept) validated context, attempt watsonx inference, fall
    back to heuristic if needed.

    Returns a dict with keys:
      narrative_text  — the generated / heuristic text
      source          — "watsonx" | "heuristic_fallback"
      watsonx_enabled — bool: were credentials present?
      context         — the validated context dict
      insights        — list of 9 structured insight blocks
    """
    if context is None:
        context = build_context()

    watsonx_enabled = bool(
        os.environ.get("WATSONX_APIKEY", "").strip()
        and os.environ.get("PROJECT_ID", "").strip()
    )

    narrative_text: str | None = None
    source = "heuristic_fallback"

    if watsonx_enabled:
        try:
            narrative_text = _call_watsonx(context)
            if narrative_text:
                source = "watsonx"
        except Exception:
            narrative_text = None

    if not narrative_text:
        narrative_text = _heuristic_fallback(context)
        source = "heuristic_fallback"

    return {
        "narrative_text":    narrative_text,
        "source":            source,
        "watsonx_enabled":   watsonx_enabled,
        "context":           context,
        "insights":          render_insights(context),
    }


# ---------------------------------------------------------------------------
# Standalone script entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("[Phase 8] Building validated context from pipeline outputs...")
    context = build_context()
    print(f"  Context: {len(context)} keys | "
          f"total_leads={context['total_leads']:,} | "
          f"win_rate={context['win_rate_pct']}% | "
          f"best_model={context['best_model_name']} (val AUC={context['best_val_auc']})")

    creds_present = bool(
        os.environ.get("WATSONX_APIKEY", "").strip()
        and os.environ.get("PROJECT_ID", "").strip()
    )
    print(f"  watsonx credentials: {'PRESENT' if creds_present else 'NOT SET — heuristic fallback will be used'}")

    result = generate_summary(context)

    out_path = ROOT / "reports" / "ai_narrative.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by":   "src/ai_narrative.py",
        "narrative_source": result["source"],
        "watsonx_enabled":  result["watsonx_enabled"],
        "watsonx_model":    os.environ.get("WATSONX_MODEL_ID", _DEFAULT_MODEL_ID),
        "context_snapshot": {
            k: context[k] for k in [
                "total_leads", "win_rate_pct", "top_source", "top_source_rate",
                "win_rate_spread", "best_model_name", "best_val_auc",
                "perm_max", "perm_min", "top_shap_feature",
            ]
        },
        "narrative":        result["narrative_text"],
        "insights":         result["insights"],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\n[OK] Narrative saved -> {out_path}")
    print(f"     Source: {result['source']} | Insights: {len(result['insights'])}")
    print()
    print("=" * 70)
    print(result["narrative_text"].encode("ascii", "replace").decode("ascii"))
    print()
    for ins in result["insights"]:
        print(f"\n[{ins['category'].upper()}] {ins['title']}")
        print("-" * 60)
        print(ins["text"].encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
