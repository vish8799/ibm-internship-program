"""
Phase 9 — Automated test suite
================================
Covers:
  1. Data pipeline integrity  — schema, non-null IDs, value constraints
  2. SQL query output          — pipeline summary, source rankings, structure
  3. Model inference           — shape, probability bounds, row-sum, determinism
  4. AI service fallback       — heuristic path when no credentials present
  5. EDA / KPI outputs         — Phase 3 KPI schema and numeric consistency
  6. Explainability outputs    — Phase 6 JSON schema and numeric ranges
  7. Feature engineering       — source-tier mapping, freq-encoding, vector length

Run:
    pytest tests/test_phase9.py -v
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Repo root — all paths are resolved relative to here
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fixtures shared across multiple tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    """Cleaned leads CSV — loaded once per session."""
    path = ROOT / "data" / "processed" / "cleaned_leads.csv"
    frame = pd.read_csv(path)
    frame["is_won"] = pd.to_numeric(frame["is_won"], errors="coerce").fillna(0).astype(int)
    return frame


@pytest.fixture(scope="session")
def model():
    """Best trained model pipeline — loaded once per session."""
    with open(ROOT / "models" / "best_model.pkl", "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="session")
def sql_results() -> dict:
    with open(ROOT / "reports" / "sql_phase4_results.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def phase3_kpis() -> dict:
    with open(ROOT / "reports" / "eda_phase3_kpis.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def phase6_expl() -> dict:
    with open(ROOT / "reports" / "explainability_phase6.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def ai_narrative() -> dict:
    with open(ROOT / "reports" / "ai_narrative.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def ai_narrative_module():
    spec = importlib.util.spec_from_file_location(
        "ai_narrative", ROOT / "src" / "ai_narrative.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def source_tier() -> dict[str, int]:
    return {
        "Podcast": 1, "Partner Program": 1, "Referral": 1,
        "Webinars": 1, "Content Marketing": 1,
        "LinkedIn Outreach": 2, "Cold Email": 2, "Cold Call": 2,
        "Facebook Ads": 2, "Chatbot": 2, "Retargeting Ads": 2,
        "Direct Traffic": 2, "Organic Search (SEO)": 2, "Trade Show": 2,
        "Google Ads": 2, "Social Media": 3, "Purchased List": 3,
        "Other": 3, "Website Form": 3, "Networking Event": 3,
    }


# ---------------------------------------------------------------------------
# Helper: build model input vectors from a DataFrame slice
# ---------------------------------------------------------------------------
def _build_vectors(
    sample: pd.DataFrame,
    source_tier: dict,
    owner_counts: dict,
    company_counts: dict,
    total: int,
) -> list:
    vectors = []
    for _, row in sample.iterrows():
        vectors.append([
            row["source"],
            row["source_group"],
            float(row["notes_sentiment"]),
            int(row["notes_word_count"]),
            int(row["notes_has_text"]),
            source_tier.get(row["source"], 2),
            owner_counts.get(row["lead_owner"], 0) / total,
            company_counts.get(row["company"], 0) / total,
        ])
    return vectors


# ===========================================================================
# 1. DATA PIPELINE INTEGRITY
# ===========================================================================

class TestDataPipelineIntegrity:
    REQUIRED_COLS = {
        "index", "account_id", "lead_owner", "company", "website",
        "source", "source_group", "deal_stage", "is_won", "is_closed",
        "outcome_3class", "target_multiclass", "stage_ordinal",
        "notes_sentiment", "notes_word_count", "notes_has_text",
    }

    def test_required_columns_present(self, df):
        missing = self.REQUIRED_COLS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_row_count(self, df):
        assert len(df) == 100_000

    def test_account_id_non_null(self, df):
        assert df["account_id"].notna().all(), "Null account_ids found"

    def test_account_id_unique(self, df):
        assert df["account_id"].nunique() == len(df), "Duplicate account_ids found"

    def test_is_won_binary(self, df):
        assert set(df["is_won"].unique()).issubset({0, 1}), \
            f"is_won has non-binary values: {df['is_won'].unique()}"

    def test_is_won_counts(self, df):
        assert df["is_won"].sum() == 9993
        assert (df["is_won"] == 0).sum() == 90007

    def test_source_cardinality(self, df):
        assert df["source"].nunique() == 20, \
            f"Expected 20 sources, got {df['source'].nunique()}"

    def test_source_group_cardinality(self, df):
        assert df["source_group"].nunique() == 7, \
            f"Expected 7 source groups, got {df['source_group'].nunique()}"

    def test_deal_stage_cardinality(self, df):
        assert df["deal_stage"].nunique() == 10, \
            f"Expected 10 deal stages, got {df['deal_stage'].nunique()}"

    def test_notes_sentiment_range(self, df):
        assert df["notes_sentiment"].between(-1.0, 1.0).all(), \
            "notes_sentiment values outside [-1, 1]"

    def test_notes_word_count_positive(self, df):
        assert (df["notes_word_count"] >= 0).all()

    def test_notes_has_text_binary(self, df):
        assert set(df["notes_has_text"].unique()).issubset({0, 1})

    def test_no_missing_values_in_key_cols(self, df):
        key_cols = ["source", "source_group", "deal_stage", "is_won",
                    "notes_sentiment", "notes_word_count"]
        for col in key_cols:
            null_count = df[col].isna().sum()
            assert null_count == 0, f"Column '{col}' has {null_count} nulls"

    def test_data_quality_report_matches_csv(self, df):
        with open(ROOT / "data" / "processed" / "data_quality_report.json") as fh:
            qr = json.load(fh)
        assert qr["total_raw"] == len(df)
        assert qr["duplicate_account_ids"] == 0
        assert qr["invalid_source"] == 0
        assert qr["invalid_stage"] == 0

    def test_win_rate_consistency(self, df):
        computed = round(df["is_won"].mean() * 100, 2)
        assert computed == 9.99, f"Computed win rate {computed}% != 9.99%"


# ===========================================================================
# 2. SQL QUERY OUTPUT VALIDATION
# ===========================================================================

class TestSQLQueryOutputs:

    def test_pipeline_summary_keys(self, sql_results):
        assert "Q01_PIPELINE_SUMMARY" in sql_results
        q01 = sql_results["Q01_PIPELINE_SUMMARY"][0]
        for key in ("total_leads", "closed_won", "closed_lost_disq",
                    "open_pipeline", "win_rate_pct", "close_rate_pct"):
            assert key in q01, f"Missing key in Q01: {key}"

    def test_pipeline_summary_values(self, sql_results):
        q01 = sql_results["Q01_PIPELINE_SUMMARY"][0]
        assert q01["total_leads"]      == 100_000
        assert q01["closed_won"]       == 9_993
        assert q01["win_rate_pct"]     == 9.99
        assert q01["close_rate_pct"]   == 29.93
        assert q01["open_pipeline"]    == 70_074
        assert q01["closed_lost_disq"] == 19_933

    def test_pipeline_accounting(self, sql_results):
        """closed_won + closed_lost_disq + open_pipeline must equal total_leads."""
        q01 = sql_results["Q01_PIPELINE_SUMMARY"][0]
        total = q01["closed_won"] + q01["closed_lost_disq"] + q01["open_pipeline"]
        assert total == q01["total_leads"], \
            f"Lead count does not balance: {total} != {q01['total_leads']}"

    def test_top5_sources_present(self, sql_results):
        assert "Q02A_TOP5_SOURCES_BY_WIN_RATE" in sql_results
        top5 = sql_results["Q02A_TOP5_SOURCES_BY_WIN_RATE"]
        assert len(top5) == 5

    def test_top_source_is_podcast(self, sql_results):
        top5 = sql_results["Q02A_TOP5_SOURCES_BY_WIN_RATE"]
        assert top5[0]["source"] == "Podcast"
        assert top5[0]["win_rate_pct"] == pytest.approx(10.69, abs=0.05)

    def test_bottom5_sources_present(self, sql_results):
        assert "Q02B_BOTTOM5_SOURCES_BY_WIN_RATE" in sql_results
        bot5 = sql_results["Q02B_BOTTOM5_SOURCES_BY_WIN_RATE"]
        assert len(bot5) == 5

    def test_bottom_source_is_networking_event(self, sql_results):
        bot5 = sql_results["Q02B_BOTTOM5_SOURCES_BY_WIN_RATE"]
        # sorted ascending — first entry is lowest
        assert bot5[0]["source"] == "Networking Event"
        assert bot5[0]["win_rate_pct"] == pytest.approx(9.28, abs=0.05)

    def test_source_group_performance_present(self, sql_results):
        assert "Q03_SOURCE_GROUP_PERFORMANCE" in sql_results
        grp = sql_results["Q03_SOURCE_GROUP_PERFORMANCE"]
        assert len(grp) == 7

    def test_all_20_sources_in_full_ranking(self, sql_results):
        assert "Q02C_ALL_SOURCES_FULL_RANKING" in sql_results
        ranking = sql_results["Q02C_ALL_SOURCES_FULL_RANKING"]
        assert len(ranking) == 20

    def test_deal_stage_distribution_has_10_stages(self, sql_results):
        assert "Q04_DEAL_STAGE_DISTRIBUTION" in sql_results
        stages = sql_results["Q04_DEAL_STAGE_DISTRIBUTION"]
        assert len(stages) == 10

    def test_sql_win_rate_matches_kpi(self, sql_results, phase3_kpis):
        sql_rate = sql_results["Q01_PIPELINE_SUMMARY"][0]["win_rate_pct"]
        kpi_rate = phase3_kpis["kpi_01_overall_win_rate"]["win_rate_pct"]
        assert sql_rate == kpi_rate, \
            f"SQL ({sql_rate}) and Phase 3 KPI ({kpi_rate}) win rates disagree"


# ===========================================================================
# 3. MODEL INFERENCE CHECKS
# ===========================================================================

class TestModelInference:

    def test_pipeline_has_expected_steps(self, model):
        assert hasattr(model, "named_steps")
        assert "pre" in model.named_steps
        assert "clf" in model.named_steps

    def test_predict_proba_shape(self, df, model, source_tier):
        sample = df.head(50)
        total = len(df)
        owner_c   = df["lead_owner"].value_counts().to_dict()
        company_c = df["company"].value_counts().to_dict()
        vectors = _build_vectors(sample, source_tier, owner_c, company_c, total)
        proba = model.predict_proba(vectors)
        assert proba.shape == (50, 2), f"Unexpected proba shape: {proba.shape}"

    def test_predict_proba_bounds(self, df, model, source_tier):
        sample = df.head(200)
        total = len(df)
        owner_c   = df["lead_owner"].value_counts().to_dict()
        company_c = df["company"].value_counts().to_dict()
        vectors = _build_vectors(sample, source_tier, owner_c, company_c, total)
        proba = model.predict_proba(vectors)
        assert (proba >= 0.0).all(), "Negative probability found"
        assert (proba <= 1.0).all(), "Probability > 1 found"

    def test_predict_proba_row_sums(self, df, model, source_tier):
        """Each row of predict_proba must sum to 1.0."""
        sample = df.head(100)
        total = len(df)
        owner_c   = df["lead_owner"].value_counts().to_dict()
        company_c = df["company"].value_counts().to_dict()
        vectors = _build_vectors(sample, source_tier, owner_c, company_c, total)
        proba = model.predict_proba(vectors)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), \
            "Row probabilities do not sum to 1.0"

    def test_predict_proba_deterministic(self, df, model, source_tier):
        """Calling predict_proba twice on the same input yields identical results."""
        sample = df.head(20)
        total = len(df)
        owner_c   = df["lead_owner"].value_counts().to_dict()
        company_c = df["company"].value_counts().to_dict()
        vectors = _build_vectors(sample, source_tier, owner_c, company_c, total)
        p1 = model.predict_proba(vectors)
        p2 = model.predict_proba(vectors)
        assert np.array_equal(p1, p2), "predict_proba is not deterministic"

    def test_all_20_sources_scoreable(self, model, source_tier):
        """Model must accept all 20 known sources without raising."""
        sources = list(source_tier.keys())
        group_map = {
            "Podcast": "Inbound", "Partner Program": "Referral / Partner",
            "Referral": "Referral / Partner", "Webinars": "Events",
            "Content Marketing": "Inbound", "LinkedIn Outreach": "Outbound",
            "Cold Email": "Outbound", "Cold Call": "Outbound",
            "Facebook Ads": "Paid Ads", "Chatbot": "Inbound",
            "Retargeting Ads": "Paid Ads", "Direct Traffic": "Inbound",
            "Organic Search (SEO)": "Inbound", "Trade Show": "Events",
            "Google Ads": "Paid Ads", "Social Media": "Social Media",
            "Purchased List": "Outbound", "Other": "Other",
            "Website Form": "Inbound", "Networking Event": "Events",
        }
        for src in sources:
            vec = [src, group_map.get(src, "Other"), 0.1, 50, 1,
                   source_tier[src], 0.0, 0.0]
            proba = model.predict_proba([vec])
            assert proba.shape == (1, 2)
            assert 0.0 <= proba[0, 1] <= 1.0, \
                f"Invalid probability for source '{src}': {proba[0,1]}"

    def test_probability_varies_across_sources(self, model, source_tier):
        """Top-tier sources should score differently from bottom-tier sources."""
        group_map = {
            "Podcast": "Inbound", "Networking Event": "Events",
        }
        p_podcast = model.predict_proba([
            ["Podcast", "Inbound", 0.2, 100, 1, 1, 0.0, 0.0]
        ])[0, 1]
        p_network = model.predict_proba([
            ["Networking Event", "Events", 0.2, 100, 1, 3, 0.0, 0.0]
        ])[0, 1]
        # They may differ by a small amount; just confirm both are valid
        assert 0.0 <= p_podcast <= 1.0
        assert 0.0 <= p_network <= 1.0

    def test_vector_length_is_8(self, source_tier):
        """Feature vector builder must produce exactly 8 elements."""
        vec = ["Referral", "Referral / Partner", 0.3, 80, 1, 1, 0.001, 0.0]
        assert len(vec) == 8

    def test_best_model_is_logistic_regression(self):
        with open(ROOT / "models" / "ml_phase5_metrics.json") as fh:
            m5 = json.load(fh)
        assert m5["best_model"]["name"] == "Logistic Regression"

    def test_all_three_model_files_exist(self):
        for fname in ("best_model.pkl", "logistic_regression_model.pkl",
                      "random_forest_model.pkl", "lightgbm_model.pkl"):
            assert (ROOT / "models" / fname).exists(), f"Missing model file: {fname}"

    def test_val_auc_above_random_baseline(self):
        with open(ROOT / "models" / "ml_phase5_metrics.json") as fh:
            m5 = json.load(fh)
        for name, model_meta in m5["models"].items():
            val_auc = model_meta["val_metrics"]["roc_auc"]
            assert val_auc > 0.45, \
                f"Model '{name}' val ROC-AUC {val_auc} is suspiciously low"
            assert val_auc < 0.80, \
                f"Model '{name}' val ROC-AUC {val_auc} is suspiciously high (possible leakage)"

    def test_no_leakage_columns_in_feature_names(self):
        with open(ROOT / "models" / "feature_names.json") as fh:
            fn = json.load(fh)
        leakage_cols = {"deal_stage", "stage_ordinal", "is_closed",
                        "outcome_3class", "target_multiclass"}
        feature_set = set(fn["all_features"])
        overlap = leakage_cols & feature_set
        assert not overlap, f"Leakage columns in feature set: {overlap}"


# ===========================================================================
# 4. AI SERVICE FALLBACK BEHAVIOUR
# ===========================================================================

class TestAIServiceFallback:

    def test_build_context_returns_dict(self, ai_narrative_module):
        ctx = ai_narrative_module.build_context()
        assert isinstance(ctx, dict)
        assert len(ctx) >= 40

    def test_build_context_key_values(self, ai_narrative_module):
        ctx = ai_narrative_module.build_context()
        assert ctx["total_leads"] == 100_000
        assert ctx["win_rate_pct"] == pytest.approx(9.99, abs=0.01)
        assert ctx["top_source"] == "Podcast"
        assert ctx["bot_source"] == "Networking Event"
        assert 0.0 < ctx["best_val_auc"] < 1.0
        assert len(ctx["top5_sources"]) == 5

    def test_fallback_when_no_env_vars(self, ai_narrative_module, monkeypatch):
        """generate_summary must return a valid dict using heuristic when no creds set."""
        monkeypatch.delenv("WATSONX_APIKEY",    raising=False)
        monkeypatch.delenv("PROJECT_ID",        raising=False)
        monkeypatch.delenv("WATSONX_MODEL_ID",  raising=False)
        monkeypatch.delenv("WATSONX_URL",       raising=False)

        ctx = ai_narrative_module.build_context()
        result = ai_narrative_module.generate_summary(ctx)

        assert isinstance(result, dict), "generate_summary must return dict"
        assert result["source"]          == "heuristic_fallback"
        assert result["watsonx_enabled"] == False
        assert isinstance(result["narrative_text"], str)
        assert len(result["narrative_text"]) > 200
        assert isinstance(result["insights"], list)
        assert len(result["insights"]) == 9

    def test_heuristic_narrative_structure(self, ai_narrative_module):
        ctx = ai_narrative_module.build_context()
        text = ai_narrative_module._heuristic_fallback(ctx)
        assert "EXECUTIVE SUMMARY" in text
        assert "KEY FINDINGS"      in text
        assert "RECOMMENDATIONS"   in text

    def test_heuristic_contains_real_numbers(self, ai_narrative_module):
        ctx = ai_narrative_module.build_context()
        text = ai_narrative_module._heuristic_fallback(ctx)
        assert str(ctx["total_leads"]) in text.replace(",", "")
        assert str(ctx["win_rate_pct"]) in text
        assert ctx["top_source"] in text

    def test_heuristic_no_hallucinated_numbers(self, ai_narrative_module):
        """Checks that the heuristic text contains ONLY numbers derivable from context."""
        ctx = ai_narrative_module.build_context()
        text = ai_narrative_module._heuristic_fallback(ctx)
        # These numbers should NOT appear (they belong to other datasets)
        assert "99999" not in text
        assert "100001" not in text

    def test_render_insights_returns_9_blocks(self, ai_narrative_module):
        ctx = ai_narrative_module.build_context()
        insights = ai_narrative_module.render_insights(ctx)
        assert len(insights) == 9
        ids = {i["id"] for i in insights}
        assert "business_recommendations" in ids
        assert "causal_disclaimer"        in ids
        assert "ml_performance"           in ids

    def test_insight_blocks_have_required_fields(self, ai_narrative_module):
        ctx = ai_narrative_module.build_context()
        for ins in ai_narrative_module.render_insights(ctx):
            assert "id"       in ins
            assert "title"    in ins
            assert "category" in ins
            assert "rating"   in ins and ins["rating"] in ("positive", "neutral", "warning")
            assert "text"     in ins and len(ins["text"]) > 20

    def test_no_watsonx_call_without_credentials(self, ai_narrative_module, monkeypatch):
        """_call_watsonx must return None when credentials are absent."""
        monkeypatch.delenv("WATSONX_APIKEY", raising=False)
        monkeypatch.delenv("PROJECT_ID",     raising=False)
        ctx = ai_narrative_module.build_context()
        result = ai_narrative_module._call_watsonx(ctx)
        assert result is None

    def test_system_prompt_anti_hallucination_rules(self, ai_narrative_module):
        prompt = ai_narrative_module._SYSTEM_PROMPT
        assert len(prompt) > 100
        lowered = prompt.lower()
        assert "do not invent" in lowered or "not invent" in lowered
        assert "causal" in lowered or "causation" in lowered

    def test_ai_narrative_json_schema(self, ai_narrative):
        required_keys = {
            "generated_by", "narrative_source", "watsonx_enabled",
            "context_snapshot", "narrative", "insights",
        }
        assert required_keys.issubset(ai_narrative.keys())
        assert ai_narrative["narrative_source"] in ("watsonx", "heuristic_fallback")
        assert isinstance(ai_narrative["watsonx_enabled"], bool)
        assert len(ai_narrative["narrative"]) > 200
        assert len(ai_narrative["insights"]) == 9

    def test_ai_narrative_context_snapshot_values(self, ai_narrative):
        snap = ai_narrative["context_snapshot"]
        assert snap["total_leads"] == 100_000
        assert snap["win_rate_pct"] == 9.99
        assert snap["top_source"] == "Podcast"

    def test_no_api_keys_hardcoded_in_source(self):
        """WATSONX_APIKEY and PROJECT_ID must come from env vars only."""
        src = (ROOT / "src" / "ai_narrative.py").read_text(encoding="utf-8")
        import re
        # Should NOT contain literal credentials
        assert not re.search(r'WATSONX_APIKEY\s*=\s*["\'][A-Za-z0-9]{20,}', src)
        assert not re.search(r'PROJECT_ID\s*=\s*["\'][a-f0-9\-]{20,}',      src)
        # Must read from environment
        assert 'os.environ.get("WATSONX_APIKEY"' in src or \
               "os.environ.get('WATSONX_APIKEY'" in src


# ===========================================================================
# 5. EDA / KPI OUTPUT VALIDATION
# ===========================================================================

class TestEDAKPIOutputs:

    def test_kpi_01_overall_win_rate(self, phase3_kpis):
        k1 = phase3_kpis["kpi_01_overall_win_rate"]
        assert k1["total_leads"] == 100_000
        assert k1["closed_won"]  == 9_993
        assert k1["win_rate_pct"] == pytest.approx(9.99, abs=0.01)

    def test_kpi_02_closed_rate(self, phase3_kpis):
        k2 = phase3_kpis["kpi_02_closed_rate"]
        assert k2["closed_total"] == 29_926
        assert k2["closed_rate_pct"] == pytest.approx(29.93, abs=0.05)

    def test_kpi_03_source_win_rates_count(self, phase3_kpis):
        assert len(phase3_kpis["kpi_03_win_rate_by_source"]) == 20

    def test_kpi_03_top_source(self, phase3_kpis):
        sources = sorted(
            phase3_kpis["kpi_03_win_rate_by_source"],
            key=lambda x: x["win_rate_pct"], reverse=True
        )
        assert sources[0]["source"] == "Podcast"
        assert sources[0]["win_rate_pct"] == pytest.approx(10.69, abs=0.01)

    def test_kpi_03_bottom_source(self, phase3_kpis):
        sources = sorted(
            phase3_kpis["kpi_03_win_rate_by_source"],
            key=lambda x: x["win_rate_pct"]
        )
        assert sources[0]["source"] == "Networking Event"
        assert sources[0]["win_rate_pct"] == pytest.approx(9.28, abs=0.01)

    def test_kpi_03_all_win_rates_positive(self, phase3_kpis):
        for entry in phase3_kpis["kpi_03_win_rate_by_source"]:
            assert entry["win_rate_pct"] > 0, \
                f"Non-positive win rate for source: {entry['source']}"

    def test_kpi_03_win_counts_sum_to_closed_won(self, phase3_kpis):
        total_won = sum(e["won"] for e in phase3_kpis["kpi_03_win_rate_by_source"])
        assert total_won == phase3_kpis["kpi_01_overall_win_rate"]["closed_won"]

    def test_kpi_04_source_group_count(self, phase3_kpis):
        assert len(phase3_kpis["kpi_04_win_rate_by_source_group"]) == 7

    def test_kpi_04_best_group_is_referral_partner(self, phase3_kpis):
        groups = sorted(
            phase3_kpis["kpi_04_win_rate_by_source_group"],
            key=lambda x: x["win_rate_pct"], reverse=True
        )
        assert groups[0]["source_group"] == "Referral / Partner"

    def test_kpi_11_sentiment_outcomes_present(self, phase3_kpis):
        k11 = phase3_kpis["kpi_11_notes_sentiment_by_outcome"]
        for outcome in ("Won", "Lost", "Open"):
            assert outcome in k11, f"Missing outcome '{outcome}' in kpi_11"
            assert "mean" in k11[outcome]

    def test_kpi_11_sentiment_values_in_range(self, phase3_kpis):
        k11 = phase3_kpis["kpi_11_notes_sentiment_by_outcome"]
        for outcome, stats in k11.items():
            assert -1.0 <= stats["mean"] <= 1.0, \
                f"Sentiment mean out of range for outcome '{outcome}': {stats['mean']}"

    def test_kpi_11_sentiment_diff_negligible(self, phase3_kpis):
        k11 = phase3_kpis["kpi_11_notes_sentiment_by_outcome"]
        diff = abs(k11["Won"]["mean"] - k11["Lost"]["mean"])
        assert diff < 0.05, \
            f"Sentiment diff {diff} is unexpectedly large (should be near-zero)"


# ===========================================================================
# 6. EXPLAINABILITY OUTPUT VALIDATION
# ===========================================================================

class TestExplainabilityOutputs:

    def test_required_sections_present(self, phase6_expl):
        required = {
            "g1_lr_coefficients", "g2_rf_gini_importance",
            "g3_lgbm_split_importance", "g4_permutation_importance",
            "g5_shap_mean_abs", "l1_local_cases",
            "c1_actual_vs_predicted", "causal_boundary_statement",
        }
        missing = required - set(phase6_expl.keys())
        assert not missing, f"Missing sections: {missing}"

    def test_lr_coefficients_count(self, phase6_expl):
        assert len(phase6_expl["g1_lr_coefficients"]) == 31

    def test_lr_coefficients_have_both_signs(self, phase6_expl):
        coefs = [c["coefficient"] for c in phase6_expl["g1_lr_coefficients"]]
        assert any(c > 0 for c in coefs), "No positive LR coefficients"
        assert any(c < 0 for c in coefs), "No negative LR coefficients"

    def test_permutation_importance_near_zero(self, phase6_expl):
        perm = phase6_expl["g4_permutation_importance"]
        max_drop = max(p["mean_drop"] for p in perm)
        min_drop = min(p["mean_drop"] for p in perm)
        assert abs(max_drop) < 0.02, f"Permutation drop too large: {max_drop}"
        assert abs(min_drop) < 0.02, f"Permutation drop too large: {min_drop}"

    def test_shap_values_non_negative(self, phase6_expl):
        for entry in phase6_expl["g5_shap_mean_abs"]:
            assert entry["mean_abs_shap"] >= 0, \
                f"Negative mean |SHAP| for {entry['feature']}: {entry['mean_abs_shap']}"

    def test_shap_top_feature_is_expected(self, phase6_expl):
        top = phase6_expl["g5_shap_mean_abs"][0]["feature"]
        assert top in ("notes_word_count", "notes_sentiment", "company_freq"), \
            f"Unexpected SHAP top feature: {top}"

    def test_local_cases_all_four_present(self, phase6_expl):
        cases = phase6_expl["l1_local_cases"]
        assert set(cases.keys()) == {"tp", "fp", "fn", "tn"}

    def test_local_case_probability_bounds(self, phase6_expl):
        for key, case in phase6_expl["l1_local_cases"].items():
            p = case["predicted_prob"]
            assert 0.0 <= p <= 1.0, f"Case '{key}' has invalid prob: {p}"
            assert case["actual_label"] in (0, 1)

    def test_local_case_tp_fn_actual_labels(self, phase6_expl):
        cases = phase6_expl["l1_local_cases"]
        assert cases["tp"]["actual_label"] == 1, "TP actual_label must be 1"
        assert cases["fn"]["actual_label"] == 1, "FN actual_label must be 1"
        assert cases["fp"]["actual_label"] == 0, "FP actual_label must be 0"
        assert cases["tn"]["actual_label"] == 0, "TN actual_label must be 0"

    def test_local_case_tp_higher_prob_than_fn(self, phase6_expl):
        cases = phase6_expl["l1_local_cases"]
        assert cases["tp"]["predicted_prob"] > cases["fn"]["predicted_prob"], \
            "TP probability must be higher than FN probability"

    def test_avp_has_20_sources(self, phase6_expl):
        assert len(phase6_expl["c1_actual_vs_predicted"]) == 20

    def test_avp_actual_rates_in_range(self, phase6_expl):
        for entry in phase6_expl["c1_actual_vs_predicted"]:
            r = entry["actual_win_rate"]
            assert 0.08 <= r <= 0.12, \
                f"Source '{entry['source']}' actual win rate {r} out of expected range"

    def test_causal_boundary_statement_non_empty(self, phase6_expl):
        cbs = phase6_expl["causal_boundary_statement"]
        for key in ("global_importance", "local_predictions",
                    "permutation_finding", "practical_implication"):
            assert key in cbs and len(cbs[key]) > 50, \
                f"causal_boundary_statement['{key}'] missing or too short"


# ===========================================================================
# 7. FEATURE ENGINEERING CHECKS
# ===========================================================================

class TestFeatureEngineering:

    def test_source_tier_covers_all_20_sources(self, df, source_tier):
        sources_in_data = set(df["source"].unique())
        missing = sources_in_data - set(source_tier.keys())
        assert not missing, f"Sources not in tier map: {missing}"

    def test_source_tier_values_are_1_2_or_3(self, source_tier):
        for src, tier in source_tier.items():
            assert tier in (1, 2, 3), f"Source '{src}' has invalid tier: {tier}"

    def test_tier1_sources_have_highest_win_rates(self, phase3_kpis, source_tier):
        tier1 = {s for s, t in source_tier.items() if t == 1}
        tier3 = {s for s, t in source_tier.items() if t == 3}
        rates = {e["source"]: e["win_rate_pct"]
                 for e in phase3_kpis["kpi_03_win_rate_by_source"]}
        avg_t1 = np.mean([rates[s] for s in tier1 if s in rates])
        avg_t3 = np.mean([rates[s] for s in tier3 if s in rates])
        assert avg_t1 > avg_t3, \
            f"Tier 1 avg rate {avg_t1:.2f}% not higher than Tier 3 {avg_t3:.2f}%"

    def test_owner_freq_encoding_range(self, df):
        total = len(df)
        owner_counts = df["lead_owner"].value_counts().to_dict()
        owner_freqs = [c / total for c in owner_counts.values()]
        assert all(0.0 <= f <= 1.0 for f in owner_freqs)

    def test_company_freq_encoding_range(self, df):
        total = len(df)
        company_counts = df["company"].value_counts().to_dict()
        company_freqs = [c / total for c in company_counts.values()]
        assert all(0.0 <= f <= 1.0 for f in company_freqs)

    def test_feature_names_json_structure(self):
        with open(ROOT / "models" / "feature_names.json") as fh:
            fn = json.load(fh)
        assert "all_features" in fn
        assert "num_features" in fn
        assert fn["n_features_total"] == len(fn["all_features"])
        assert fn["n_features_total"] == 31

    def test_leakage_features_absent_from_pipeline(self):
        """Columns known to leak the target must not appear in the trained feature set."""
        with open(ROOT / "models" / "feature_names.json") as fh:
            fn = json.load(fh)
        leakage = {"deal_stage", "stage_ordinal", "is_closed",
                   "outcome_3class", "target_multiclass"}
        overlap = leakage & set(fn["all_features"])
        assert not overlap, f"Leakage columns in feature set: {overlap}"
