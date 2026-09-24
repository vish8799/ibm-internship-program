# Lead Intelligence Platform

A production-grade, end-to-end B2B lead analytics and AI-narrative platform built across nine phases — from raw data ingestion to an interactive Streamlit dashboard with IBM watsonx.ai integration.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Dataset Description](#3-dataset-description)
4. [Repository Structure](#4-repository-structure)
5. [Setup & Execution Guide](#5-setup--execution-guide)
6. [Phase-by-Phase Execution](#6-phase-by-phase-execution)
7. [KPI & Analytics Findings](#7-kpi--analytics-findings)
8. [ML Performance Benchmarks](#8-ml-performance-benchmarks)
9. [Explainability Summary](#9-explainability-summary)
10. [AI Integration Architecture](#10-ai-integration-architecture)
11. [Interactive Dashboard](#11-interactive-dashboard)
12. [Testing](#12-testing)
13. [Project Limitations & Ethical Considerations](#13-project-limitations--ethical-considerations)
14. [Glossary](#14-glossary)

---

## 1. Project Overview

The Lead Intelligence Platform ingests a 100,000-record B2B CRM export, runs structured analytics across nine sequential phases, and surfaces actionable channel-investment recommendations. A Streamlit dashboard provides live filtering, interactive scoring, and AI-generated executive summaries backed by IBM watsonx.ai Granite (with a deterministic heuristic fallback when credentials are absent).

**Key capabilities:**

| Capability | Technology |
|---|---|
| Data quality & preprocessing | Pandas, custom validators |
| EDA & KPI computation | Pandas, Matplotlib, Seaborn |
| SQL analytics (20 queries) | SQLite via sqlite3 |
| Predictive modelling | scikit-learn (LR, RF), LightGBM |
| Model explainability | SHAP, Permutation Importance |
| AI narrative generation | IBM watsonx.ai Granite / heuristic fallback |
| Interactive dashboard | Streamlit 1.64, Plotly |
| Automated tests | pytest (83 tests, 7 test classes) |

---

## 2. Architecture

```
leads-100000.csv (raw)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Phase 1 – Inspection       src/phase1_inspect.py │
│  Phase 2 – Preprocessing    src/data_preprocessing.py │
│           └─► data/processed/cleaned_leads.csv   │
└─────────────────────────────────────────────────┘
        │
        ├─► Phase 3 – EDA & KPIs        reports/eda_phase3_kpis.json
        ├─► Phase 4 – SQL Analytics     reports/sql_phase4_results.json
        ├─► Phase 5 – ML Training       models/*.pkl  models/ml_phase5_metrics.json
        ├─► Phase 6 – Explainability    reports/explainability_phase6.json
        ├─► Phase 7 – Dashboard         dashboard/streamlit_app.py
        ├─► Phase 8 – AI Narrative      reports/ai_narrative.json
        └─► Phase 9 – Tests & Docs      tests/test_phase9.py  README.md
```

Every phase reads only the outputs of prior phases. No phase mutates upstream files. All paths are relative to the repository root and resolved via `pathlib.Path(__file__).resolve().parents[n]`.

---

## 3. Dataset Description

| Property | Value |
|---|---|
| Source file | `leads-100000.csv` |
| Rows | 100,000 |
| Raw columns | 14 (Index, Account Id, Lead Owner, First Name, Last Name, Company, Phone 1, Phone 2, Email 1, Email 2, Website, Source, Deal Stage, Notes) |
| Derived columns | 8 (`source_group`, `is_won`, `is_closed`, `outcome_3class`, `stage_ordinal`, `notes_sentiment`, `notes_word_count`, `notes_has_text`) |
| Missing values | 0 |
| Duplicate Account IDs | 0 |
| Invalid Source values | 0 |
| Invalid Deal Stage values | 0 |
| Lead sources | 20 (Podcast, Partner Program, Referral, Webinars, Content Marketing, LinkedIn Outreach, Cold Email, Cold Call, Facebook Ads, Chatbot, Retargeting Ads, Direct Traffic, Organic Search (SEO), Trade Show, Google Ads, Social Media, Purchased List, Other, Website Form, Networking Event) |
| Source groups | 7 (Referral / Partner, Events, Outbound, Inbound, Paid Ads, Social Media, Other) |
| Deal stages | 10 (New Lead, Contacted, Qualified, Proposal Sent, Negotiation, Re-engagement, On Hold, Disqualified, Closed Lost, Closed Won) |
| Target: `is_won` | 9,993 positive (9.99%), 90,007 negative (90.01%) |
| Class imbalance ratio | ~9:1 |

### Notes Sentiment
TextBlob polarity analysis applied to the Notes field. Values range from −1 (very negative) to +1 (very positive). All 100,000 records contain text (`notes_has_text = 1` throughout). Mean polarity is near-neutral across all outcomes (Won: 0.0575, Lost: 0.0646, Open: 0.0621).

---

## 4. Repository Structure

```
.
├── leads-100000.csv               # Original raw dataset (do not modify)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/
│   ├── raw/leads-100000.csv       # Symlink / copy for Phase 2 input
│   └── processed/
│       ├── cleaned_leads.csv      # Output of Phase 2 preprocessing
│       ├── data_quality_report.json
│       ├── preprocessing_report.json
│       ├── leads.db               # SQLite database (Phase 4)
│       └── pii_vault.csv          # Pseudonymised PII (Phase 2)
│
├── models/
│   ├── best_model.pkl             # Logistic Regression (selected by val AUC)
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── lightgbm_model.pkl
│   ├── feature_names.json         # 31 feature names + 8 raw input names
│   └── ml_phase5_metrics.json     # Full metrics for all 3 models
│
├── reports/
│   ├── eda_phase3_kpis.json       # 12 KPI blocks from Phase 3
│   ├── sql_phase4_results.json    # 20 SQL query results from Phase 4
│   ├── explainability_phase6.json # G1–G5 global + L1 local SHAP results
│   ├── ai_narrative.json          # Phase 8 executive summary
│   ├── ai_insights.json           # Phase 8 structured insight blocks
│   └── figures/                   # 22 PNG figures (EDA + explainability)
│
├── src/
│   ├── phase1_inspect.py          # Raw data inspection
│   ├── data_preprocessing.py      # Phase 2: cleaning + feature engineering
│   ├── eda_phase3.py              # Phase 3: KPI computation + figures
│   ├── sql_phase4.py              # Phase 4: SQLite analytics
│   ├── ml_phase5.py               # Phase 5: model training + evaluation
│   ├── explainability_phase6.py   # Phase 6: SHAP + permutation importance
│   ├── ai_narrative.py            # Phase 8: watsonx.ai + heuristic fallback
│   ├── ai_insights_phase8.py      # Phase 8: structured insight generator
│   ├── validate_phase[2-8].py     # Per-phase smoke tests
│   └── __pycache__/
│
├── dashboard/
│   └── streamlit_app.py           # Phase 7/8 interactive dashboard (1,058 lines)
│
└── tests/
    └── test_phase9.py             # 83 automated pytest tests (7 classes)
```

---

## 5. Setup & Execution Guide

### Prerequisites

- Python 3.10 or later
- pip

### Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` installs: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `streamlit`, `plotly`, `shap`, `textblob`.

### Run the dashboard (Phase 7/8)

```bash
streamlit run dashboard/streamlit_app.py
```

Opens at `http://localhost:8501`. No environment variables are required for the heuristic-fallback mode.

### Enable IBM watsonx.ai narrative generation (Phase 8)

```bash
export WATSONX_APIKEY="your-ibm-cloud-iam-key"
export PROJECT_ID="your-watsonx-project-guid"
# Optional overrides:
# export WATSONX_URL="https://eu-de.ml.cloud.ibm.com"
# export WATSONX_MODEL_ID="ibm/granite-13b-chat-v2"

python src/ai_narrative.py
# Or click "Regenerate AI Narrative" inside the dashboard AI Insights tab
```

### Run the automated test suite

```bash
pytest tests/test_phase9.py -v
# Expected: 83 passed in ~5 s
```

### Run individual phase scripts

```bash
python src/data_preprocessing.py      # Phase 2
python src/eda_phase3.py              # Phase 3
python src/sql_phase4.py              # Phase 4
python src/ml_phase5.py               # Phase 5
python src/explainability_phase6.py   # Phase 6
python src/ai_narrative.py            # Phase 8
```

### Run per-phase validation scripts

```bash
python src/validate_phase2.py
python src/validate_phase3.py
python src/validate_phase4.py
python src/validate_phase5.py
python src/validate_phase6.py
python src/validate_phase7.py
python src/validate_phase8.py
```

---

## 6. Phase-by-Phase Execution

| Phase | Script | Output | Status |
|---|---|---|---|
| 1 – Inspection | `src/phase1_inspect.py` | Console report | ✅ |
| 2 – Preprocessing | `src/data_preprocessing.py` | `data/processed/cleaned_leads.csv` | ✅ |
| 3 – EDA & KPIs | `src/eda_phase3.py` | `reports/eda_phase3_kpis.json` + 12 figures | ✅ |
| 4 – SQL Analytics | `src/sql_phase4.py` | `reports/sql_phase4_results.json` + SQLite DB | ✅ |
| 5 – ML Training | `src/ml_phase5.py` | `models/*.pkl` + `ml_phase5_metrics.json` | ✅ |
| 6 – Explainability | `src/explainability_phase6.py` | `reports/explainability_phase6.json` + 10 figures | ✅ |
| 7 – Dashboard | `streamlit run dashboard/streamlit_app.py` | Live Streamlit app | ✅ |
| 8 – AI Narrative | `src/ai_narrative.py` | `reports/ai_narrative.json` | ✅ |
| 9 – Tests & Docs | `pytest tests/test_phase9.py` | 83 tests, 0 failures | ✅ |

---

## 7. KPI & Analytics Findings

### Overall Pipeline

| Metric | Value |
|---|---|
| Total leads | 100,000 |
| Closed Won | 9,993 (9.99%) |
| Closed Lost + Disqualified | 19,933 (19.93%) |
| Open pipeline | 70,074 (70.07%) |
| Overall close rate | 29.93% |

### Win Rate by Lead Source (Top 5 / Bottom 5)

| Rank | Source | Leads | Won | Win Rate |
|---|---|---|---|---|
| 1 | Podcast | 5,134 | 549 | **10.69%** |
| 2 | Partner Program | 5,076 | 541 | **10.66%** |
| 3 | Referral | 4,994 | 529 | **10.59%** |
| 4 | Webinars | 4,979 | 526 | **10.56%** |
| 5 | Content Marketing | 4,935 | 516 | **10.46%** |
| … | … | … | … | … |
| 16 | Social Media | 4,996 | 483 | 9.67% |
| 17 | Purchased List | 5,015 | 479 | 9.55% |
| 18 | Other | 4,872 | 464 | 9.52% |
| 19 | Website Form | 5,069 | 471 | 9.29% |
| 20 | Networking Event | 4,870 | 452 | **9.28%** |

Win rate spread across all 20 sources: **1.41 percentage points** (9.28%–10.69%). All sources have approximately equal lead volumes (~5,000), so the rate differences reflect lead quality rather than volume bias.

### Win Rate by Source Group

| Source Group | Leads | Won | Win Rate |
|---|---|---|---|
| Referral / Partner | 10,070 | 1,070 | **10.63%** |
| Events | 20,062 | 2,020 | 10.07% |
| Outbound | 19,958 | 2,007 | 10.06% |
| Inbound | 25,012 | 2,468 | 9.87% |
| Paid Ads | 15,030 | 1,481 | 9.85% |
| Social Media | 4,996 | 483 | 9.67% |
| Other | 4,872 | 464 | **9.52%** |

Inbound is the largest volume group (25.0% of all leads) but achieves only 9.87% — a volume-quality trade-off for channel investment decisions.

### Notes Sentiment (Cross-Outcome)

| Outcome | Mean Polarity | Std |
|---|---|---|
| Won | 0.0575 | — |
| Lost | 0.0646 | — |
| Open | 0.0621 | — |

Absolute Won–Lost difference: **0.0071** — statistically negligible. Notes sentiment is not a useful predictor.

> **Causal boundary:** These findings are statistical associations observed in the dataset. Higher win rates for Podcast, Partner Program, and Referral sources do not prove those channels cause higher win rates. Unmeasured confounders (lead quality, industry, deal size, sales rep assignment) may fully explain the observed differences.

---

## 8. ML Performance Benchmarks

### Training Configuration

| Setting | Value |
|---|---|
| Target | `is_won` (binary: 1 = Closed Won) |
| Split | 70% train / 15% validation / 15% test (stratified, seed=42) |
| Train samples | 69,999 |
| Validation samples | 15,001 |
| Test samples | 15,000 |
| Class imbalance handling | `class_weight='balanced'` |
| Features | 8 raw → 31 post-OHE (20 source OHE + 5 source_group OHE + 6 numeric) |
| Positive class prevalence | 9.99% |
| Leakage prevention | `deal_stage`, `stage_ordinal`, `is_closed`, `outcome_3class`, `target_multiclass` excluded |

### Model Results

| Model | Val ROC-AUC | Test ROC-AUC | CV Mean ± Std | Train ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** ✓ | **0.5113** | 0.5166 | 0.5003 ± 0.0054 | 0.5131 |
| Random Forest | 0.5050 | 0.5061 | 0.5063 ± 0.0084 | **0.6525** |
| LightGBM | 0.4963 | 0.4998 | 0.5085 ± 0.0047 | **0.6888** |

**Best model selected:** Logistic Regression (highest validation ROC-AUC = 0.5113, saved to `models/best_model.pkl`).

Random Forest and LightGBM exhibit significant train-to-validation ROC-AUC drop (RF: 0.6525 → 0.5050; LGBM: 0.6888 → 0.4963), confirming overfitting on the available feature set. No model achieves meaningful generalisation beyond the 0.50 random baseline.

### Diagnosis

All three models perform within noise range of a random classifier (ROC-AUC ≈ 0.50–0.51). Permutation importance (Phase 6) confirms this: shuffling any feature causes a ROC-AUC change between −0.0097 and +0.0090, entirely within statistical noise.

**This is the analytically correct result for the available feature set.** The CRM fields in this dataset (source, sentiment, word count, owner frequency) do not individually predict whether a lead will close. A useful predictive model would require additional features such as deal size, response time, engagement score, industry vertical, and decision-maker seniority.

---

## 9. Explainability Summary

### Global Feature Importance (consensus across all 5 methods)

| Rank | Feature | SHAP mean\|val\| | Gini | LightGBM splits | Permutation drop |
|---|---|---|---|---|---|
| 1 | `notes_word_count` | **0.00571** | 0.171 | 1,701 | +0.00419 |
| 2 | `notes_sentiment` | 0.00518 | **0.392** | **4,337** | **+0.00895** |
| 3 | `company_freq` | 0.00428 | 0.227 | 2,131 | −0.00477 |
| 4 | `source_tier` | 0.00267 | 0.016 | 352 | +0.00223 |
| 5 | `owner_freq` | 0.00175 | 0.050 | 520 | +0.00252 |

All permutation drops are within ±0.01. No feature provides genuine predictive lift.

> **Gini bias note:** Gini importance inflates continuous features (`notes_sentiment`, `company_freq`, `notes_word_count`) because trees split them more frequently regardless of true predictive value. SHAP and permutation importance are more reliable.

### Local Predictions (SHAP Waterfall)

| Case | Predicted Prob | Actual | Top Driver |
|---|---|---|---|
| True Positive | 0.5915 | Won | `company_freq` +0.045 |
| False Positive | 0.6512 | Lost | `company_freq` +0.058 |
| False Negative | 0.3867 | Won | `source_Referral` −0.032 |
| True Negative | 0.3936 | Lost | `source_Referral` −0.029 |

The FN and TN cases have near-identical feature profiles, illustrating the model's inability to discriminate at the individual-lead level.

### Causal Boundary Statement

SHAP values and coefficients describe what the **model** learned from training data. They do not identify causal mechanisms. A SHAP value of +0.02 for `source_Referral` means the model associates this feature with higher win probability — not that Referral **causes** wins. All explainability findings are associations; causal claims require controlled experiments.

---

## 10. AI Integration Architecture

### Module: `src/ai_narrative.py`

```
build_context()
    │  Reads 5 validated JSON outputs (Phases 2/3/4/5/6)
    │  Returns 56-key dict — every value traces to a file
    ▼
generate_summary(context)
    │
    ├─► [WATSONX_APIKEY + PROJECT_ID set?]
    │       │ Yes → _get_iam_token() → POST /ml/v1/text/generation
    │       │        Model: ibm/granite-13b-chat-v2 (default)
    │       │        Returns: structured narrative text
    │       │
    │       └─ No / timeout / error
    │              └─► _heuristic_fallback(context)
    │                     Template-based, 2,453-char output
    │                     3 sections: EXECUTIVE SUMMARY / KEY FINDINGS / RECOMMENDATIONS
    │                     All numbers substituted from validated context dict
    │
    └─► render_insights(context) → 9 structured insight blocks
```

### System Prompt Design

The watsonx.ai system prompt enforces:
1. **Use only provided numbers** — no hallucination
2. **No causal claims** — all findings are "associated with", "observed pattern"
3. **Fixed output structure** — exactly 3 sections, ≤500 words
4. **Grounded in context** — every statistic references the validated JSON

### Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `WATSONX_APIKEY` | For LLM path | — | IBM Cloud IAM API key |
| `PROJECT_ID` | For LLM path | — | watsonx.ai project GUID |
| `WATSONX_URL` | No | `https://us-south.ml.cloud.ibm.com` | Regional endpoint |
| `WATSONX_MODEL_ID` | No | `ibm/granite-13b-chat-v2` | Granite model variant |

No credentials are hard-coded in any source file. The project runs fully without API keys using the heuristic fallback.

### Output: `reports/ai_narrative.json`

```json
{
  "generated_by": "src/ai_narrative.py",
  "narrative_source": "heuristic_fallback",
  "watsonx_enabled": false,
  "context_snapshot": { "total_leads": 100000, "win_rate_pct": 9.99, ... },
  "narrative": "EXECUTIVE SUMMARY\n...",
  "insights": [ { "id": "data_quality", "title": "...", "rating": "positive", ... }, ... ]
}
```

---

## 11. Interactive Dashboard

### Launch

```bash
streamlit run dashboard/streamlit_app.py
```

### Views

| Tab | Content |
|---|---|
| 📋 Overview | KPI cards (Total Leads, Conversion Rate, Top Source, Top Group) · Deal-stage bar · Source win-rate bar · Volume vs conversion scatter · Sentiment histogram |
| 📈 Performance | Source group win rate & volume bars · Source×Group heatmap · Won/Not-Won stacked bar · Full source summary table |
| 🔬 Explainability | SHAP global (G5) · Permutation importance (G4) · LR coefficients (G1) · SHAP waterfall local cases TP/FP/FN/TN (L1) · Causal boundary statement |
| 🤖 AI Insights | Provenance banner · Context snapshot · Executive briefing · 9 tabbed insight cards · Regenerate button |
| 🎯 What-If Scorer | Lead attribute form · Plotly gauge chart · Win probability metric · Feature contribution bar |

### Filters (sidebar)

- **Lead Source** — multiselect, all 20 sources
- **Source Group** — multiselect, all 7 groups
- **Lead Owner** — case-insensitive substring search

All metrics, KPI cards, and charts update live on filter change. No hardcoded values — every number is computed from the loaded CSV or loaded from JSON at runtime.

---

## 12. Testing

### Test Suite

```bash
pytest tests/test_phase9.py -v
```

**Result: 83 passed, 0 failed, 0 errors** (runtime ~4–5 s)

### Test Classes

| Class | Tests | Coverage |
|---|---|---|
| `TestDataPipelineIntegrity` | 15 | Schema, row count, uniqueness, binary constraints, quality report cross-check |
| `TestSQLQueryOutputs` | 11 | Pipeline summary values, accounting identity, source rankings, stage distribution, cross-phase consistency |
| `TestModelInference` | 12 | Pipeline steps, `predict_proba` shape/bounds/row-sums/determinism, all 20 sources, leakage detection |
| `TestAIServiceFallback` | 13 | `build_context` keys/values, heuristic structure/numbers/no-hallucination, `render_insights` schema, `_call_watsonx` guard, system prompt rules, JSON output schema, no hardcoded keys |
| `TestEDAKPIOutputs` | 12 | KPI 01/02/03/04/11 values, source ranking correctness, win count accounting, sentiment negligibility |
| `TestExplainabilityOutputs` | 13 | All 8 sections present, LR coef count/signs, permutation near-zero, SHAP non-negative, local case labels/probabilities, AVP ranges, causal boundary completeness |
| `TestFeatureEngineering` | 7 | Source tier completeness/values, tier-1 > tier-3 rate validation, freq-encoding range, feature names JSON, leakage absence |

### Per-Phase Smoke Tests

Each phase also ships a dedicated validation script:

```bash
python src/validate_phase2.py   # 10 checks
python src/validate_phase3.py
python src/validate_phase4.py
python src/validate_phase5.py
python src/validate_phase6.py   # 11 checks — all pass
python src/validate_phase7.py   # 12 checks — all pass
python src/validate_phase8.py   # 14 checks — all pass
```

---

## 13. Project Limitations & Ethical Considerations

### Predictive Model Limitations

**The models trained in Phase 5 are not deployable for individual-lead scoring.** All three classifiers achieve validation ROC-AUC ≈ 0.50–0.51, indistinguishable from a random classifier. Permutation importance confirms no available feature provides genuine predictive lift. Deploying these scores to rank or prioritise individual leads would be misleading and potentially harmful (e.g., systematically deprioritising leads from lower-scored sources without factual basis).

The analytical findings (win rate by channel) are valid for **group-level, strategic channel investment decisions** — not for individual-lead prediction.

### Features Required Before Deployment

A useful lead scorer would require at minimum:
- Deal size / contract value
- Response time (lead created → first contact)
- Number of activities (calls, emails, meetings)
- Engagement score (email opens, web visits)
- Industry vertical / company size
- Decision-maker title / seniority

### Data & Privacy

- All personally identifiable information (PII) — first name, last name, phone, email — was removed from `cleaned_leads.csv` during preprocessing and stored separately in `data/processed/pii_vault.csv`.
- `lead_owner` was retained only as a frequency-encoded feature (count/total) — the raw name string is not used in model training.
- `company` was similarly frequency-encoded; the raw company name is not in the feature set.

### Causal Attribution

Throughout this project, all findings are described as **statistical associations**. Higher win rates for specific channels (Podcast: 10.69%, Networking Event: 9.28%) reflect observed patterns in historical data. They do not establish that those channels *cause* higher or lower win rates. Confounders — lead quality, industry, deal complexity, sales rep assignment — may fully explain the differences.

Business decisions based on these findings should be validated through controlled A/B experiments or quasi-experimental methods (e.g., difference-in-differences, regression discontinuity) before significant budget reallocation.

### AI-Generated Content

The Phase 8 AI narrative (whether generated by IBM watsonx.ai Granite or the heuristic fallback) draws exclusively from validated JSON outputs produced by prior phases. The system prompt explicitly prohibits hallucination of statistics. However:
- LLM-generated text should be reviewed by a domain expert before external publication.
- The heuristic fallback is deterministic and auditable — every number is a Python `str.format()` substitution from validated context.

### Model Bias

The dataset contains 93,012 unique lead owners, each with an average of 1.07 leads. This means owner-level performance cannot be reliably estimated. Frequency-encoded owner features carry near-zero signal (permutation importance: +0.0025). Using owner identity as a predictor in a real deployment context would risk introducing fairness concerns (e.g., penalising new or part-time sales reps based on data sparsity rather than performance).

---

## 14. Glossary

| Term | Definition |
|---|---|
| ROC-AUC | Area under the Receiver Operating Characteristic curve. 0.50 = random, 1.0 = perfect. |
| Permutation Importance | Mean ROC-AUC drop when a feature is randomly shuffled. Near-zero = no genuine signal. |
| SHAP | SHapley Additive exPlanations. Additive attribution of feature contributions to a single prediction. |
| Gini Importance | Tree-based impurity reduction metric. Biased toward high-cardinality and continuous features. |
| Win Rate | Closed Won / Total Leads for a given segment. |
| Source Tier | Ordinal encoding (1/2/3) based on Phase 4 win-rate ranking: Tier 1 ≥ 10.46%, Tier 2 = 9.70%–10.38%, Tier 3 ≤ 9.67%. |
| Frequency Encoding | Feature `f = count(entity) / total_leads`. Captures entity size without target leakage. |
| Class Imbalance | 9,993 positive vs 90,007 negative; addressed with `class_weight='balanced'`. |
| Heuristic Fallback | Deterministic template-based text generation used when watsonx.ai credentials are absent. |
| IBM Granite | IBM's foundation model family for enterprise AI. Used via the watsonx.ai `/ml/v1/text/generation` API. |

---

*Lead Intelligence Platform — built across 9 phases with Python, scikit-learn, LightGBM, SHAP, Streamlit, Plotly, and IBM watsonx.ai.*
