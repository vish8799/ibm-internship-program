"""
Phase 8 smoke test — validates AI narrative module and dashboard integration.

Checks:
  1.  Required source files exist (ai_narrative.py, ai_insights_phase8.py)
  2.  Required input JSONs all loadable
  3.  build_context() returns correct keys and plausible values
  4.  heuristic fallback renders without format errors
  5.  render_insights() returns 9 blocks with required fields
  6.  No API keys are hard-coded in ai_narrative.py
  7.  generate_summary() returns correct structure (no watsonx creds)
  8.  reports/ai_narrative.json exists, non-empty, schema valid
  9.  reports/ai_insights.json exists, 9 insights, no LLM key required
  10. All insight texts are non-empty and contain numbers from context
  11. Causal boundary: key phrases present in narrative and insights
  12. System prompt references key anti-hallucination rules
  13. Dashboard streamlit_app.py contains AI Insights route and loader
  14. Dashboard imports os at module level (for WATSONX_MODEL_ID env read)
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# 1. Required source files exist
# ---------------------------------------------------------------------------
REQUIRED_SRC = [
    ROOT / "src"     / "ai_narrative.py",
    ROOT / "src"     / "ai_insights_phase8.py",
]
for p in REQUIRED_SRC:
    assert p.exists(), f"Missing source file: {p}"
    assert p.stat().st_size > 1000, f"Suspiciously small source file: {p}"
print(f"[OK] Source files: {len(REQUIRED_SRC)} present")

# ---------------------------------------------------------------------------
# 2. Required input JSONs all loadable
# ---------------------------------------------------------------------------
INPUT_JSONS = [
    ROOT / "data"    / "processed" / "data_quality_report.json",
    ROOT / "reports" / "eda_phase3_kpis.json",
    ROOT / "models"  / "ml_phase5_metrics.json",
    ROOT / "reports" / "explainability_phase6.json",
    ROOT / "reports" / "sql_phase4_results.json",
]
for p in INPUT_JSONS:
    assert p.exists(), f"Missing input JSON: {p}"
    with open(p) as f:
        data = json.load(f)
    assert data, f"Empty JSON: {p}"
print(f"[OK] Input JSONs: {len(INPUT_JSONS)} files loadable")

# ---------------------------------------------------------------------------
# 3. build_context() returns correct keys and plausible values
# ---------------------------------------------------------------------------
import importlib.util
spec = importlib.util.spec_from_file_location("ai_narrative", ROOT / "src" / "ai_narrative.py")
narr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(narr)

ctx = narr.build_context()

REQUIRED_CTX_KEYS = [
    "total_leads", "win_rate_pct", "top_source", "top_source_rate",
    "bot_source", "bot_source_rate", "win_rate_spread",
    "best_sg", "best_sg_rate", "worst_sg",
    "sent_won", "sent_lost", "sent_diff",
    "best_model_name", "best_val_auc", "auc_above_baseline",
    "perm_max", "perm_min", "top_shap_feature", "top_shap_value",
    "top_gini_feature", "top_gini_imp",
    "missing_values", "duplicate_ids", "invalid_sources", "invalid_stages",
    "open_pipeline", "closed_won", "close_rate_pct",
    "split_train", "split_val", "split_test",
    "top5_sources",
]
missing_keys = [k for k in REQUIRED_CTX_KEYS if k not in ctx]
assert not missing_keys, f"Missing context keys: {missing_keys}"

# Value plausibility
assert ctx["total_leads"] == 100_000
assert 9.0 <= ctx["win_rate_pct"] <= 11.0, f"Win rate out of range: {ctx['win_rate_pct']}"
assert ctx["top_source"] == "Podcast"
assert ctx["bot_source"] == "Networking Event"
assert 0.4 < ctx["best_val_auc"] < 0.7, f"ROC-AUC implausible: {ctx['best_val_auc']}"
assert abs(ctx["perm_max"]) < 0.02 and abs(ctx["perm_min"]) < 0.02
assert len(ctx["top5_sources"]) == 5
assert ctx["sql_win_rate_pct"] == ctx["win_rate_pct"], \
    "SQL and KPI win rates disagree — potential data inconsistency"

print(f"[OK] build_context(): {len(ctx)} keys, all plausibility checks pass")
print(f"     win_rate={ctx['win_rate_pct']}% | top_source={ctx['top_source']} | "
      f"best_model={ctx['best_model_name']} ({ctx['best_val_auc']})")

# ---------------------------------------------------------------------------
# 4. Heuristic fallback renders without format errors
# ---------------------------------------------------------------------------
heuristic_text = narr._heuristic_fallback(ctx)
assert len(heuristic_text) > 500, f"Heuristic too short: {len(heuristic_text)}"
assert "EXECUTIVE SUMMARY" in heuristic_text
assert "KEY FINDINGS" in heuristic_text
assert "RECOMMENDATIONS" in heuristic_text
# Numbers from context must be present
assert str(ctx["total_leads"]) in heuristic_text.replace(",", "")
assert str(ctx["win_rate_pct"]) in heuristic_text
assert ctx["top_source"] in heuristic_text
assert ctx["bot_source"] in heuristic_text
print(f"[OK] Heuristic fallback: {len(heuristic_text)} chars, all required sections present")

# ---------------------------------------------------------------------------
# 5. render_insights() returns 9 blocks with required fields
# ---------------------------------------------------------------------------
insights = narr.render_insights(ctx)
assert len(insights) == 9, f"Expected 9 insights, got {len(insights)}"
INSIGHT_IDS = {
    "data_quality", "pipeline_overview", "source_performance",
    "source_group_insight", "sentiment_finding", "ml_performance",
    "feature_importance", "business_recommendations", "causal_disclaimer",
}
for ins in insights:
    assert "id"       in ins and ins["id"] in INSIGHT_IDS, f"Bad insight id: {ins.get('id')}"
    assert "title"    in ins and len(ins["title"]) > 3
    assert "category" in ins and len(ins["category"]) > 1
    assert "rating"   in ins and ins["rating"] in ("positive", "neutral", "warning")
    assert "text"     in ins and len(ins["text"]) > 30, f"Empty text in: {ins['id']}"
print(f"[OK] render_insights(): {len(insights)} blocks, all {len(INSIGHT_IDS)} required IDs present")

# ---------------------------------------------------------------------------
# 6. No API keys hard-coded in ai_narrative.py
# ---------------------------------------------------------------------------
src_text = (ROOT / "src" / "ai_narrative.py").read_text(encoding="utf-8")

# Must read from env — check key strings appear only in os.environ.get() calls
assert 'os.environ.get("WATSONX_APIKEY"' in src_text or \
       "os.environ.get('WATSONX_APIKEY'" in src_text, \
    "WATSONX_APIKEY must be read from os.environ.get()"
assert 'os.environ.get("PROJECT_ID"' in src_text or \
       "os.environ.get('PROJECT_ID'" in src_text, \
    "PROJECT_ID must be read from os.environ.get()"

# Must NOT contain literal credential patterns (30+ char alphanumeric strings
# that look like API keys — not counting the template/const names)
HARDCODED_PATTERNS = [
    r'WATSONX_APIKEY\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']',
    r'PROJECT_ID\s*=\s*["\'][a-f0-9\-]{20,}["\']',
    r'apikey\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']',
]
for pat in HARDCODED_PATTERNS:
    hits = re.findall(pat, src_text)
    assert not hits, f"Potential hardcoded credential found: {hits}"

print(f"[OK] No hardcoded API keys in ai_narrative.py")

# ---------------------------------------------------------------------------
# 7. generate_summary() returns correct structure (no watsonx creds set)
# ---------------------------------------------------------------------------
# Ensure env vars are NOT set for this test
for ev in ("WATSONX_APIKEY", "PROJECT_ID"):
    os.environ.pop(ev, None)

result = narr.generate_summary(ctx)
assert "narrative_text"  in result
assert "source"          in result
assert "watsonx_enabled" in result
assert "context"         in result
assert "insights"        in result

assert result["source"]          == "heuristic_fallback"
assert result["watsonx_enabled"] == False
assert len(result["narrative_text"]) > 200
assert len(result["insights"])   == 9
print(f"[OK] generate_summary() (no creds): "
      f"source={result['source']}, {len(result['insights'])} insights")

# ---------------------------------------------------------------------------
# 8. reports/ai_narrative.json schema valid
# ---------------------------------------------------------------------------
narr_path = ROOT / "reports" / "ai_narrative.json"
assert narr_path.exists(), "reports/ai_narrative.json not found"
with open(narr_path, encoding="utf-8") as f:
    narr_json = json.load(f)

REQUIRED_NARR_KEYS = [
    "generated_by", "narrative_source", "watsonx_enabled",
    "context_snapshot", "narrative", "insights",
]
for k in REQUIRED_NARR_KEYS:
    assert k in narr_json, f"Missing key in ai_narrative.json: {k}"

assert len(narr_json["narrative"]) > 200
assert len(narr_json["insights"]) == 9
snap = narr_json["context_snapshot"]
assert "total_leads" in snap and snap["total_leads"] == 100_000
assert "win_rate_pct" in snap and snap["win_rate_pct"] == 9.99

print(f"[OK] reports/ai_narrative.json: valid schema, "
      f"source={narr_json['narrative_source']}, "
      f"{len(narr_json['insights'])} insights")

# ---------------------------------------------------------------------------
# 9. reports/ai_insights.json exists, 9 insights, correct structure
# ---------------------------------------------------------------------------
insights_path = ROOT / "reports" / "ai_insights.json"
assert insights_path.exists(), "reports/ai_insights.json not found"
with open(insights_path) as f:
    ai_json = json.load(f)

assert ai_json["total_insights"] == 9
assert len(ai_json["insights"]) == 9
assert ai_json["llm_used"] == False   # no key configured in CI
for ins in ai_json["insights"]:
    assert "id" in ins and "title" in ins and "display_text" in ins
    assert len(ins["display_text"]) > 30

print(f"[OK] reports/ai_insights.json: {ai_json['total_insights']} insights, "
      f"llm_used={ai_json['llm_used']}")

# ---------------------------------------------------------------------------
# 10. All insight texts are non-empty and contain numbers from context
# ---------------------------------------------------------------------------
for ins in narr_json["insights"]:
    text = ins["text"]
    assert len(text) > 30, f"Insight '{ins['id']}' text too short"

# Most insight blocks should contain numbers — exclude methodology/disclaimer
numeric_insights = [
    i for i in narr_json["insights"]
    if i["id"] not in ("causal_disclaimer",)
]
for ins in numeric_insights:
    has_number = bool(re.search(r'\d', ins["text"]))
    assert has_number, f"Insight '{ins['id']}' contains no numeric data"

print(f"[OK] All 9 insight texts are non-empty; {len(numeric_insights)} data-backed with numeric values")

# ---------------------------------------------------------------------------
# 11. Causal boundary: key phrases in narrative and insights
# ---------------------------------------------------------------------------
full_narrative = narr_json["narrative"].lower()
causal_phrases = [
    "association", "statistic", "not establish causation", "a/b",
]
found_phrases = [p for p in causal_phrases if p in full_narrative]
assert len(found_phrases) >= 2, \
    f"Expected causal-boundary language in narrative, found: {found_phrases}"

# Check causal_disclaimer insight
disclaimer = next(
    (i for i in narr_json["insights"] if i["id"] == "causal_disclaimer"), None
)
assert disclaimer is not None, "Missing causal_disclaimer insight"
assert "association" in disclaimer["text"].lower() or "causation" in disclaimer["text"].lower()

print(f"[OK] Causal boundary language present in narrative and insights")

# ---------------------------------------------------------------------------
# 12. System prompt references key anti-hallucination rules
# ---------------------------------------------------------------------------
system_prompt = narr._SYSTEM_PROMPT
assert "DO NOT invent" in system_prompt or "do not invent" in system_prompt.lower()
assert "causation" in system_prompt.lower() or "causal" in system_prompt.lower()
assert "CONTEXT" in system_prompt or "context" in system_prompt
print(f"[OK] System prompt contains anti-hallucination rules")

# ---------------------------------------------------------------------------
# 13. Dashboard contains AI Insights route and loader
# ---------------------------------------------------------------------------
app_src = (ROOT / "dashboard" / "streamlit_app.py").read_text(encoding="utf-8")

assert 'NARRATIVE_PATH' in app_src, "Dashboard must define NARRATIVE_PATH"
assert 'load_narrative' in app_src, "Dashboard must define load_narrative()"
assert 'render_ai_insights' in app_src, "Dashboard must define render_ai_insights()"
assert '"AI Insights"' in app_src, "Dashboard router must handle 'AI Insights' view"
assert '🤖 AI Insights' in app_src, "Sidebar must include AI Insights radio option"
assert 'ai_narrative.json' in app_src, "Dashboard must reference ai_narrative.json path"

print(f"[OK] Dashboard: NARRATIVE_PATH, load_narrative, render_ai_insights, router all present")

# ---------------------------------------------------------------------------
# 14. Dashboard imports os at module level
# ---------------------------------------------------------------------------
# The render_ai_insights function uses os.environ.get() for WATSONX_MODEL_ID
top_section = app_src[:app_src.index("def load_data")]
assert "import os" in top_section, \
    "os must be imported at module level in dashboard (needed by render_ai_insights)"
print(f"[OK] Dashboard imports os at module level")

# ---------------------------------------------------------------------------
print()
print("=== ALL PHASE 8 VALIDATION CHECKS PASSED ===")
print(f"  ai_narrative.py     : {len(src_text.splitlines())} lines")
print(f"  Inference path      : watsonx.ai (WATSONX_APIKEY+PROJECT_ID) -> heuristic fallback")
print(f"  Heuristic quality   : {len(heuristic_text)} chars, 3 sections, fully data-grounded")
print(f"  Insight blocks      : 9 (Data/KPI/Analytics/Feature/ML/Explainability/Recs/Method)")
print(f"  API key handling    : read from env vars only, no hard-coded credentials")
print(f"  Causal disclaimers  : present in narrative text, system prompt, and insight cards")
print(f"  Dashboard tabs      : 5 (Overview/Performance/Explainability/AI Insights/Scorer)")
print(f"  Output files        : reports/ai_narrative.json · reports/ai_insights.json")
