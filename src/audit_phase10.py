"""
Phase 10 — Final Project Audit
================================
Runs a comprehensive automated audit across the entire repository.

Checks:
  A1  Raw data integrity        — SHA-256 fingerprint, row count, uniqueness
  A2  Raw vs processed          — source/stage cross-check for 50 sampled rows
  A3  Security compliance       — no hardcoded API keys or PII in source files
  A4  PII in cleaned CSV        — name/phone/email columns absent
  A5  Reproducibility           — all expected output files present + non-empty
  A6  Phase output consistency  — numeric cross-checks across all phase JSONs
  A7  Relative path compliance  — no absolute paths in source files
  A8  Env-var credential reads  — watsonx creds only via os.environ.get()
  A9  No target leakage         — leakage columns absent from feature names
  A10 Test suite                — 83 pytest tests pass
  A11 Dashboard module          — imports cleanly, has all required views
  A12 README completeness       — all 14 sections present

Run:
    python src/audit_phase10.py
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results: list[tuple[str, str, str]] = []   # (check_id, status, detail)


def record(check_id: str, status: str, detail: str) -> None:
    results.append((check_id, status, detail))
    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}[status]
    print(f"  {icon} {check_id}: {detail}")


# ---------------------------------------------------------------------------
# A1 — Raw data integrity (SHA-256, row count, uniqueness)
# ---------------------------------------------------------------------------
print("\n=== A1  Raw data integrity ===")

RAW_PATH = ROOT / "leads-100000.csv"
assert RAW_PATH.exists(), "leads-100000.csv missing from repo root"

sha256 = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
record("A1.sha256",   PASS, f"SHA-256 = {sha256}")
record("A1.filesize", PASS, f"File size = {RAW_PATH.stat().st_size:,} bytes")

with open(RAW_PATH, encoding="utf-8") as fh:
    raw_reader = list(csv.DictReader(fh))

record("A1.rowcount", PASS if len(raw_reader) == 100_000 else FAIL,
       f"Row count = {len(raw_reader):,} (expected 100,000)")

acc_ids = [r["Account Id"] for r in raw_reader]
record("A1.uniqueids", PASS if len(set(acc_ids)) == 100_000 else FAIL,
       f"Unique Account IDs = {len(set(acc_ids)):,}")

expected_headers = {
    "Index", "Account Id", "Lead Owner", "First Name", "Last Name",
    "Company", "Phone 1", "Phone 2", "Email 1", "Email 2",
    "Website", "Source", "Deal Stage", "Notes",
}
actual_headers = set(raw_reader[0].keys())
missing_h = expected_headers - actual_headers
record("A1.headers", PASS if not missing_h else FAIL,
       f"All 14 expected raw columns present" if not missing_h
       else f"Missing headers: {missing_h}")

raw_copy = ROOT / "data" / "raw" / "leads-100000.csv"
if raw_copy.exists():
    copy_hash = hashlib.sha256(raw_copy.read_bytes()).hexdigest()
    record("A1.rawcopy", PASS if copy_hash == sha256 else FAIL,
           f"data/raw copy hash {'matches' if copy_hash == sha256 else 'DIFFERS'}")
else:
    record("A1.rawcopy", WARN, "data/raw/leads-100000.csv not present (acceptable)")


# ---------------------------------------------------------------------------
# A2 — Raw vs processed cross-check
# ---------------------------------------------------------------------------
print("\n=== A2  Raw vs processed cross-check ===")

import pandas as pd
clean_df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_leads.csv")
raw_index = {r["Account Id"]: r for r in raw_reader}

import random
random.seed(42)
sample_ids = random.sample(list(raw_index.keys()), 50)
source_mismatches = []
stage_mismatches  = []
missing_in_clean  = []

for aid in sample_ids:
    raw = raw_index[aid]
    match = clean_df[clean_df["account_id"] == aid]
    if len(match) == 0:
        missing_in_clean.append(aid)
        continue
    proc = match.iloc[0]
    if raw["Source"] != proc["source"]:
        source_mismatches.append(aid)
    if raw["Deal Stage"] != proc["deal_stage"]:
        stage_mismatches.append(aid)

record("A2.missing",  PASS if not missing_in_clean else FAIL,
       f"All 50 sampled IDs found in cleaned CSV")
record("A2.source",   PASS if not source_mismatches else FAIL,
       f"Source column preserved correctly in all 50 samples")
record("A2.dealstage",PASS if not stage_mismatches else FAIL,
       f"Deal Stage preserved correctly in all 50 samples")
record("A2.rowcount", PASS if len(clean_df) == 100_000 else FAIL,
       f"cleaned_leads.csv row count = {len(clean_df):,}")


# ---------------------------------------------------------------------------
# A3 — Security compliance: no hardcoded credentials in source files
# ---------------------------------------------------------------------------
print("\n=== A3  Security compliance ===")

src_files = (
    list((ROOT / "src").glob("*.py"))
    + list((ROOT / "dashboard").glob("*.py"))
    + list((ROOT / "tests").glob("*.py"))
)
src_files = [f for f in src_files if "__pycache__" not in str(f)]

CRED_PATTERNS = [
    # Hardcoded IBM Cloud IAM keys — real keys are exactly 44-char alphanumeric
    r'WATSONX_APIKEY\s*=\s*["\'][A-Za-z0-9+/=_-]{20,}["\']',
    # Hardcoded project GUIDs
    r'PROJECT_ID\s*=\s*["\'][a-f0-9-]{30,}["\']',
    # Hardcoded OpenAI keys
    r'sk-[A-Za-z0-9]{30,}',
    # Hardcoded password fields
    r'password\s*=\s*["\'][^"\']{8,}["\']',
]

cred_hits: list[str] = []
for pyfile in src_files:
    text = pyfile.read_text(encoding="utf-8", errors="ignore")
    for pat in CRED_PATTERNS:
        hits = re.findall(pat, text, re.IGNORECASE)
        for hit in hits:
            # Exclude os.environ.get context (legitimate reads)
            ctx_start = max(0, text.find(hit) - 60)
            ctx_end   = text.find(hit) + len(hit) + 60
            ctx = text[ctx_start:ctx_end]
            if "os.environ" in ctx or "env_var" in ctx.lower():
                continue
            # Exclude comments
            line_start = text.rfind("\n", 0, text.find(hit)) + 1
            line = text[line_start:text.find("\n", text.find(hit))]
            if line.lstrip().startswith("#"):
                continue
            cred_hits.append(f"{pyfile.name}: {hit[:60]}")

record("A3.no_hardcoded_creds", PASS if not cred_hits else FAIL,
       f"No hardcoded credentials in {len(src_files)} source files"
       if not cred_hits else f"Credential hits: {cred_hits[:3]}")

# Check env-var patterns ARE present in ai_narrative.py
narr_text = (ROOT / "src" / "ai_narrative.py").read_text(encoding="utf-8")
has_env_read = 'os.environ.get("WATSONX_APIKEY"' in narr_text or \
               "os.environ.get('WATSONX_APIKEY'" in narr_text
record("A3.env_var_reads", PASS if has_env_read else FAIL,
       "WATSONX_APIKEY read via os.environ.get() in ai_narrative.py")


# ---------------------------------------------------------------------------
# A4 — PII absent from cleaned CSV
# ---------------------------------------------------------------------------
print("\n=== A4  PII absent from cleaned CSV ===")

PII_COLS = ["first_name", "last_name", "phone_1", "phone_2",
            "email_1", "email_2", "First Name", "Last Name",
            "Phone 1", "Phone 2", "Email 1", "Email 2"]
cols_lower = [c.lower().replace(" ", "_") for c in clean_df.columns]
pii_found = [c for c in PII_COLS if c.lower().replace(" ", "_") in cols_lower]
record("A4.pii_columns", PASS if not pii_found else FAIL,
       f"No PII columns (name/phone/email) in cleaned_leads.csv"
       if not pii_found else f"PII columns present: {pii_found}")

# lead_owner should be present but not contain raw email addresses
if "lead_owner" in clean_df.columns:
    email_in_owner = clean_df["lead_owner"].astype(str).str.contains(
        r"@[a-zA-Z0-9.]+\.[a-zA-Z]{2,}", regex=True
    ).sum()
    record("A4.owner_no_email", PASS if email_in_owner == 0 else WARN,
           f"lead_owner column contains no email-format strings ({email_in_owner} found)")

# Check pii_vault exists
pii_vault = ROOT / "data" / "processed" / "pii_vault.csv"
record("A4.pii_vault", PASS if pii_vault.exists() else WARN,
       f"PII vault present at data/processed/pii_vault.csv"
       if pii_vault.exists() else "pii_vault.csv not found (PII may not have been vaulted)")


# ---------------------------------------------------------------------------
# A5 — Reproducibility: all expected output files present + non-empty
# ---------------------------------------------------------------------------
print("\n=== A5  Reproducibility — output files ===")

EXPECTED_FILES = {
    # Phase 2
    "data/processed/cleaned_leads.csv":          10_000,
    "data/processed/data_quality_report.json":   100,
    # Phase 3
    "reports/eda_phase3_kpis.json":               500,
    "reports/figures/fig_01_win_rate_by_source.png": 10_000,
    # Phase 4
    "reports/sql_phase4_results.json":            5_000,
    "data/processed/leads.db":                   500_000,
    # Phase 5
    "models/best_model.pkl":                     5_000,
    "models/logistic_regression_model.pkl":      5_000,
    "models/random_forest_model.pkl":            5_000,
    "models/lightgbm_model.pkl":                 5_000,
    "models/ml_phase5_metrics.json":             500,
    "models/feature_names.json":                 200,
    # Phase 6
    "reports/explainability_phase6.json":        5_000,
    "reports/figures/expl_g5_shap_summary.png":  10_000,
    "reports/figures/expl_l1_shap_waterfall_tp.png": 10_000,
    # Phase 7
    "dashboard/streamlit_app.py":                20_000,
    # Phase 8
    "reports/ai_narrative.json":                 1_000,
    "reports/ai_insights.json":                  1_000,
    # Phase 9
    "tests/test_phase9.py":                      5_000,
    "README.md":                                 5_000,
    "requirements.txt":                          50,
}

all_present = True
for rel_path, min_bytes in EXPECTED_FILES.items():
    fpath = ROOT / rel_path
    exists  = fpath.exists()
    big_enough = exists and fpath.stat().st_size >= min_bytes
    if not exists:
        record(f"A5.{Path(rel_path).name}", FAIL, f"MISSING: {rel_path}")
        all_present = False
    elif not big_enough:
        record(f"A5.{Path(rel_path).name}", FAIL,
               f"Too small ({fpath.stat().st_size} < {min_bytes}): {rel_path}")
        all_present = False

if all_present:
    record("A5.summary", PASS, f"All {len(EXPECTED_FILES)} expected output files present and non-empty")


# ---------------------------------------------------------------------------
# A6 — Phase output consistency: numeric cross-checks
# ---------------------------------------------------------------------------
print("\n=== A6  Cross-phase numeric consistency ===")

with open(ROOT / "reports" / "eda_phase3_kpis.json") as f:
    kpis = json.load(f)
with open(ROOT / "reports" / "sql_phase4_results.json") as f:
    sql  = json.load(f)
with open(ROOT / "models" / "ml_phase5_metrics.json") as f:
    ml5  = json.load(f)
with open(ROOT / "reports" / "explainability_phase6.json") as f:
    expl = json.load(f)
with open(ROOT / "reports" / "ai_narrative.json", encoding="utf-8") as f:
    narr = json.load(f)

# Phase 3 vs SQL
kpi_rate = kpis["kpi_01_overall_win_rate"]["win_rate_pct"]
sql_rate = sql["Q01_PIPELINE_SUMMARY"][0]["win_rate_pct"]
record("A6.p3_vs_p4_winrate", PASS if kpi_rate == sql_rate else FAIL,
       f"Phase 3 ({kpi_rate}%) == SQL Phase 4 ({sql_rate}%) win rate")

# Phase 3 vs cleaned CSV
csv_rate = round(clean_df["is_won"].mean() * 100, 2)
record("A6.p3_vs_csv_winrate", PASS if kpi_rate == csv_rate else FAIL,
       f"KPI win rate ({kpi_rate}%) == computed from CSV ({csv_rate}%)")

# Phase 3 won count == sum of source won counts
total_won_by_source = sum(e["won"] for e in kpis["kpi_03_win_rate_by_source"])
record("A6.source_won_sum", PASS if total_won_by_source == kpis["kpi_01_overall_win_rate"]["closed_won"] else FAIL,
       f"Sum of source won counts ({total_won_by_source}) == closed_won ({kpis['kpi_01_overall_win_rate']['closed_won']})")

# Phase 5 best model documented in Phase 8
narr_model = narr["context_snapshot"].get("best_model_name")
ml5_model  = ml5["best_model"]["name"]
record("A6.p5_vs_p8_model", PASS if narr_model == ml5_model else FAIL,
       f"Phase 5 best model ({ml5_model}) == Phase 8 snapshot ({narr_model})")

# Phase 6 permutation range
perm_max = max(p["mean_drop"] for p in expl["g4_permutation_importance"])
perm_min = min(p["mean_drop"] for p in expl["g4_permutation_importance"])
record("A6.perm_range", PASS if abs(perm_max) < 0.02 and abs(perm_min) < 0.02 else FAIL,
       f"Permutation importance in noise range: [{perm_min:.5f}, {perm_max:.5f}]")

# Phase 6 SHAP count matches feature count
shap_count = len(expl["g5_shap_mean_abs"])
record("A6.shap_feature_count", PASS if shap_count >= 31 else WARN,
       f"SHAP feature count = {shap_count} (expected >= 31)")

# Phase 8 narrative source is heuristic or watsonx (not empty)
record("A6.narr_source", PASS if narr["narrative_source"] in ("watsonx","heuristic_fallback") else FAIL,
       f"ai_narrative.json source = {narr['narrative_source']}")


# ---------------------------------------------------------------------------
# A7 — Relative path compliance
# ---------------------------------------------------------------------------
print("\n=== A7  Relative path compliance ===")

abs_path_hits: list[str] = []
DRIVE_PATTERN = re.compile(r'[A-Z]:\\|/home/[a-z]+/|/Users/[A-Za-z]+/')

for pyfile in src_files + [ROOT / "dashboard" / "streamlit_app.py"]:
    if not pyfile.exists():
        continue
    text = pyfile.read_text(encoding="utf-8", errors="ignore")
    for m in DRIVE_PATTERN.finditer(text):
        ctx_start = max(0, m.start() - 20)
        context   = text[ctx_start: m.end() + 40]
        # Skip if it is inside a comment or docstring example
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start:text.find("\n", m.start())]
        if line.lstrip().startswith("#") or ">>>" in line:
            continue
        abs_path_hits.append(f"{pyfile.name}: {context.strip()[:80]}")

record("A7.no_abs_paths", PASS if not abs_path_hits else FAIL,
       f"No absolute file paths in {len(src_files)} source files"
       if not abs_path_hits else f"Absolute paths: {abs_path_hits[:2]}")

# Verify ROOT is resolved via pathlib in key files
for fname in ["ai_narrative.py", "streamlit_app.py"]:
    fpath = next(
        (f for f in src_files + [ROOT/"dashboard"/"streamlit_app.py"] if f.name == fname),
        None
    )
    if fpath and fpath.exists():
        txt = fpath.read_text(encoding="utf-8", errors="ignore")
        has_pathlib = "Path(__file__)" in txt
        record(f"A7.pathlib_{fname}", PASS if has_pathlib else WARN,
               f"{fname} uses Path(__file__) for root resolution")


# ---------------------------------------------------------------------------
# A8 — Env-var credential reads (watsonx)
# ---------------------------------------------------------------------------
print("\n=== A8  Env-var credential pattern ===")

narr_src = (ROOT / "src" / "ai_narrative.py").read_text(encoding="utf-8")
for var in ("WATSONX_APIKEY", "PROJECT_ID", "WATSONX_URL", "WATSONX_MODEL_ID"):
    present = f'os.environ.get("{var}"' in narr_src or f"os.environ.get('{var}'" in narr_src
    record(f"A8.{var}", PASS if present else FAIL,
           f"{var} read via os.environ.get()")


# ---------------------------------------------------------------------------
# A9 — No target leakage in feature set
# ---------------------------------------------------------------------------
print("\n=== A9  Target leakage check ===")

with open(ROOT / "models" / "feature_names.json") as f:
    fn = json.load(f)

LEAKAGE_COLS = {"deal_stage", "stage_ordinal", "is_closed",
                "outcome_3class", "target_multiclass", "is_won"}
feature_set  = set(fn["all_features"])
overlap      = LEAKAGE_COLS & feature_set
record("A9.no_leakage", PASS if not overlap else FAIL,
       f"No leakage columns in feature set"
       if not overlap else f"Leakage columns found: {overlap}")

record("A9.feature_count", PASS if fn["n_features_total"] == 31 else FAIL,
       f"Feature count = {fn['n_features_total']} (expected 31)")

# Verify model does not reference raw PII columns
with open(ROOT / "models" / "best_model.pkl", "rb") as f:
    best_model = pickle.load(f)
record("A9.model_pipeline", PASS if hasattr(best_model, "named_steps") else FAIL,
       "best_model.pkl is a scikit-learn Pipeline with named_steps")

clf = best_model.named_steps.get("clf")
record("A9.classifier_type", PASS if clf is not None else FAIL,
       f"Classifier = {type(clf).__name__ if clf else 'None'}")


# ---------------------------------------------------------------------------
# A10 — Test suite: 83 pytest tests
# ---------------------------------------------------------------------------
print("\n=== A10 pytest test suite ===")

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_phase9.py", "-q", "--tb=short"],
    capture_output=True, text=True, cwd=str(ROOT), timeout=120,
)
passed = re.search(r"(\d+) passed", result.stdout)
failed = re.search(r"(\d+) failed", result.stdout)
n_pass = int(passed.group(1)) if passed else 0
n_fail = int(failed.group(1)) if failed else 0

record("A10.tests_pass", PASS if n_pass == 83 and n_fail == 0 else FAIL,
       f"{n_pass} passed, {n_fail} failed (expected 83 / 0)")
if result.returncode != 0 and n_fail > 0:
    record("A10.test_errors", FAIL, result.stdout[-500:])


# ---------------------------------------------------------------------------
# A11 — Dashboard module: imports cleanly, all 5 views present
# ---------------------------------------------------------------------------
print("\n=== A11 Dashboard module ===")

app_src = (ROOT / "dashboard" / "streamlit_app.py").read_text(encoding="utf-8")
REQUIRED_VIEWS = ["Overview", "Performance", "Explainability", "AI Insights", "What-If Scorer"]
for view in REQUIRED_VIEWS:
    record(f"A11.view_{view.replace(' ','_')}", PASS if view in app_src else FAIL,
           f"View '{view}' present in dashboard")

record("A11.cache_data",    PASS if "@st.cache_data"     in app_src else FAIL,
       "Uses @st.cache_data for CSV and JSON")
record("A11.cache_resource",PASS if "@st.cache_resource" in app_src else FAIL,
       "Uses @st.cache_resource for model")
record("A11.no_hardcoded_metrics",
       PASS if "100000" not in app_src and "9993" not in app_src else FAIL,
       "No hardcoded metric literals in dashboard source")

# Import check (syntax + deps)
import_result = subprocess.run(
    [sys.executable, "-c",
     "import ast; ast.parse(open('dashboard/streamlit_app.py').read()); print('OK')"],
    capture_output=True, text=True, cwd=str(ROOT), timeout=15,
)
record("A11.syntax_ok", PASS if import_result.returncode == 0 else FAIL,
       "dashboard/streamlit_app.py parses without syntax errors")


# ---------------------------------------------------------------------------
# A12 — README completeness
# ---------------------------------------------------------------------------
print("\n=== A12 README completeness ===")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
REQUIRED_SECTIONS = [
    "Project Overview",
    "Architecture",
    "Dataset Description",
    "Repository Structure",
    "Setup",
    "KPI",
    "ML Performance",
    "Explainability",
    "AI Integration",
    "Interactive Dashboard",
    "Testing",
    "Limitations",
    "Ethical",
    "Glossary",
]
for section in REQUIRED_SECTIONS:
    record(f"A12.{section[:15].replace(' ','_')}", PASS if section in readme else FAIL,
           f"README contains '{section}' section")

readme_lines = len(readme.splitlines())
record("A12.length", PASS if readme_lines >= 300 else WARN,
       f"README has {readme_lines} lines (expected >= 300)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 10 AUDIT SUMMARY")
print("=" * 70)

totals = {"PASS": 0, "FAIL": 0, "WARN": 0}
for cid, status, detail in results:
    totals[status] += 1

total_checks = sum(totals.values())
print(f"  Total checks : {total_checks}")
print(f"  PASS         : {totals['PASS']}")
print(f"  FAIL         : {totals['FAIL']}")
print(f"  WARN         : {totals['WARN']}")
print()

if totals["FAIL"] == 0:
    print("  OVERALL STATUS : ALL CHECKS PASSED")
    print("  SIGN-OFF       : APPROVED - Project is production-ready")
else:
    print("  OVERALL STATUS : FAILURES DETECTED")
    print("  FAILURES:")
    for cid, status, detail in results:
        if status == "FAIL":
            print(f"    - {cid}: {detail}")

if totals["WARN"] > 0:
    print("\n  WARNINGS:")
    for cid, status, detail in results:
        if status == "WARN":
            print(f"    - {cid}: {detail}")

print("=" * 70)
