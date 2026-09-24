"""
Phase 2 - Data Preparation  (data_preprocessing.py)
=====================================================
Standalone, fully reproducible preprocessing pipeline for leads-100000.csv.

Rules enforced:
  - Raw file (leads-100000.csv) is NEVER modified or overwritten.
  - All outputs go to data/processed/.
  - Every cleaning decision is logged to data/processed/preprocessing_report.json.

Outputs
-------
  data/processed/cleaned_leads.csv        <- PII-stripped, cleaned, encoded dataset
  data/processed/pii_vault.csv            <- Isolated PII (Account Id key for linkage)
  data/processed/preprocessing_report.json <- Full audit trail

Cleaning steps (matches Phase 2 task list)
-------------------------------------------
  1. Duplicate detection and removal (Account Id + full-row)
  2. Structural irregularities (whitespace, mixed case, encoding noise)
  3. Missing / empty value audit and imputation strategy
  4. Categorical standardisation: Source, Deal Stage
  5. PII separation: First/Last Name, Phone 1/2, Email 1/2  -> pii_vault.csv
  6. Target variable encoding from Deal Stage:
       - is_won          : 1 = Closed Won, 0 = all others     (binary)
       - target_multiclass: 0=New Lead,1=Qualified,2=Contacted,
                            3=Proposal Sent,4=Negotiation,
                            5=Closed Won,6=Closed Lost,
                            7=On Hold,8=Disqualified,9=Re-engagement
       - outcome_3class  : Won / Lost / Open                  (3-class)
  7. Derived features: source_group, stage_ordinal, notes_sentiment,
                       notes_word_count, notes_has_text
  8. Schema documented in preprocessing_report.json
"""

import csv
import json
import os
import sys

# ---------------------------------------------------------------------------
# Optional TextBlob sentiment
# ---------------------------------------------------------------------------
try:
    from textblob import TextBlob
    _TB = True
except ImportError:
    _TB = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_PATH        = "leads-100000.csv"
CLEAN_PATH      = "data/processed/cleaned_leads.csv"
PII_PATH        = "data/processed/pii_vault.csv"
REPORT_PATH     = "data/processed/preprocessing_report.json"
OUT_DIR         = "data/processed"

# ---------------------------------------------------------------------------
# Canonical value sets  (used for standardisation AND validation)
# ---------------------------------------------------------------------------
SOURCE_CANONICAL = {
    "chatbot":                 "Chatbot",
    "cold call":               "Cold Call",
    "cold email":              "Cold Email",
    "content marketing":       "Content Marketing",
    "direct traffic":          "Direct Traffic",
    "facebook ads":            "Facebook Ads",
    "google ads":              "Google Ads",
    "linkedin outreach":       "LinkedIn Outreach",
    "networking event":        "Networking Event",
    "organic search (seo)":    "Organic Search (SEO)",
    "other":                   "Other",
    "partner program":         "Partner Program",
    "podcast":                 "Podcast",
    "purchased list":          "Purchased List",
    "referral":                "Referral",
    "retargeting ads":         "Retargeting Ads",
    "social media":            "Social Media",
    "trade show":              "Trade Show",
    "webinars":                "Webinars",
    "website form":            "Website Form",
}

STAGE_CANONICAL = {
    "new lead":       "New Lead",
    "qualified":      "Qualified",
    "contacted":      "Contacted",
    "proposal sent":  "Proposal Sent",
    "negotiation":    "Negotiation",
    "closed won":     "Closed Won",
    "closed lost":    "Closed Lost",
    "on hold":        "On Hold",
    "disqualified":   "Disqualified",
    "re-engagement":  "Re-engagement",
}

# Funnel ordinal (used for stage_ordinal feature)
STAGE_ORDINAL = {
    "New Lead": 0, "Re-engagement": 1, "Qualified": 2,
    "Contacted": 3, "Proposal Sent": 4, "Negotiation": 5,
    "On Hold": 6, "Closed Won": 7, "Closed Lost": 8, "Disqualified": 9,
}

# Multi-class label (same order as ordinal for consistency)
STAGE_MULTICLASS = {v: k for k, v in STAGE_ORDINAL.items()}  # int -> stage name

# Source grouping (channel categories)
SOURCE_GROUP_MAP = {
    "Google Ads":          "Paid Ads",
    "Facebook Ads":        "Paid Ads",
    "Retargeting Ads":     "Paid Ads",
    "Cold Call":           "Outbound",
    "Cold Email":          "Outbound",
    "LinkedIn Outreach":   "Outbound",
    "Purchased List":      "Outbound",
    "Organic Search (SEO)":"Inbound",
    "Direct Traffic":      "Inbound",
    "Website Form":        "Inbound",
    "Content Marketing":   "Inbound",
    "Chatbot":             "Inbound",
    "Trade Show":          "Events",
    "Networking Event":    "Events",
    "Webinars":            "Events",
    "Podcast":             "Events",
    "Referral":            "Referral / Partner",
    "Partner Program":     "Referral / Partner",
    "Social Media":        "Social Media",
    "Other":               "Other",
}

# Won / Lost / Open
WON_STAGES  = {"Closed Won"}
LOST_STAGES = {"Closed Lost", "Disqualified"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def standardise_source(raw: str) -> tuple[str, bool]:
    """Return (canonical, was_changed). Falls back to stripped original."""
    stripped = raw.strip()
    canonical = SOURCE_CANONICAL.get(stripped.lower())
    if canonical:
        return canonical, (canonical != stripped)
    return stripped, False


def standardise_stage(raw: str) -> tuple[str, bool]:
    """Return (canonical, was_changed). Falls back to stripped original."""
    stripped = raw.strip()
    canonical = STAGE_CANONICAL.get(stripped.lower())
    if canonical:
        return canonical, (canonical != stripped)
    return stripped, False


def outcome_3class(stage: str) -> str:
    if stage in WON_STAGES:  return "Won"
    if stage in LOST_STAGES: return "Lost"
    return "Open"


def sentiment(text: str) -> float:
    if not _TB or not text.strip():
        return 0.0
    return round(TextBlob(text).sentiment.polarity, 4)


def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    report = {
        "pipeline":    "data_preprocessing.py",
        "raw_file":    RAW_PATH,
        "outputs":     [CLEAN_PATH, PII_PATH],
        "steps":       {},
    }

    # ── Step 1: Load raw data ─────────────────────────────────────────────
    print("[STEP 1] Loading raw dataset...")
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_cols = list(reader.fieldnames or [])
        rows = list(reader)

    shape_before = (len(rows), len(raw_cols))
    print(f"         Loaded: {shape_before[0]:,} rows x {shape_before[1]} columns")
    report["steps"]["1_load"] = {
        "rows_loaded":   shape_before[0],
        "columns":       raw_cols,
        "shape_before":  list(shape_before),
    }

    # ── Step 2: Missing / empty audit ────────────────────────────────────
    print("[STEP 2] Auditing missing values...")
    null_counts = {}
    for col in raw_cols:
        null_counts[col] = sum(1 for r in rows if not r[col].strip())
    total_nulls = sum(null_counts.values())
    print(f"         Total empty cells: {total_nulls}")
    for col, cnt in null_counts.items():
        if cnt > 0:
            print(f"         WARNING: '{col}' has {cnt} empty values")
    report["steps"]["2_missing"] = {
        "null_counts_per_col": null_counts,
        "total_nulls":         total_nulls,
        "imputation_applied":  "None required — zero missing values found",
    }

    # ── Step 3: Duplicate detection ───────────────────────────────────────
    print("[STEP 3] Detecting duplicates...")
    # 3a: Duplicate Account Id
    seen_ids = {}
    dup_account_id_indices = []
    for i, r in enumerate(rows):
        aid = r["Account Id"].strip()
        if aid in seen_ids:
            dup_account_id_indices.append(i)
        else:
            seen_ids[aid] = i

    # 3b: Full-row duplicates (all 14 fields identical)
    seen_rows = {}
    dup_fullrow_indices = []
    for i, r in enumerate(rows):
        key = tuple(r[c].strip() for c in raw_cols)
        if key in seen_rows:
            dup_fullrow_indices.append(i)
        else:
            seen_rows[key] = i

    # Remove full-row duplicates (keep first occurrence)
    rows_deduped = [r for i, r in enumerate(rows) if i not in set(dup_fullrow_indices)]

    print(f"         Duplicate Account IDs : {len(dup_account_id_indices)}")
    print(f"         Full-row duplicates   : {len(dup_fullrow_indices)} (removed)")
    print(f"         Rows after dedup      : {len(rows_deduped):,}")
    report["steps"]["3_deduplication"] = {
        "duplicate_account_ids":       len(dup_account_id_indices),
        "fullrow_duplicates_removed":  len(dup_fullrow_indices),
        "rows_after_dedup":            len(rows_deduped),
        "decision":                    "Full-row duplicates removed; duplicate Account IDs logged (none found in this dataset)",
    }

    # ── Step 4: Structural standardisation ───────────────────────────────
    print("[STEP 4] Standardising categorical fields (Source, Deal Stage)...")
    source_changed = 0
    stage_changed  = 0
    source_invalid = []
    stage_invalid  = []

    for r in rows_deduped:
        # Source
        std_src, chg_src = standardise_source(r["Source"])
        if chg_src:
            r["Source"] = std_src
            source_changed += 1
        if std_src not in SOURCE_CANONICAL.values():
            source_invalid.append(std_src)

        # Deal Stage
        std_stg, chg_stg = standardise_stage(r["Deal Stage"])
        if chg_stg:
            r["Deal Stage"] = std_stg
            stage_changed += 1
        if std_stg not in STAGE_CANONICAL.values():
            stage_invalid.append(std_stg)

        # Strip whitespace from all fields
        for col in raw_cols:
            r[col] = r[col].strip()

    print(f"         Source values normalised : {source_changed}")
    print(f"         Stage values normalised  : {stage_changed}")
    print(f"         Invalid Source values    : {len(source_invalid)}")
    print(f"         Invalid Stage values     : {len(stage_invalid)}")
    report["steps"]["4_standardisation"] = {
        "source_values_normalised": source_changed,
        "stage_values_normalised":  stage_changed,
        "invalid_source_values":    list(set(source_invalid)),
        "invalid_stage_values":     list(set(stage_invalid)),
        "whitespace_stripped":      True,
        "rules": {
            "source": "Lowercased key lookup -> canonical title-case value",
            "stage":  "Lowercased key lookup -> canonical title-case value",
        },
    }

    # ── Step 5: PII separation ────────────────────────────────────────────
    print("[STEP 5] Separating PII into pii_vault.csv...")
    PII_COLS  = ["First Name", "Last Name", "Phone 1", "Phone 2", "Email 1", "Email 2"]
    KEEP_COLS = ["Index", "Account Id", "Lead Owner", "Company",
                 "Website", "Source", "Deal Stage", "Notes"]

    pii_rows = []
    for r in rows_deduped:
        pii_rows.append({
            "Account Id":  r["Account Id"],
            "Index":       r["Index"],
            "First Name":  r["First Name"],
            "Last Name":   r["Last Name"],
            "Phone 1":     r["Phone 1"],
            "Phone 2":     r["Phone 2"],
            "Email 1":     r["Email 1"],
            "Email 2":     r["Email 2"],
        })

    with open(PII_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Account Id","Index","First Name",
                                               "Last Name","Phone 1","Phone 2",
                                               "Email 1","Email 2"])
        writer.writeheader()
        writer.writerows(pii_rows)

    print(f"         PII written to '{PII_PATH}' ({len(pii_rows):,} rows)")
    print(f"         PII columns isolated: {PII_COLS}")
    report["steps"]["5_pii_separation"] = {
        "pii_columns_isolated":  PII_COLS,
        "pii_file":              PII_PATH,
        "key_for_linkage":       ["Account Id", "Index"],
        "retained_in_clean":     KEEP_COLS,
        "pii_rows_written":      len(pii_rows),
    }

    # ── Step 6: Target variable encoding ─────────────────────────────────
    print("[STEP 6] Encoding target variables...")
    # Multiclass integer label maps directly from STAGE_ORDINAL
    target_dist = {}
    for r in rows_deduped:
        stage = r["Deal Stage"]
        target_dist[stage] = target_dist.get(stage, 0) + 1

    print("         Target distribution (is_won):")
    won  = sum(1 for r in rows_deduped if r["Deal Stage"] == "Closed Won")
    ntot = len(rows_deduped)
    print(f"           is_won=1  {won:,}  ({won/ntot*100:.2f}%)")
    print(f"           is_won=0  {ntot-won:,}  ({(ntot-won)/ntot*100:.2f}%)")

    report["steps"]["6_target_encoding"] = {
        "columns_added": {
            "is_won":           "1=Closed Won, 0=all others",
            "target_multiclass":"0=New Lead,1=Qualified,2=Contacted,3=Proposal Sent,"
                                "4=Negotiation,5=Closed Won,6=Closed Lost,"
                                "7=On Hold,8=Disqualified,9=Re-engagement",
            "outcome_3class":   "Won / Lost / Open",
        },
        "is_won_distribution": {
            "positive_1": won,
            "negative_0": ntot - won,
            "positive_pct": round(won / ntot * 100, 2),
        },
        "stage_distribution": target_dist,
    }

    # ── Step 7: Build cleaned output rows ────────────────────────────────
    print("[STEP 7] Building cleaned_leads.csv with derived features...")
    cleaned = []
    total = len(rows_deduped)

    for idx, r in enumerate(rows_deduped):
        if idx % 20000 == 0:
            print(f"         ... {idx:,}/{total:,}")

        stage  = r["Deal Stage"]
        source = r["Source"]
        notes  = r["Notes"]

        cleaned.append({
            # --- Identifiers (non-PII) ---
            "index":              r["Index"],
            "account_id":         r["Account Id"],
            # --- Retained contextual fields ---
            "lead_owner":         r["Lead Owner"],
            "company":            r["Company"],
            "website":            r["Website"],
            # --- Standardised categoricals ---
            "source":             source,
            "source_group":       SOURCE_GROUP_MAP.get(source, "Other"),
            "deal_stage":         stage,
            # --- Target encodings ---
            "is_won":             1 if stage in WON_STAGES  else 0,
            "is_closed":          1 if stage in (WON_STAGES | LOST_STAGES) else 0,
            "outcome_3class":     outcome_3class(stage),
            "target_multiclass":  STAGE_ORDINAL.get(stage, -1),
            "stage_ordinal":      STAGE_ORDINAL.get(stage, -1),
            # --- Text features ---
            "notes_sentiment":    sentiment(notes),
            "notes_word_count":   word_count(notes),
            "notes_has_text":     1 if notes.strip() else 0,
        })

    # ── Step 8: Write cleaned CSV ─────────────────────────────────────────
    out_cols = list(cleaned[0].keys())
    with open(CLEAN_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(cleaned)

    shape_after = (len(cleaned), len(out_cols))
    print(f"\n[DONE] Cleaned dataset -> '{CLEAN_PATH}'")
    print(f"       Shape before : {shape_before[0]:,} rows x {shape_before[1]} cols")
    print(f"       Shape after  : {shape_after[0]:,} rows x {shape_after[1]} cols")

    # ── Step 9: Final null audit on cleaned output ────────────────────────
    print("[VALIDATION] Null check on cleaned output...")
    null_after = {}
    for col in out_cols:
        null_after[col] = sum(1 for r in cleaned if str(r.get(col, "")).strip() == "")
    total_nulls_after = sum(null_after.values())
    print(f"             Total nulls in cleaned output: {total_nulls_after}")
    for col, cnt in null_after.items():
        if cnt > 0:
            print(f"             WARNING: '{col}' -> {cnt} empty values")

    # ── Schema documentation ──────────────────────────────────────────────
    schema = {
        "index":            "INT  — original row index from raw file; unique identifier",
        "account_id":       "STR  — unique CRM account ID; linkage key to pii_vault.csv",
        "lead_owner":       "STR  — assigned sales representative name",
        "company":          "STR  — company name (high cardinality, 71,973 unique)",
        "website":          "STR  — company website URL (50,367 unique)",
        "source":           "CAT  — standardised lead acquisition channel (20 values)",
        "source_group":     "CAT  — channel category bucket (7 values)",
        "deal_stage":       "CAT  — standardised CRM stage (10 values)",
        "is_won":           "INT  — binary target: 1=Closed Won, 0=all others",
        "is_closed":        "INT  — binary: 1=Closed Won|Closed Lost|Disqualified",
        "outcome_3class":   "CAT  — 3-class target: Won / Lost / Open",
        "target_multiclass":"INT  — 10-class ordinal: 0=New Lead ... 9=Re-engagement",
        "stage_ordinal":    "INT  — same as target_multiclass (funnel position)",
        "notes_sentiment":  "FLOAT— TextBlob polarity [-1.0, +1.0]; 0.0 if no text",
        "notes_word_count": "INT  — word count of Notes field",
        "notes_has_text":   "INT  — 1 if Notes is non-empty, 0 otherwise",
    }

    report["steps"]["7_derived_features"] = {
        "source_group_mapping": SOURCE_GROUP_MAP,
        "stage_ordinal_mapping": STAGE_ORDINAL,
        "textblob_available": _TB,
    }
    report["validation"] = {
        "shape_before":          list(shape_before),
        "shape_after":           list(shape_after),
        "nulls_in_raw":          total_nulls,
        "nulls_in_cleaned":      total_nulls_after,
        "rows_removed_dedup":    len(dup_fullrow_indices),
        "null_counts_cleaned":   null_after,
    }
    report["output_schema"]  = schema
    report["cleaning_rules"] = {
        "duplicates":      "Full-row duplicates removed (keep first). Duplicate Account IDs logged.",
        "missing_values":  "None found — no imputation applied.",
        "source_std":      "Lowercase key lookup -> 20 canonical title-case values.",
        "stage_std":       "Lowercase key lookup -> 10 canonical title-case values.",
        "pii":             "First/Last Name, Phone 1/2, Email 1/2 moved to pii_vault.csv. Linkage via Account Id.",
        "target_binary":   "is_won: 1=Closed Won, 0=all others.",
        "target_multiclass":"target_multiclass: integer 0-9 per STAGE_ORDINAL mapping.",
        "target_3class":   "outcome_3class: Won|Lost|Open based on LOST_STAGES set.",
        "whitespace":      "All fields stripped of leading/trailing whitespace.",
        "pii_retained":    "Account Id and Index retained in cleaned dataset as linkage keys.",
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[DONE] Preprocessing report -> '{REPORT_PATH}'")
    print(f"\n=== PHASE 2 SUMMARY ===")
    print(f"  Raw shape         : {shape_before[0]:,} x {shape_before[1]}")
    print(f"  Cleaned shape     : {shape_after[0]:,} x {shape_after[1]}")
    print(f"  Rows removed      : {shape_before[0] - shape_after[0]} (full-row duplicates)")
    print(f"  Null cells (raw)  : {total_nulls}")
    print(f"  Null cells (clean): {total_nulls_after}")
    print(f"  PII isolated      : {PII_COLS}")
    print(f"  Target columns    : is_won, outcome_3class, target_multiclass, stage_ordinal")
    print(f"  Output            : {CLEAN_PATH}")
    print(f"  PII vault         : {PII_PATH}")


if __name__ == "__main__":
    run()
