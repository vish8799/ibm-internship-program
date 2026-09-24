"""
Phase 2 — Data Preparation
Reads the raw CSV, validates schema, cleans data, engineers base features,
and writes a processed dataset to data/processed/leads_processed.csv.
The original dataset is never modified.
"""

import csv
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# TextBlob sentiment is optional — fall back gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    from textblob import TextBlob
    _TEXTBLOB_OK = True
except ImportError:
    _TEXTBLOB_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_PATH       = "leads-100000.csv"
PROCESSED_PATH = "data/processed/leads_processed.csv"
REPORT_PATH    = "data/processed/data_quality_report.json"

EXPECTED_COLUMNS = [
    "Index", "Account Id", "Lead Owner", "First Name", "Last Name",
    "Company", "Phone 1", "Phone 2", "Email 1", "Email 2",
    "Website", "Source", "Deal Stage", "Notes"
]

# Canonical valid values (used for validation only — not filtering)
VALID_SOURCES = {
    "Chatbot", "Cold Call", "Cold Email", "Content Marketing",
    "Direct Traffic", "Facebook Ads", "Google Ads", "LinkedIn Outreach",
    "Networking Event", "Organic Search (SEO)", "Other",
    "Partner Program", "Podcast", "Purchased List", "Referral",
    "Retargeting Ads", "Social Media", "Trade Show", "Webinars",
    "Website Form",
}

VALID_STAGES = {
    "New Lead", "Qualified", "Contacted", "Proposal Sent", "Negotiation",
    "Closed Won", "Closed Lost", "On Hold", "Disqualified", "Re-engagement",
}

# Funnel ordering used for ordinal encoding
STAGE_ORDER = [
    "New Lead", "Re-engagement", "Qualified", "Contacted",
    "Proposal Sent", "Negotiation", "On Hold",
    "Closed Won", "Closed Lost", "Disqualified",
]

# Binary outcome groupings
WON_STAGES  = {"Closed Won"}
LOST_STAGES = {"Closed Lost", "Disqualified"}
OPEN_STAGES = {"New Lead", "Qualified", "Contacted", "Proposal Sent",
               "Negotiation", "On Hold", "Re-engagement"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentiment(text: str) -> float:
    """Return polarity in [-1, 1] via TextBlob, or 0.0 if unavailable."""
    if not _TEXTBLOB_OK or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity


def _notes_word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _source_group(source: str) -> str:
    """Map each source into a higher-level channel category."""
    paid_ads   = {"Google Ads", "Facebook Ads", "Retargeting Ads"}
    outbound   = {"Cold Call", "Cold Email", "LinkedIn Outreach", "Purchased List"}
    inbound    = {"Organic Search (SEO)", "Direct Traffic", "Website Form",
                  "Content Marketing", "Chatbot"}
    events     = {"Trade Show", "Networking Event", "Webinars", "Podcast"}
    referral   = {"Referral", "Partner Program"}
    social     = {"Social Media"}
    if source in paid_ads:   return "Paid Ads"
    if source in outbound:   return "Outbound"
    if source in inbound:    return "Inbound"
    if source in events:     return "Events"
    if source in referral:   return "Referral / Partner"
    if source in social:     return "Social Media"
    return "Other"


def _outcome(stage: str) -> str:
    if stage in WON_STAGES:  return "Won"
    if stage in LOST_STAGES: return "Lost"
    return "Open"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run():
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    # --- 1. Load raw data ---------------------------------------------------
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_columns = reader.fieldnames or []
        rows = list(reader)

    total_raw = len(rows)
    print(f"[INFO] Loaded {total_raw:,} records from '{RAW_PATH}'")

    # --- 2. Schema validation -----------------------------------------------
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in raw_columns]
    extra_cols   = [c for c in raw_columns if c not in EXPECTED_COLUMNS]
    if missing_cols:
        print(f"[ERROR] Missing columns: {missing_cols}")
        sys.exit(1)
    print(f"[INFO] Schema OK — {len(raw_columns)} columns present")
    if extra_cols:
        print(f"[WARN] Unexpected extra columns (ignored): {extra_cols}")

    # --- 3. Per-row validation & enrichment --------------------------------
    quality = {
        "total_raw": total_raw,
        "missing_values_per_col": {},
        "invalid_source": 0,
        "invalid_stage": 0,
        "duplicate_account_ids": 0,
    }

    seen_ids   = {}
    dup_indices = []

    # Count missing per column
    for col in EXPECTED_COLUMNS:
        quality["missing_values_per_col"][col] = sum(
            1 for r in rows if not r[col].strip()
        )

    # Duplicate Account IDs
    for i, r in enumerate(rows):
        aid = r["Account Id"]
        if aid in seen_ids:
            dup_indices.append(i)
        else:
            seen_ids[aid] = i
    quality["duplicate_account_ids"] = len(dup_indices)

    # Invalid categoricals
    quality["invalid_source"] = sum(
        1 for r in rows if r["Source"].strip() not in VALID_SOURCES
    )
    quality["invalid_stage"] = sum(
        1 for r in rows if r["Deal Stage"].strip() not in VALID_STAGES
    )

    print(f"[INFO] Duplicate Account IDs : {quality['duplicate_account_ids']}")
    print(f"[INFO] Invalid Source values : {quality['invalid_source']}")
    print(f"[INFO] Invalid Stage values  : {quality['invalid_stage']}")
    print(f"[INFO] Missing values total  : {sum(quality['missing_values_per_col'].values())}")

    # --- 4. Build processed rows -------------------------------------------
    print("[INFO] Building processed dataset (sentiment scoring may take a moment)...")

    stage_ordinal = {s: i for i, s in enumerate(STAGE_ORDER)}
    processed = []
    total = len(rows)

    for idx, r in enumerate(rows):
        if idx % 20000 == 0:
            print(f"  ... {idx:,}/{total:,}")

        source      = r["Source"].strip()
        stage       = r["Deal Stage"].strip()
        notes       = r["Notes"].strip()

        sentiment   = _sentiment(notes)
        word_count  = _notes_word_count(notes)
        src_group   = _source_group(source)
        outcome     = _outcome(stage)
        is_won      = 1 if stage in WON_STAGES  else 0
        is_closed   = 1 if stage in (WON_STAGES | LOST_STAGES) else 0
        stage_ord   = stage_ordinal.get(stage, -1)

        processed.append({
            # identifiers (kept for reference, not used in ML)
            "index":       r["Index"],
            "account_id":  r["Account Id"],
            # analytical fields
            "source":      source,
            "source_group": src_group,
            "deal_stage":  stage,
            "stage_ordinal": stage_ord,
            "outcome":     outcome,
            "is_won":      is_won,
            "is_closed":   is_closed,
            # text features
            "notes_sentiment": round(sentiment, 4),
            "notes_word_count": word_count,
            # retained for dashboard display
            "lead_owner":  r["Lead Owner"].strip(),
            "company":     r["Company"].strip(),
        })

    # --- 5. Write processed CSV --------------------------------------------
    out_cols = list(processed[0].keys())
    with open(PROCESSED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(processed)

    quality["total_processed"] = len(processed)
    quality["textblob_available"] = _TEXTBLOB_OK

    # --- 6. Write quality report -------------------------------------------
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    print(f"\n[DONE] Processed dataset -> '{PROCESSED_PATH}'")
    print(f"[DONE] Quality report    -> '{REPORT_PATH}'")
    print(f"[DONE] Records written   : {len(processed):,}")


if __name__ == "__main__":
    run()
