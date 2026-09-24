"""
Phase 4 - SQL / Data Analytics
================================
Loads cleaned_leads.csv into a persistent SQLite database, executes all
analytical queries from sql/lead_analytics.sql, cross-validates results
against Phase 3 EDA KPIs, and writes:

  data/processed/leads_analytics.db  -- SQLite database (reusable)
  reports/sql_phase4_results.json     -- all query results
  reports/sql_phase4_validation.json  -- cross-validation report

Usage:
    python src/sql_phase4.py
"""

import csv
import json
import os
import re
import sqlite3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLEAN_PATH   = "data/processed/cleaned_leads.csv"
DB_PATH      = "data/processed/leads_analytics.db"
SQL_PATH     = "sql/lead_analytics.sql"
KPI_PATH     = "reports/eda_phase3_kpis.json"
RESULTS_PATH = "reports/sql_phase4_results.json"
VALID_PATH   = "reports/sql_phase4_validation.json"

os.makedirs("reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ---------------------------------------------------------------------------
# Phase 3 reference values for cross-validation
# ---------------------------------------------------------------------------
PHASE3_REF = {
    "overall_win_rate_pct":          9.99,
    "closed_rate_pct":               29.93,
    "podcast_win_rate_pct":          10.69,
    "podcast_total":                 5134,
    "podcast_won":                   549,
    "networking_event_win_rate_pct": 9.28,
    "networking_event_total":        4870,
    "networking_event_won":          452,
    "referral_partner_win_rate_pct": 10.63,
    "referral_partner_total":        10070,
    "referral_partner_won":          1070,
    "sentiment_won":                 0.0575,
    "sentiment_lost":                0.0646,
    "top_source":                    "Podcast",
    "bottom_source":                 "Networking Event",
    "total_leads":                   100000,
    "closed_won":                    9993,
    "source_group_best":             "Referral / Partner",
    "source_group_best_rate":        10.63,
}


# ---------------------------------------------------------------------------
# 1. Build / refresh the SQLite database
# ---------------------------------------------------------------------------
def build_db(conn: sqlite3.Connection) -> int:
    """Drop and recreate the leads table from cleaned_leads.csv."""
    conn.execute("DROP TABLE IF EXISTS leads")
    conn.execute("""
        CREATE TABLE leads (
            idx               INTEGER,
            account_id        TEXT,
            lead_owner        TEXT,
            company           TEXT,
            website           TEXT,
            source            TEXT,
            source_group      TEXT,
            deal_stage        TEXT,
            is_won            INTEGER,
            is_closed         INTEGER,
            outcome_3class    TEXT,
            target_multiclass INTEGER,
            stage_ordinal     INTEGER,
            notes_sentiment   REAL,
            notes_word_count  INTEGER,
            notes_has_text    INTEGER
        )
    """)
    # Create useful indices
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source       ON leads (source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source_group ON leads (source_group)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_deal_stage   ON leads (deal_stage)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_is_won       ON leads (is_won)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome      ON leads (outcome_3class)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_owner        ON leads (lead_owner)")

    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            (
                int(r["index"]),      r["account_id"],   r["lead_owner"],
                r["company"],         r["website"],       r["source"],
                r["source_group"],    r["deal_stage"],
                int(r["is_won"]),     int(r["is_closed"]),
                r["outcome_3class"],  int(r["target_multiclass"]),
                int(r["stage_ordinal"]),
                float(r["notes_sentiment"]),
                int(r["notes_word_count"]),
                int(r["notes_has_text"]),
            )
            for r in reader
        ]

    conn.executemany(
        "INSERT INTO leads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# 2. Parse and execute queries from the SQL file
# ---------------------------------------------------------------------------
def parse_queries(sql_path: str) -> list[dict]:
    """
    Parse sql/lead_analytics.sql into named query blocks.
    Each query is identified by a line starting with -- Q\\d+ immediately
    before the SELECT statement.
    """
    with open(sql_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on comment lines that start a named query block: -- Q\w+_NAME
    pattern = re.compile(r"--\s+(Q\d+\w+)\n(.*?)(?=--\s+Q\d+|\Z)", re.DOTALL)
    queries = []
    for m in pattern.finditer(content):
        name = m.group(1).strip()
        body = m.group(2).strip()
        # Extract only the SQL statement (skip any trailing comment lines)
        sql_lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("--") and not sql_lines:
                continue          # skip leading comment-only lines
            sql_lines.append(line)
        sql_body = "\n".join(sql_lines).strip().rstrip(";")
        if sql_body:
            queries.append({"name": name, "sql": sql_body})
    return queries


def run_query(conn: sqlite3.Connection, sql: str) -> list[dict]:
    """Execute a single SELECT and return list-of-dicts."""
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# 3. Cross-validation against Phase 3 EDA
# ---------------------------------------------------------------------------
def cross_validate(results: dict, conn: sqlite3.Connection) -> dict:
    """
    Compare SQL query outputs against known Phase 3 reference values.
    Spot-check queries are run directly here to guarantee exact key names.
    Returns a validation report with pass/fail per check.
    """
    checks = []
    failures = 0

    def check(name, got, expected, tolerance=0.02):
        nonlocal failures
        try:
            passed = abs(float(got) - float(expected)) <= tolerance
        except (TypeError, ValueError):
            passed = False
        if not passed:
            failures += 1
        checks.append({
            "check":    name,
            "expected": expected,
            "got":      got,
            "passed":   passed,
            "diff":     round(abs(float(got) - float(expected)), 4)
                        if got not in (None, -1) else 999,
        })

    def scalar(sql):
        """Run a single-cell query and return its value."""
        cur = conn.execute(sql)
        row = cur.fetchone()
        return row[0] if row else None

    # CV01 — overall win rate (from Q01_PIPELINE_SUMMARY)
    row = results.get("Q01_PIPELINE_SUMMARY", [{}])[0]
    check("Overall win rate (%)",
          row.get("win_rate_pct"),
          PHASE3_REF["overall_win_rate_pct"])

    # CV02 — pipeline totals
    check("Total leads",
          row.get("total_leads"),
          PHASE3_REF["total_leads"], tolerance=0)
    check("Closed Won count",
          row.get("closed_won"),
          PHASE3_REF["closed_won"], tolerance=0)
    check("Closed rate (%)",
          row.get("close_rate_pct"),
          PHASE3_REF["closed_rate_pct"])

    # CV03 — Podcast (from Q02A top-5; confirm rank 1 is Podcast)
    top5 = results.get("Q02A_TOP5_SOURCES_BY_WIN_RATE", [])
    top_src  = top5[0].get("source", "") if top5 else ""
    top_rate = top5[0].get("win_rate_pct") if top5 else None
    top_tot  = top5[0].get("total_leads")  if top5 else None
    top_won  = top5[0].get("closed_won")   if top5 else None
    passed_top = (top_src == PHASE3_REF["top_source"])
    if not passed_top: failures += 1
    checks.append({"check": "Top source name",
                   "expected": PHASE3_REF["top_source"],
                   "got": top_src, "passed": passed_top, "diff": 0})
    check("Podcast win rate (%)",  top_rate, PHASE3_REF["podcast_win_rate_pct"])
    check("Podcast total leads",   top_tot,  PHASE3_REF["podcast_total"],   tolerance=0)
    check("Podcast closed won",    top_won,  PHASE3_REF["podcast_won"],     tolerance=0)

    # CV04 — Networking Event (from Q02B bottom-5; rank 1 is worst)
    bot5    = results.get("Q02B_BOTTOM5_SOURCES_BY_WIN_RATE", [])
    bot_src = bot5[0].get("source", "") if bot5 else ""
    bot_rate= bot5[0].get("win_rate_pct") if bot5 else None
    bot_tot = bot5[0].get("total_leads")  if bot5 else None
    bot_won = bot5[0].get("closed_won")   if bot5 else None
    passed_bot = (bot_src == PHASE3_REF["bottom_source"])
    if not passed_bot: failures += 1
    checks.append({"check": "Bottom source name",
                   "expected": PHASE3_REF["bottom_source"],
                   "got": bot_src, "passed": passed_bot, "diff": 0})
    check("Networking Event win rate (%)", bot_rate,
          PHASE3_REF["networking_event_win_rate_pct"])
    check("Networking Event total leads",  bot_tot,
          PHASE3_REF["networking_event_total"], tolerance=0)
    check("Networking Event closed won",   bot_won,
          PHASE3_REF["networking_event_won"],   tolerance=0)

    # CV05 — Referral / Partner group (from Q03)
    sg_rows = {r["source_group"]: r for r in
               results.get("Q03_SOURCE_GROUP_PERFORMANCE", [])}
    rp = sg_rows.get("Referral / Partner", {})
    check("Referral/Partner win rate (%)",
          rp.get("win_rate_pct"),
          PHASE3_REF["referral_partner_win_rate_pct"])
    check("Referral/Partner total leads",
          rp.get("total_leads"),
          PHASE3_REF["referral_partner_total"], tolerance=0)
    check("Referral/Partner closed won",
          rp.get("closed_won"),
          PHASE3_REF["referral_partner_won"], tolerance=0)
    passed_sg = (list(sg_rows.keys())[0] == PHASE3_REF["source_group_best"])
    if not passed_sg: failures += 1
    checks.append({"check": "Best source group name",
                   "expected": PHASE3_REF["source_group_best"],
                   "got": list(sg_rows.keys())[0] if sg_rows else "",
                   "passed": passed_sg, "diff": 0})

    # CV06 — sentiment by outcome (from Q09A)
    sent_rows = {r["outcome_3class"]: r for r in
                 results.get("Q09A_NOTES_STATS_BY_OUTCOME", [])}
    check("Sentiment mean (Won)",
          sent_rows.get("Won",  {}).get("avg_sentiment"),
          PHASE3_REF["sentiment_won"],  tolerance=0.001)
    check("Sentiment mean (Lost)",
          sent_rows.get("Lost", {}).get("avg_sentiment"),
          PHASE3_REF["sentiment_lost"], tolerance=0.001)

    return {
        "total_checks": len(checks),
        "passed":       len(checks) - failures,
        "failed":       failures,
        "all_passed":   failures == 0,
        "checks":       checks,
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def run():
    print("[STEP 1] Building SQLite database from cleaned_leads.csv...")
    conn = sqlite3.connect(DB_PATH)
    n_rows = build_db(conn)
    print(f"         Loaded {n_rows:,} rows -> '{DB_PATH}'")

    print("[STEP 2] Parsing SQL queries from sql/lead_analytics.sql...")
    queries = parse_queries(SQL_PATH)
    print(f"         Parsed {len(queries)} named query blocks")

    print("[STEP 3] Executing queries...")
    results = {}
    for q in queries:
        try:
            rows = run_query(conn, q["sql"])
            results[q["name"]] = rows
            print(f"  [OK]  {q['name']:<50} {len(rows)} rows")
        except Exception as e:
            print(f"  [ERR] {q['name']:<50} {e}")
            results[q["name"]] = {"error": str(e)}

    print("[STEP 4] Cross-validating against Phase 3 EDA KPIs...")
    validation = cross_validate(results, conn)
    conn.close()
    print(f"         Checks passed : {validation['passed']} / {validation['total_checks']}")
    if not validation["all_passed"]:
        print("         FAILED CHECKS:")
        for c in validation["checks"]:
            if not c["passed"]:
                print(f"           FAIL: {c['check']} — expected={c['expected']}, got={c['got']}")

    print("[STEP 5] Writing outputs...")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(VALID_PATH, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    print(f"[DONE] Query results  -> '{RESULTS_PATH}'")
    print(f"[DONE] Validation     -> '{VALID_PATH}'")
    print(f"[DONE] Database       -> '{DB_PATH}'")

    # ── Print key findings ─────────────────────────────────────────────
    print("\n=== PHASE 4 KEY SQL FINDINGS ===")

    print("\n  Top 5 sources by win rate (Q02A):")
    for r in results.get("Q02A_TOP5_SOURCES_BY_WIN_RATE", []):
        print(f"    #{r['win_rate_rank']}  {r['source']:<30} "
              f"{r['win_rate_pct']}%  ({r['total_leads']:,} leads)")

    print("\n  Bottom 5 sources by win rate (Q02B):")
    for r in results.get("Q02B_BOTTOM5_SOURCES_BY_WIN_RATE", []):
        print(f"    #{r['worst_rank']}  {r['source']:<30} "
              f"{r['win_rate_pct']}%  ({r['total_leads']:,} leads)")

    print("\n  Source group performance (Q03):")
    for r in results.get("Q03_SOURCE_GROUP_PERFORMANCE", []):
        print(f"    {r['source_group']:<25} {r['win_rate_pct']}%  "
              f"n={r['total_leads']:,}  won_share={r['won_share_pct']}%")

    print("\n  Lead owner load distribution (Q05C):")
    for r in results.get("Q05C_OWNER_LOAD_SIMPLIFIED", []):
        print(f"    {r['leads_per_owner']} lead(s)/owner: {r['owner_count']:,} owners")

    print("\n  Owner performance stats (Q05D):")
    for r in results.get("Q05D_OWNER_WIN_RATE_SUMMARY_STATS", []):
        print(f"    Owners(>=2 leads)={r['owners_with_gte2_leads']:,}  "
              f"avg_win={r['avg_win_rate_pct']}%  "
              f"range={r['min_win_rate_pct']}%-{r['max_win_rate_pct']}%")

    print("\n  Won-to-lost ratio top 5 / bottom 3 (Q07):")
    ratios = results.get("Q07_WON_TO_LOST_RATIO", [])
    for r in ratios[:5]:
        print(f"    {r['source']:<30} ratio={r['won_to_lost_ratio']}  ({r['win_loss_balance']})")
    print("    ...")
    for r in ratios[-3:]:
        print(f"    {r['source']:<30} ratio={r['won_to_lost_ratio']}  ({r['win_loss_balance']})")

    print("\n  Notes stats by outcome (Q09A):")
    for r in results.get("Q09A_NOTES_STATS_BY_OUTCOME", []):
        print(f"    {r['outcome_3class']:<8} n={r['lead_count']:,}  "
              f"avg_words={r['avg_word_count']}  avg_sent={r['avg_sentiment']}")

    print(f"\n  Cross-validation: {validation['passed']}/{validation['total_checks']} checks passed  "
          f"({'ALL PASS' if validation['all_passed'] else 'FAILURES DETECTED'})")


if __name__ == "__main__":
    run()
