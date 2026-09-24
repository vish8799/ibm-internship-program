"""
Phase 4 - SQL / Data Analytics
Loads the processed dataset into an in-memory SQLite database and runs
a comprehensive set of analytical queries that answer meaningful business
questions about the leads pipeline.
Outputs: reports/sql_analytics_results.json
"""

import csv
import json
import os
import sqlite3

PROCESSED_PATH = "data/processed/leads_processed.csv"
OUTPUT_PATH    = "reports/sql_analytics_results.json"
DB_PATH        = "data/processed/leads.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_db(conn: sqlite3.Connection):
    """Create and populate the leads table from the processed CSV."""
    conn.execute("DROP TABLE IF EXISTS leads")
    conn.execute("""
        CREATE TABLE leads (
            idx             INTEGER,
            account_id      TEXT,
            source          TEXT,
            source_group    TEXT,
            deal_stage      TEXT,
            stage_ordinal   INTEGER,
            outcome         TEXT,
            is_won          INTEGER,
            is_closed       INTEGER,
            notes_sentiment REAL,
            notes_word_count INTEGER,
            lead_owner      TEXT,
            company         TEXT
        )
    """)
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            (
                int(r["index"]), r["account_id"], r["source"], r["source_group"],
                r["deal_stage"], int(r["stage_ordinal"]),
                r["outcome"], int(r["is_won"]), int(r["is_closed"]),
                float(r["notes_sentiment"]), int(r["notes_word_count"]),
                r["lead_owner"], r["company"],
            )
            for r in reader
        ]
    conn.executemany("INSERT INTO leads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    print(f"[DB] Loaded {len(rows):,} rows into SQLite")
    return len(rows)


def q(conn, sql, params=()):
    """Execute a query and return list-of-dicts."""
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Analytical queries
# ---------------------------------------------------------------------------

QUERIES = {

    # Q1 ── Overall pipeline summary
    "q1_pipeline_summary": """
        SELECT
            COUNT(*)                                    AS total_leads,
            SUM(is_won)                                 AS closed_won,
            SUM(CASE WHEN outcome='Lost' THEN 1 END)    AS closed_lost_disq,
            SUM(CASE WHEN outcome='Open' THEN 1 END)    AS open_pipeline,
            ROUND(SUM(is_won)*100.0/COUNT(*), 2)        AS win_rate_pct,
            ROUND(SUM(is_closed)*100.0/COUNT(*), 2)     AS close_rate_pct
        FROM leads
    """,

    # Q2 ── Win rate by source, ranked
    "q2_win_rate_by_source": """
        SELECT
            source,
            COUNT(*)                                AS total_leads,
            SUM(is_won)                             AS won,
            ROUND(SUM(is_won)*100.0/COUNT(*), 2)   AS win_rate_pct,
            RANK() OVER (ORDER BY SUM(is_won)*1.0/COUNT(*) DESC) AS rank
        FROM leads
        GROUP BY source
        ORDER BY win_rate_pct DESC
    """,

    # Q3 ── Win rate by source group
    "q3_win_rate_by_source_group": """
        SELECT
            source_group,
            COUNT(*)                                AS total_leads,
            SUM(is_won)                             AS won,
            ROUND(SUM(is_won)*100.0/COUNT(*), 2)   AS win_rate_pct
        FROM leads
        GROUP BY source_group
        ORDER BY win_rate_pct DESC
    """,

    # Q4 ── Deal stage distribution with pct
    "q4_deal_stage_distribution": """
        SELECT
            deal_stage,
            COUNT(*)                                AS lead_count,
            ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM leads), 2) AS pct_of_total
        FROM leads
        GROUP BY deal_stage
        ORDER BY lead_count DESC
    """,

    # Q5 ── Outcome breakdown per source (segmentation)
    "q5_outcome_by_source": """
        SELECT
            source,
            SUM(CASE WHEN outcome='Won'  THEN 1 ELSE 0 END) AS won,
            SUM(CASE WHEN outcome='Lost' THEN 1 ELSE 0 END) AS lost,
            SUM(CASE WHEN outcome='Open' THEN 1 ELSE 0 END) AS open,
            COUNT(*)                                         AS total,
            ROUND(SUM(is_won)*100.0/COUNT(*), 2)            AS win_rate_pct
        FROM leads
        GROUP BY source
        ORDER BY win_rate_pct DESC
    """,

    # Q6 ── Source efficiency: won per 1000 leads (throughput)
    "q6_source_throughput": """
        SELECT
            source,
            COUNT(*)                                            AS total_leads,
            SUM(is_won)                                         AS won,
            ROUND(SUM(is_won)*1000.0/COUNT(*), 1)              AS won_per_1000_leads,
            ROUND(SUM(is_closed)*100.0/COUNT(*), 2)            AS close_rate_pct
        FROM leads
        GROUP BY source
        ORDER BY won_per_1000_leads DESC
    """,

    # Q7 ── Stage distribution per source group (pivot-style)
    "q7_stage_by_source_group": """
        SELECT
            source_group,
            SUM(CASE WHEN deal_stage='New Lead'      THEN 1 END) AS new_lead,
            SUM(CASE WHEN deal_stage='Qualified'     THEN 1 END) AS qualified,
            SUM(CASE WHEN deal_stage='Contacted'     THEN 1 END) AS contacted,
            SUM(CASE WHEN deal_stage='Proposal Sent' THEN 1 END) AS proposal_sent,
            SUM(CASE WHEN deal_stage='Negotiation'   THEN 1 END) AS negotiation,
            SUM(CASE WHEN deal_stage='Closed Won'    THEN 1 END) AS closed_won,
            SUM(CASE WHEN deal_stage='Closed Lost'   THEN 1 END) AS closed_lost,
            SUM(CASE WHEN deal_stage='Disqualified'  THEN 1 END) AS disqualified,
            SUM(CASE WHEN deal_stage='On Hold'       THEN 1 END) AS on_hold,
            SUM(CASE WHEN deal_stage='Re-engagement' THEN 1 END) AS re_engagement,
            COUNT(*)                                              AS total
        FROM leads
        GROUP BY source_group
        ORDER BY total DESC
    """,

    # Q8 ── Sentiment quartile analysis by outcome
    "q8_sentiment_quartiles_by_outcome": """
        SELECT
            outcome,
            COUNT(*)                                    AS total,
            ROUND(AVG(notes_sentiment), 4)             AS avg_sentiment,
            ROUND(MIN(notes_sentiment), 4)             AS min_sentiment,
            ROUND(MAX(notes_sentiment), 4)             AS max_sentiment,
            SUM(CASE WHEN notes_sentiment > 0.1  THEN 1 END) AS positive_notes,
            SUM(CASE WHEN notes_sentiment < -0.1 THEN 1 END) AS negative_notes,
            SUM(CASE WHEN notes_sentiment BETWEEN -0.1 AND 0.1 THEN 1 END) AS neutral_notes
        FROM leads
        GROUP BY outcome
        ORDER BY avg_sentiment DESC
    """,

    # Q9 ── Top 15 companies by Closed Won count
    "q9_top_companies_by_wins": """
        SELECT
            company,
            COUNT(*)        AS total_leads,
            SUM(is_won)     AS closed_won,
            ROUND(SUM(is_won)*100.0/COUNT(*), 1) AS win_rate_pct
        FROM leads
        GROUP BY company
        HAVING COUNT(*) >= 2
        ORDER BY closed_won DESC, win_rate_pct DESC
        LIMIT 15
    """,

    # Q10 ── Leads stuck in early stages (potential pipeline risk)
    "q10_pipeline_risk_segments": """
        SELECT
            deal_stage,
            source_group,
            COUNT(*) AS lead_count
        FROM leads
        WHERE deal_stage IN ('New Lead', 'Qualified', 'On Hold', 'Re-engagement')
        GROUP BY deal_stage, source_group
        ORDER BY deal_stage, lead_count DESC
    """,

    # Q11 ── Source group contribution to total Closed Won
    "q11_source_group_won_share": """
        SELECT
            source_group,
            SUM(is_won)                                             AS closed_won,
            ROUND(SUM(is_won)*100.0/(SELECT SUM(is_won) FROM leads), 2) AS pct_of_all_won
        FROM leads
        GROUP BY source_group
        ORDER BY closed_won DESC
    """,

    # Q12 ── Closed Won vs Closed Lost ratio by source
    "q12_won_vs_lost_ratio": """
        SELECT
            source,
            SUM(CASE WHEN deal_stage='Closed Won'  THEN 1 END) AS won,
            SUM(CASE WHEN deal_stage='Closed Lost' THEN 1 END) AS lost,
            SUM(CASE WHEN deal_stage='Disqualified' THEN 1 END) AS disqualified,
            CASE
                WHEN SUM(CASE WHEN deal_stage='Closed Lost' THEN 1 END) > 0
                THEN ROUND(
                    CAST(SUM(CASE WHEN deal_stage='Closed Won' THEN 1 END) AS REAL) /
                    SUM(CASE WHEN deal_stage='Closed Lost' THEN 1 END), 3)
                ELSE NULL
            END AS won_to_lost_ratio
        FROM leads
        GROUP BY source
        ORDER BY won_to_lost_ratio DESC
    """,

    # Q13 ── Notes word count stats by outcome
    "q13_notes_wordcount_by_outcome": """
        SELECT
            outcome,
            COUNT(*)                                AS total,
            ROUND(AVG(notes_word_count), 2)        AS avg_word_count,
            MIN(notes_word_count)                  AS min_word_count,
            MAX(notes_word_count)                  AS max_word_count
        FROM leads
        GROUP BY outcome
        ORDER BY avg_word_count DESC
    """,

    # Q14 ── High-sentiment leads in open pipeline (potential upsell targets)
    "q14_high_sentiment_open_leads": """
        SELECT
            source,
            deal_stage,
            COUNT(*) AS lead_count,
            ROUND(AVG(notes_sentiment), 4) AS avg_sentiment
        FROM leads
        WHERE outcome = 'Open'
          AND notes_sentiment > 0.3
        GROUP BY source, deal_stage
        ORDER BY lead_count DESC
        LIMIT 20
    """,

    # Q15 -- win rate benchmark comparison across key segments
    "q15_win_rate_benchmark_summary": """
        SELECT
            'Overall'           AS segment,
            COUNT(*)            AS total_leads,
            SUM(is_won)         AS won,
            ROUND(SUM(is_won)*100.0/COUNT(*), 2) AS win_rate_pct
        FROM leads
        UNION ALL
        SELECT 'Top Source (Podcast)', COUNT(*), SUM(is_won),
               ROUND(SUM(is_won)*100.0/COUNT(*),2)
        FROM leads WHERE source='Podcast'
        UNION ALL
        SELECT 'Bottom Source (Networking Event)', COUNT(*), SUM(is_won),
               ROUND(SUM(is_won)*100.0/COUNT(*),2)
        FROM leads WHERE source='Networking Event'
        UNION ALL
        SELECT 'Source Group: Referral/Partner', COUNT(*), SUM(is_won),
               ROUND(SUM(is_won)*100.0/COUNT(*),2)
        FROM leads WHERE source_group='Referral / Partner'
        UNION ALL
        SELECT 'Source Group: Paid Ads', COUNT(*), SUM(is_won),
               ROUND(SUM(is_won)*100.0/COUNT(*),2)
        FROM leads WHERE source_group='Paid Ads'
    """,
}


def run():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Persist DB so dashboard can reuse it
    conn = sqlite3.connect(DB_PATH)
    load_db(conn)

    results = {}
    for name, sql in QUERIES.items():
        # Skip the placeholder entry (overwritten below)
        try:
            rows = q(conn, sql)
            results[name] = rows
            print(f"[SQL] {name}: {len(rows)} rows")
        except Exception as e:
            print(f"[ERR] {name}: {e}")
            results[name] = {"error": str(e)}

    conn.close()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] SQL results -> '{OUTPUT_PATH}'")

    # ── Print key findings ──────────────────────────────────────────────
    print("\n=== SQL ANALYTICS KEY FINDINGS ===")

    summary = results["q1_pipeline_summary"][0]
    print(f"\nQ1  Pipeline: {summary['total_leads']:,} leads | "
          f"Won={summary['closed_won']:,} | "
          f"Win Rate={summary['win_rate_pct']}% | "
          f"Close Rate={summary['close_rate_pct']}%")

    print("\nQ2  Top 5 sources by win rate:")
    for r in results["q2_win_rate_by_source"][:5]:
        print(f"    #{r['rank']} {r['source']:<30} {r['win_rate_pct']}%  ({r['total_leads']:,} leads)")

    print("\nQ3  Source group win rates:")
    for r in results["q3_win_rate_by_source_group"]:
        print(f"    {r['source_group']:<25} {r['win_rate_pct']}%  ({r['total_leads']:,} leads)")

    print("\nQ11 Source group share of all Closed Won:")
    for r in results["q11_source_group_won_share"]:
        print(f"    {r['source_group']:<25} {r['closed_won']:,} won  ({r['pct_of_all_won']}%)")

    print("\nQ12 Won:Lost ratio by source (top 5):")
    for r in results["q12_won_vs_lost_ratio"][:5]:
        print(f"    {r['source']:<30} ratio={r['won_to_lost_ratio']}")

    print("\nQ15 Benchmark comparison:")
    for r in results["q15_win_rate_benchmark_summary"]:
        print(f"    {r['segment']:<45} {r['win_rate_pct']}%  (n={r['total_leads']:,})")


if __name__ == "__main__":
    run()
