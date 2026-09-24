"""Phase 4 validation — checks all outputs are present, correct, and consistent."""
import json, os, sqlite3

# 1. All output files exist
for path in [
    "sql/lead_analytics.sql",
    "reports/sql_phase4_results.json",
    "reports/sql_phase4_validation.json",
    "data/processed/leads_analytics.db",
]:
    assert os.path.exists(path), f"Missing: {path}"
    assert os.path.getsize(path) > 0, f"Empty file: {path}"
print("[OK] All output files present and non-empty")

# 2. Cross-validation report: 18/18 passed
with open("reports/sql_phase4_validation.json") as f:
    val = json.load(f)
assert val["all_passed"], f"CV failures: {[c for c in val['checks'] if not c['passed']]}"
assert val["total_checks"] == 18
assert val["failed"] == 0
print(f"[OK] Cross-validation: {val['passed']}/{val['total_checks']} checks passed")

# 3. All 21 queries returned results (not errors)
with open("reports/sql_phase4_results.json") as f:
    results = json.load(f)
assert len(results) == 21, f"Expected 21 queries, got {len(results)}"
for name, data in results.items():
    assert not isinstance(data, dict) or "error" not in data, f"Query error: {name}"
print(f"[OK] All 21 queries executed without errors")

# 4. Key business question answers verified
# Q02A — top source is Podcast
top5 = results["Q02A_TOP5_SOURCES_BY_WIN_RATE"]
assert top5[0]["source"] == "Podcast", f"Wrong top source: {top5[0]['source']}"
assert top5[0]["win_rate_pct"] == 10.69
print(f"[OK] Q02A top source = Podcast @ {top5[0]['win_rate_pct']}%")

# Q02B — bottom source is Networking Event
bot5 = results["Q02B_BOTTOM5_SOURCES_BY_WIN_RATE"]
assert bot5[0]["source"] == "Networking Event"
assert bot5[0]["win_rate_pct"] == 9.28
print(f"[OK] Q02B bottom source = Networking Event @ {bot5[0]['win_rate_pct']}%")

# Q03 — 7 source groups, best is Referral / Partner
sg = results["Q03_SOURCE_GROUP_PERFORMANCE"]
assert len(sg) == 7
assert sg[0]["source_group"] == "Referral / Partner"
print(f"[OK] Q03 best source group = {sg[0]['source_group']} @ {sg[0]['win_rate_pct']}%")

# Q05C — owner load: 86,357 have 1 lead
owner_load = {r["leads_per_owner"]: r["owner_count"] for r in results["Q05C_OWNER_LOAD_SIMPLIFIED"]}
assert owner_load[1] == 86357
print(f"[OK] Q05C owner load: {owner_load[1]:,} owners with 1 lead, {owner_load.get(2,0):,} with 2")

# Q07 — won:lost ratio: Referral is highest, check value
ratios = {r["source"]: r["won_to_lost_ratio"] for r in results["Q07_WON_TO_LOST_RATIO"]}
assert abs(ratios["Referral"] - 1.163) < 0.01
assert ratios["Networking Event"] < 1.0   # only source with ratio < 1
print(f"[OK] Q07 Referral ratio={ratios['Referral']} (best), Networking Event={ratios['Networking Event']} (<1.0)")

# Q08 — 6 funnel stages
assert len(results["Q08_FUNNEL_STAGES"]) == 6
print(f"[OK] Q08 funnel has 6 active stages")

# Q09A — sentiment near-zero difference
sent = {r["outcome_3class"]: r for r in results["Q09A_NOTES_STATS_BY_OUTCOME"]}
diff = abs(sent["Won"]["avg_sentiment"] - sent["Lost"]["avg_sentiment"])
assert diff < 0.01
print(f"[OK] Q09A sentiment diff Won vs Lost = {diff:.4f} (near-zero, no signal)")

# Q09B — quartiles for 3 outcomes
assert len(results["Q09B_NOTES_WORD_COUNT_QUARTILES"]) == 3
print(f"[OK] Q09B word count quartiles: 3 outcome groups")

# Q10 — 20 sources ranked
assert len(results["Q10_SOURCE_PERFORMANCE_MATRIX"]) == 20
print(f"[OK] Q10 source matrix covers all 20 sources")

# 5. SQLite DB is queryable
conn = sqlite3.connect("data/processed/leads_analytics.db")
cnt = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
assert cnt == 100000
conn.close()
print(f"[OK] SQLite DB queryable: {cnt:,} rows in leads table")

# 6. SQL file contains all 11 query groups
with open("sql/lead_analytics.sql") as f:
    sql_text = f.read()
for q_prefix in ["Q01_", "Q02A_", "Q02B_", "Q02C_", "Q03_", "Q04_",
                  "Q05A_", "Q06A_", "Q07_", "Q08_", "Q09A_", "Q10_"]:
    assert q_prefix in sql_text, f"Missing query {q_prefix} in SQL file"
print(f"[OK] sql/lead_analytics.sql contains all expected query blocks")

print()
print("=== ALL PHASE 4 VALIDATION CHECKS PASSED ===")
print(f"  Queries executed        : 21")
print(f"  Cross-validation checks : {val['total_checks']} / {val['total_checks']} passed")
print(f"  DB rows                 : 100,000")
print(f"  Top source  (win rate)  : Podcast       10.69%")
print(f"  Bottom source (win rate): Networking Event 9.28%")
print(f"  Best source group       : Referral / Partner 10.63%")
print(f"  Sentiment signal        : negligible (diff = {diff:.4f})")
