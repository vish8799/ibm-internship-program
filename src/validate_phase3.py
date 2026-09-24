"""Phase 3 validation — checks all outputs are present and internally consistent."""
import json, os, csv

# 1. KPI file exists and has all 12 KPIs
with open("reports/eda_phase3_kpis.json") as f:
    kpis = json.load(f)
expected_kpi_keys = [f"kpi_{str(i).zfill(2)}" for i in range(1,13)]
for k in expected_kpi_keys:
    matches = [key for key in kpis if key.startswith(k)]
    assert matches, f"Missing KPI: {k}"
print(f"[OK] KPI file present with {len(kpis)} KPI groups")

# 2. All 12 figures saved
FIG_DIR = "reports/figures"
for i in range(1, 13):
    fname = f"fig_{str(i).zfill(2)}_"
    matches = [f for f in os.listdir(FIG_DIR) if f.startswith(fname)]
    assert matches, f"Missing figure: fig_{str(i).zfill(2)}_*"
    size = os.path.getsize(os.path.join(FIG_DIR, matches[0]))
    assert size > 10_000, f"Figure too small (likely blank): {matches[0]} ({size} bytes)"
print(f"[OK] All 12 figures present and non-empty in '{FIG_DIR}/'")

# 3. KPI-01 win rate matches raw data count
with open("data/processed/cleaned_leads.csv") as f:
    rows = list(csv.DictReader(f))
actual_won   = sum(1 for r in rows if r["is_won"] == "1")
actual_total = len(rows)
actual_rate  = round(actual_won / actual_total * 100, 2)
kpi_rate = kpis["kpi_01_overall_win_rate"]["win_rate_pct"]
assert abs(kpi_rate - actual_rate) < 0.01, f"Win rate mismatch: KPI={kpi_rate}, data={actual_rate}"
print(f"[OK] KPI-01 win rate {kpi_rate}% matches actual data {actual_rate}%")

# 4. KPI-03 source count = 20
assert len(kpis["kpi_03_win_rate_by_source"]) == 20
print(f"[OK] KPI-03 covers all 20 sources")

# 5. KPI-04 source group count = 7
assert len(kpis["kpi_04_win_rate_by_source_group"]) == 7
print(f"[OK] KPI-04 covers all 7 source groups")

# 6. KPI-08 stage count = 10
assert len(kpis["kpi_08_deal_stage_distribution"]) == 10
print(f"[OK] KPI-08 covers all 10 deal stages")

# 7. KPI-09 funnel has 6 stages
assert len(kpis["kpi_09_funnel_dropoff"]) == 6
print(f"[OK] KPI-09 funnel has 6 active stages")

# 8. KPI-10 and KPI-11 have Won/Lost/Open
for kpi_key, name in [("kpi_10_notes_wordcount_by_outcome", "wordcount"),
                       ("kpi_11_notes_sentiment_by_outcome", "sentiment")]:
    assert set(kpis[kpi_key].keys()) == {"Won","Lost","Open"}, f"Missing outcome in {kpi_key}"
print(f"[OK] KPI-10 and KPI-11 have Won/Lost/Open entries")

# 9. KPI-12 composite score — all 20 sources
assert len(kpis["kpi_12_source_composite_score"]) == 20
print(f"[OK] KPI-12 composite score covers 20 sources")

# 10. Report file
with open("reports/eda_phase3_report.json") as f:
    rpt = json.load(f)
assert "narrative" in rpt
assert "figures" in rpt
assert "correlation_vs_causation_statement" in rpt
assert len(rpt["figures"]) == 12
print(f"[OK] EDA report present with narrative and causal statement")

# 11. Sentiment values match raw computation
from statistics import mean
won_sents  = [float(r["notes_sentiment"]) for r in rows if r["outcome_3class"]=="Won"]
lost_sents = [float(r["notes_sentiment"]) for r in rows if r["outcome_3class"]=="Lost"]
kpi_won_sent  = kpis["kpi_11_notes_sentiment_by_outcome"]["Won"]["mean"]
kpi_lost_sent = kpis["kpi_11_notes_sentiment_by_outcome"]["Lost"]["mean"]
assert abs(kpi_won_sent  - round(mean(won_sents),  4)) < 0.0001
assert abs(kpi_lost_sent - round(mean(lost_sents), 4)) < 0.0001
print(f"[OK] Sentiment KPIs match raw data (Won={kpi_won_sent}, Lost={kpi_lost_sent})")

# 12. Raw file untouched
with open("leads-100000.csv") as f:
    raw = list(csv.DictReader(f))
assert len(raw) == 100000 and "First Name" in raw[0]
print(f"[OK] Raw file untouched (100,000 rows, PII present)")

print()
print("=== ALL PHASE 3 VALIDATION CHECKS PASSED ===")
print(f"  KPIs computed : {len(kpis)}")
print(f"  Figures saved : 12")
print(f"  Data source   : data/processed/cleaned_leads.csv ({len(rows):,} rows)")
print(f"  KPI-01 Win Rate    : {kpi_rate}%")
print(f"  KPI-03 Top Source  : {kpis['kpi_03_win_rate_by_source'][0]['source']} ({kpis['kpi_03_win_rate_by_source'][0]['win_rate_pct']}%)")
print(f"  KPI-11 Sent diff   : {round(abs(kpi_won_sent - kpi_lost_sent), 4)} (near-zero = no signal)")
