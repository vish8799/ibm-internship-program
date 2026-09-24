"""
Phase 3 - EDA & KPI Analytics
Computes all KPIs and EDA metrics from the processed dataset.
Outputs: reports/eda_kpi_results.json
"""

import csv
import json
import os
from collections import Counter, defaultdict
from statistics import mean, median, stdev

PROCESSED_PATH = "data/processed/leads_processed.csv"
OUTPUT_PATH    = "reports/eda_kpi_results.json"

# Funnel stage order (active pipeline stages only, in logical sequence)
FUNNEL_STAGES = [
    "New Lead", "Qualified", "Contacted",
    "Proposal Sent", "Negotiation",
    "Closed Won", "Closed Lost", "Disqualified",
    "On Hold", "Re-engagement",
]

def load():
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def run():
    os.makedirs("reports", exist_ok=True)
    rows = load()
    total = len(rows)
    print(f"[INFO] Loaded {total:,} processed records")

    results = {}

    # ------------------------------------------------------------------
    # KPI 1: Overall pipeline summary
    # ------------------------------------------------------------------
    won_count    = sum(1 for r in rows if r["is_won"] == "1")
    closed_count = sum(1 for r in rows if r["is_closed"] == "1")
    lost_count   = sum(1 for r in rows if r["deal_stage"] in ("Closed Lost", "Disqualified"))
    open_count   = sum(1 for r in rows if r["outcome"] == "Open")
    win_rate     = won_count / total * 100
    close_rate   = closed_count / total * 100
    disq_count   = sum(1 for r in rows if r["deal_stage"] == "Disqualified")

    results["kpi_summary"] = {
        "total_leads":          total,
        "closed_won":           won_count,
        "closed_lost":          lost_count,
        "disqualified":         disq_count,
        "open_pipeline":        open_count,
        "overall_win_rate_pct": round(win_rate, 2),
        "overall_close_rate_pct": round(close_rate, 2),
    }
    print(f"[KPI] Win rate: {win_rate:.2f}%  |  Closed: {closed_count:,}  |  Open: {open_count:,}")

    # ------------------------------------------------------------------
    # KPI 2: Deal stage distribution
    # ------------------------------------------------------------------
    stage_counts = Counter(r["deal_stage"] for r in rows)
    results["deal_stage_distribution"] = {
        s: {"count": stage_counts.get(s, 0),
            "pct": round(stage_counts.get(s, 0) / total * 100, 2)}
        for s in FUNNEL_STAGES
    }

    # ------------------------------------------------------------------
    # KPI 3: Win rate by Source (ranked)
    # ------------------------------------------------------------------
    source_total = Counter(r["source"] for r in rows)
    source_won   = Counter(r["source"] for r in rows if r["is_won"] == "1")
    win_by_source = []
    for src, tot in sorted(source_total.items()):
        won = source_won.get(src, 0)
        win_by_source.append({
            "source":       src,
            "total_leads":  tot,
            "won":          won,
            "win_rate_pct": round(won / tot * 100, 2),
        })
    # Sort by win rate descending
    win_by_source.sort(key=lambda x: -x["win_rate_pct"])
    results["win_rate_by_source"] = win_by_source

    # ------------------------------------------------------------------
    # KPI 4: Win rate by Source Group
    # ------------------------------------------------------------------
    sg_total = Counter(r["source_group"] for r in rows)
    sg_won   = Counter(r["source_group"] for r in rows if r["is_won"] == "1")
    win_by_sg = []
    for sg, tot in sorted(sg_total.items()):
        won = sg_won.get(sg, 0)
        win_by_sg.append({
            "source_group": sg,
            "total_leads":  tot,
            "won":          won,
            "win_rate_pct": round(won / tot * 100, 2),
        })
    win_by_sg.sort(key=lambda x: -x["win_rate_pct"])
    results["win_rate_by_source_group"] = win_by_sg

    # ------------------------------------------------------------------
    # KPI 5: Stage distribution by Source Group (heatmap data)
    # ------------------------------------------------------------------
    sg_stage = defaultdict(lambda: defaultdict(int))
    for r in rows:
        sg_stage[r["source_group"]][r["deal_stage"]] += 1
    results["stage_by_source_group"] = {
        sg: dict(stages)
        for sg, stages in sg_stage.items()
    }

    # ------------------------------------------------------------------
    # KPI 6: Funnel drop-off (active pipeline stages)
    # ------------------------------------------------------------------
    active_stages = ["New Lead", "Qualified", "Contacted",
                     "Proposal Sent", "Negotiation", "Closed Won"]
    funnel_counts = [stage_counts.get(s, 0) for s in active_stages]
    funnel_dropoff = []
    for i, (s, cnt) in enumerate(zip(active_stages, funnel_counts)):
        prev = funnel_counts[i-1] if i > 0 else cnt
        drop_pct = round((prev - cnt) / prev * 100, 2) if i > 0 and prev > 0 else 0.0
        funnel_dropoff.append({
            "stage": s,
            "count": cnt,
            "drop_off_pct_from_prev": drop_pct,
        })
    results["funnel_dropoff"] = funnel_dropoff

    # ------------------------------------------------------------------
    # KPI 7: Notes sentiment analysis
    # ------------------------------------------------------------------
    sentiments = [float(r["notes_sentiment"]) for r in rows]
    won_sent   = [float(r["notes_sentiment"]) for r in rows if r["is_won"] == "1"]
    lost_sent  = [float(r["notes_sentiment"]) for r in rows if r["deal_stage"] in ("Closed Lost", "Disqualified")]
    open_sent  = [float(r["notes_sentiment"]) for r in rows if r["outcome"] == "Open"]

    def safe_mean(lst): return round(mean(lst), 4) if lst else 0.0
    def safe_stdev(lst): return round(stdev(lst), 4) if len(lst) > 1 else 0.0

    # Sentiment bucket distribution
    def bucket(s):
        if s > 0.1:  return "Positive"
        if s < -0.1: return "Negative"
        return "Neutral"

    sent_buckets = Counter(bucket(s) for s in sentiments)
    won_sent_buckets  = Counter(bucket(s) for s in won_sent)
    lost_sent_buckets = Counter(bucket(s) for s in lost_sent)

    results["sentiment_analysis"] = {
        "overall": {
            "mean":     safe_mean(sentiments),
            "median":   round(median(sentiments), 4),
            "stdev":    safe_stdev(sentiments),
            "buckets":  dict(sent_buckets),
        },
        "by_outcome": {
            "Won":  {"mean": safe_mean(won_sent),  "stdev": safe_stdev(won_sent),  "buckets": dict(won_sent_buckets)},
            "Lost": {"mean": safe_mean(lost_sent), "stdev": safe_stdev(lost_sent), "buckets": dict(lost_sent_buckets)},
            "Open": {"mean": safe_mean(open_sent), "stdev": safe_stdev(open_sent)},
        },
    }

    # ------------------------------------------------------------------
    # KPI 8: Notes word count distribution
    # ------------------------------------------------------------------
    word_counts = [int(r["notes_word_count"]) for r in rows]
    results["notes_word_count"] = {
        "mean":   round(mean(word_counts), 2),
        "median": median(word_counts),
        "min":    min(word_counts),
        "max":    max(word_counts),
        "stdev":  round(stdev(word_counts), 2),
        "zero_notes": sum(1 for w in word_counts if w == 0),
    }

    # ------------------------------------------------------------------
    # KPI 9: Top 20 companies by lead volume
    # ------------------------------------------------------------------
    company_counts = Counter(r["company"] for r in rows)
    top_companies = [
        {"company": c, "leads": v}
        for c, v in company_counts.most_common(20)
    ]
    results["top_companies_by_leads"] = top_companies

    # ------------------------------------------------------------------
    # KPI 10: Outcome distribution by source group (stacked %)
    # ------------------------------------------------------------------
    sg_outcome = defaultdict(lambda: defaultdict(int))
    for r in rows:
        sg_outcome[r["source_group"]][r["outcome"]] += 1
    outcome_by_sg = {}
    for sg, outcomes in sg_outcome.items():
        tot = sum(outcomes.values())
        outcome_by_sg[sg] = {
            k: {"count": v, "pct": round(v / tot * 100, 2)}
            for k, v in outcomes.items()
        }
    results["outcome_by_source_group"] = outcome_by_sg

    # ------------------------------------------------------------------
    # KPI 11: Source performance ranking (composite score)
    # Composite = 0.7 * win_rate_normalised + 0.3 * volume_normalised
    # ------------------------------------------------------------------
    max_rate = max(x["win_rate_pct"] for x in win_by_source)
    min_rate = min(x["win_rate_pct"] for x in win_by_source)
    max_vol  = max(x["total_leads"]  for x in win_by_source)
    min_vol  = min(x["total_leads"]  for x in win_by_source)
    rate_range = max_rate - min_rate or 1
    vol_range  = max_vol  - min_vol  or 1

    ranked = []
    for x in win_by_source:
        norm_rate = (x["win_rate_pct"] - min_rate) / rate_range
        norm_vol  = (x["total_leads"]  - min_vol)  / vol_range
        score     = round(0.7 * norm_rate + 0.3 * norm_vol, 4)
        ranked.append({**x, "composite_score": score})
    ranked.sort(key=lambda x: -x["composite_score"])
    results["source_composite_ranking"] = ranked

    # ------------------------------------------------------------------
    # KPI 12: Stage ordinal stats by outcome
    # ------------------------------------------------------------------
    ord_won  = [int(r["stage_ordinal"]) for r in rows if r["is_won"] == "1"]
    ord_lost = [int(r["stage_ordinal"]) for r in rows if r["deal_stage"] in ("Closed Lost","Disqualified")]
    ord_open = [int(r["stage_ordinal"]) for r in rows if r["outcome"] == "Open"]
    results["stage_ordinal_by_outcome"] = {
        "Won":  {"mean": safe_mean(ord_won),  "median": median(ord_won)  if ord_won  else 0},
        "Lost": {"mean": safe_mean(ord_lost), "median": median(ord_lost) if ord_lost else 0},
        "Open": {"mean": safe_mean(ord_open), "median": median(ord_open) if ord_open else 0},
    }

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] EDA/KPI results -> '{OUTPUT_PATH}'")

    # ------------------------------------------------------------------
    # Print key findings summary
    # ------------------------------------------------------------------
    print("\n=== KEY FINDINGS ===")
    print(f"Total leads       : {total:,}")
    print(f"Overall win rate  : {win_rate:.2f}%")
    print(f"Closed Won        : {won_count:,}")
    print(f"Open pipeline     : {open_count:,}")
    print(f"\nTop 5 sources by win rate:")
    for x in win_by_source[:5]:
        print(f"  {x['source']:<30} {x['win_rate_pct']:.2f}%  ({x['total_leads']:,} leads)")
    print(f"\nBottom 5 sources by win rate:")
    for x in win_by_source[-5:]:
        print(f"  {x['source']:<30} {x['win_rate_pct']:.2f}%  ({x['total_leads']:,} leads)")
    print(f"\nSentiment mean (Won) : {results['sentiment_analysis']['by_outcome']['Won']['mean']}")
    print(f"Sentiment mean (Lost): {results['sentiment_analysis']['by_outcome']['Lost']['mean']}")
    print(f"Sentiment mean (Open): {results['sentiment_analysis']['by_outcome']['Open']['mean']}")
    print(f"\nFunnel drop-off (New Lead -> Closed Won):")
    for s in results["funnel_dropoff"]:
        print(f"  {s['stage']:<20} {s['count']:>6,}  (drop {s['drop_off_pct_from_prev']}% from prev)")

if __name__ == "__main__":
    run()
