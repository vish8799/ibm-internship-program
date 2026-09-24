-- =============================================================================
-- lead_analytics.sql
-- Phase 4 — SQL / Data Analytics
-- Dataset : leads (loaded from data/processed/cleaned_leads.csv)
-- Engine  : SQLite (executed via src/sql_phase4.py)
-- =============================================================================
-- Table schema (all columns from cleaned_leads.csv):
--   idx              INTEGER  -- original row index
--   account_id       TEXT     -- unique CRM account ID
--   lead_owner       TEXT     -- assigned sales rep
--   company          TEXT     -- company name
--   website          TEXT     -- company website
--   source           TEXT     -- 20-value acquisition channel
--   source_group     TEXT     -- 7-value channel bucket
--   deal_stage       TEXT     -- 10-value CRM stage
--   is_won           INTEGER  -- 1 = Closed Won, 0 = all others
--   is_closed        INTEGER  -- 1 = Won|Lost|Disqualified
--   outcome_3class   TEXT     -- Won / Lost / Open
--   target_multiclass INTEGER -- 0-9 ordinal stage label
--   stage_ordinal    INTEGER  -- same as target_multiclass
--   notes_sentiment  REAL     -- TextBlob polarity [-1,+1]
--   notes_word_count INTEGER  -- word count of Notes field
--   notes_has_text   INTEGER  -- 1 if Notes non-empty
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Q01  PIPELINE SUMMARY
-- Core KPIs: total leads, closed won, win rate, closed rate, open pipeline.
-- Cross-validates KPI-01 and KPI-02 from Phase 3 EDA.
-- ---------------------------------------------------------------------------
-- Q01_PIPELINE_SUMMARY
SELECT
    COUNT(*)                                            AS total_leads,
    SUM(is_won)                                         AS closed_won,
    SUM(CASE WHEN deal_stage = 'Closed Lost'
              OR deal_stage = 'Disqualified' THEN 1
         ELSE 0 END)                                    AS closed_lost_disq,
    SUM(CASE WHEN outcome_3class = 'Open'   THEN 1
         ELSE 0 END)                                    AS open_pipeline,
    ROUND(SUM(is_won) * 100.0 / COUNT(*), 2)           AS win_rate_pct,
    ROUND(SUM(is_closed) * 100.0 / COUNT(*), 2)        AS close_rate_pct
FROM leads;


-- ---------------------------------------------------------------------------
-- Q02  SOURCE PERFORMANCE — WIN RATE + VOLUME (ranked both ways)
-- Answers: Which sources convert best? Which drive the most volume?
-- Top 5 by win rate, bottom 5 by win rate, top 5 by volume.
-- Cross-validates KPI-03 and KPI-05.
-- ---------------------------------------------------------------------------
-- Q02A_TOP5_SOURCES_BY_WIN_RATE
SELECT
    source,
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS closed_won,
    ROUND(SUM(is_won) * 100.0 / COUNT(*), 2)       AS win_rate_pct,
    RANK() OVER (ORDER BY SUM(is_won)*1.0/COUNT(*) DESC) AS win_rate_rank
FROM leads
GROUP BY source
ORDER BY win_rate_pct DESC
LIMIT 5;

-- Q02B_BOTTOM5_SOURCES_BY_WIN_RATE
SELECT
    source,
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS closed_won,
    ROUND(SUM(is_won) * 100.0 / COUNT(*), 2)       AS win_rate_pct,
    RANK() OVER (ORDER BY SUM(is_won)*1.0/COUNT(*) ASC) AS worst_rank
FROM leads
GROUP BY source
ORDER BY win_rate_pct ASC
LIMIT 5;

-- Q02C_ALL_SOURCES_FULL_RANKING
SELECT
    source,
    source_group,
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS closed_won,
    SUM(CASE WHEN deal_stage='Closed Lost'
              OR deal_stage='Disqualified'
         THEN 1 ELSE 0 END)                         AS closed_lost_disq,
    SUM(CASE WHEN outcome_3class='Open' THEN 1
         ELSE 0 END)                                AS open_leads,
    ROUND(SUM(is_won)*100.0/COUNT(*), 2)            AS win_rate_pct,
    ROUND(
        CAST(SUM(is_won) AS REAL) /
        NULLIF(SUM(CASE WHEN deal_stage='Closed Lost' THEN 1 ELSE 0 END), 0)
    , 3)                                             AS won_to_lost_ratio,
    RANK() OVER (ORDER BY SUM(is_won)*1.0/COUNT(*) DESC) AS win_rate_rank,
    RANK() OVER (ORDER BY COUNT(*) DESC)             AS volume_rank
FROM leads
GROUP BY source
ORDER BY win_rate_rank;

-- Q02D_TOP5_SOURCES_BY_VOLUME
SELECT
    source,
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS closed_won,
    ROUND(SUM(is_won)*100.0/COUNT(*), 2)            AS win_rate_pct,
    RANK() OVER (ORDER BY COUNT(*) DESC)             AS volume_rank
FROM leads
GROUP BY source
ORDER BY total_leads DESC
LIMIT 5;


-- ---------------------------------------------------------------------------
-- Q03  SOURCE GROUP PERFORMANCE
-- Channel-category level: win rate, volume share, won count.
-- Cross-validates KPI-04.
-- ---------------------------------------------------------------------------
-- Q03_SOURCE_GROUP_PERFORMANCE
SELECT
    source_group,
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS closed_won,
    ROUND(SUM(is_won)*100.0/COUNT(*), 2)            AS win_rate_pct,
    ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM leads), 2)  AS volume_share_pct,
    ROUND(SUM(is_won)*100.0/(SELECT SUM(is_won) FROM leads), 2) AS won_share_pct
FROM leads
GROUP BY source_group
ORDER BY win_rate_pct DESC;


-- ---------------------------------------------------------------------------
-- Q04  DEAL STAGE DISTRIBUTION
-- Count and proportion of leads at each pipeline stage.
-- Cross-validates KPI-08.
-- ---------------------------------------------------------------------------
-- Q04_DEAL_STAGE_DISTRIBUTION
SELECT
    deal_stage,
    outcome_3class,
    COUNT(*)                                        AS lead_count,
    ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM leads), 2) AS pct_of_total
FROM leads
GROUP BY deal_stage, outcome_3class
ORDER BY lead_count DESC;


-- ---------------------------------------------------------------------------
-- Q05  LEAD OWNER PERFORMANCE
-- Volume, win count, win rate — for owners with >= 2 leads.
-- Owners with 1 lead yield binary 0%/100% — statistically unreliable.
-- Ranked by win rate (top) and by win count (top).
-- ---------------------------------------------------------------------------
-- Q05A_OWNER_PERFORMANCE_TOP10_BY_WIN_RATE
SELECT
    lead_owner,
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS wins,
    ROUND(SUM(is_won)*100.0/COUNT(*), 1)            AS win_rate_pct
FROM leads
GROUP BY lead_owner
HAVING COUNT(*) >= 2
ORDER BY win_rate_pct DESC, wins DESC
LIMIT 10;

-- Q05B_OWNER_PERFORMANCE_TOP10_BY_WIN_COUNT
SELECT
    lead_owner,
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS wins,
    ROUND(SUM(is_won)*100.0/COUNT(*), 1)            AS win_rate_pct
FROM leads
GROUP BY lead_owner
HAVING COUNT(*) >= 2
ORDER BY wins DESC, win_rate_pct DESC
LIMIT 10;

-- Q05C_OWNER_LOAD_DISTRIBUTION
-- How many owners hold 1, 2, 3, or 4 leads?
SELECT
    COUNT(*)                                        AS total_leads_assigned,
    COUNT(*)  AS total_leads_group,
    leads_count,
    num_owners
FROM (
    SELECT
        lead_owner,
        COUNT(*) AS leads_count,
        COUNT(*) OVER (PARTITION BY COUNT(*))       AS num_owners
    FROM leads
    GROUP BY lead_owner
)
GROUP BY leads_count
ORDER BY leads_count;

-- Simplified version of Q05C:
-- Q05C_OWNER_LOAD_SIMPLIFIED
SELECT
    leads_per_owner,
    COUNT(*) AS owner_count
FROM (
    SELECT lead_owner, COUNT(*) AS leads_per_owner
    FROM leads
    GROUP BY lead_owner
)
GROUP BY leads_per_owner
ORDER BY leads_per_owner;

-- Q05D_OWNER_WIN_RATE_SUMMARY_STATS
-- Mean and median win rate across multi-lead owners.
SELECT
    COUNT(DISTINCT lead_owner)                          AS owners_with_gte2_leads,
    ROUND(AVG(owner_win_rate), 2)                       AS avg_win_rate_pct,
    ROUND(MIN(owner_win_rate), 1)                       AS min_win_rate_pct,
    ROUND(MAX(owner_win_rate), 1)                       AS max_win_rate_pct
FROM (
    SELECT
        lead_owner,
        ROUND(SUM(is_won)*100.0/COUNT(*), 1)            AS owner_win_rate
    FROM leads
    GROUP BY lead_owner
    HAVING COUNT(*) >= 2
);


-- ---------------------------------------------------------------------------
-- Q06  HIGH-VALUE LEAD SEGMENTATION
-- Combines source and owner attributes to identify the highest-quality
-- lead segments: top-performing source x owner combinations.
-- ---------------------------------------------------------------------------
-- Q06A_HIGH_VALUE_SEGMENTS_SOURCE_X_OWNER
-- Owners with >= 2 leads, from top-5 sources by win rate.
SELECT
    source,
    lead_owner,
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS wins,
    ROUND(SUM(is_won)*100.0/COUNT(*), 1)            AS win_rate_pct
FROM leads
WHERE source IN ('Podcast','Partner Program','Referral','Webinars','Content Marketing')
  AND lead_owner IN (
      SELECT lead_owner FROM leads GROUP BY lead_owner HAVING COUNT(*) >= 2
  )
GROUP BY source, lead_owner
HAVING COUNT(*) >= 2
ORDER BY wins DESC, win_rate_pct DESC
LIMIT 20;

-- Q06B_HIGH_VALUE_SEGMENTS_SOURCE_GROUP_X_STAGE
-- Stage distribution within each source group — reveals where leads stall.
SELECT
    source_group,
    deal_stage,
    COUNT(*)                                        AS lead_count,
    ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER
          (PARTITION BY source_group), 2)           AS pct_within_group
FROM leads
GROUP BY source_group, deal_stage
ORDER BY source_group, lead_count DESC;

-- Q06C_OPEN_PIPELINE_BY_SOURCE_HIGH_POTENTIAL
-- Open leads from top-3 sources: largest untapped conversion opportunity.
SELECT
    source,
    deal_stage,
    COUNT(*)                                        AS open_leads,
    ROUND(
        -- Expected wins if these leads converted at the source's historical rate
        COUNT(*) * (
            SELECT ROUND(SUM(is_won)*1.0/COUNT(*), 4)
            FROM leads l2
            WHERE l2.source = l.source
        )
    , 0)                                             AS expected_wins_at_hist_rate
FROM leads l
WHERE outcome_3class = 'Open'
  AND source IN ('Podcast','Partner Program','Referral')
GROUP BY source, deal_stage
ORDER BY source, open_leads DESC;

-- Q06D_SENTIMENT_SEGMENTATION_BY_SOURCE_AND_OUTCOME
-- Average sentiment per source per outcome — tests whether sentiment
-- varies meaningfully at source level (cross-validates KPI-11 finding).
SELECT
    source,
    outcome_3class,
    COUNT(*)                                        AS lead_count,
    ROUND(AVG(notes_sentiment), 4)                 AS avg_sentiment,
    ROUND(AVG(notes_word_count), 2)                AS avg_word_count
FROM leads
GROUP BY source, outcome_3class
ORDER BY source, outcome_3class;


-- ---------------------------------------------------------------------------
-- Q07  WON-TO-LOST RATIO BY SOURCE
-- Identifies sources where losses exceed wins (ratio < 1.0).
-- Cross-validates KPI-06.
-- ---------------------------------------------------------------------------
-- Q07_WON_TO_LOST_RATIO
SELECT
    source,
    SUM(CASE WHEN deal_stage='Closed Won'  THEN 1 ELSE 0 END) AS won,
    SUM(CASE WHEN deal_stage='Closed Lost' THEN 1 ELSE 0 END) AS lost,
    SUM(CASE WHEN deal_stage='Disqualified' THEN 1 ELSE 0 END) AS disqualified,
    ROUND(
        CAST(SUM(CASE WHEN deal_stage='Closed Won' THEN 1 ELSE 0 END) AS REAL)
        / NULLIF(SUM(CASE WHEN deal_stage='Closed Lost' THEN 1 ELSE 0 END), 0)
    , 3)                                             AS won_to_lost_ratio,
    CASE
        WHEN SUM(CASE WHEN deal_stage='Closed Won' THEN 1 ELSE 0 END) >
             SUM(CASE WHEN deal_stage='Closed Lost' THEN 1 ELSE 0 END)
        THEN 'More Wins'
        WHEN SUM(CASE WHEN deal_stage='Closed Won' THEN 1 ELSE 0 END) <
             SUM(CASE WHEN deal_stage='Closed Lost' THEN 1 ELSE 0 END)
        THEN 'More Losses'
        ELSE 'Equal'
    END                                              AS win_loss_balance
FROM leads
GROUP BY source
ORDER BY won_to_lost_ratio DESC;


-- ---------------------------------------------------------------------------
-- Q08  FUNNEL DROP-OFF ANALYSIS
-- Counts leads at each active stage to quantify pipeline progression.
-- Cross-validates KPI-09.
-- ---------------------------------------------------------------------------
-- Q08_FUNNEL_STAGES
SELECT
    deal_stage,
    stage_ordinal,
    COUNT(*)                                        AS lead_count,
    ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM leads), 2) AS pct_total
FROM leads
WHERE deal_stage IN (
    'New Lead','Qualified','Contacted',
    'Proposal Sent','Negotiation','Closed Won'
)
GROUP BY deal_stage, stage_ordinal
ORDER BY stage_ordinal;


-- ---------------------------------------------------------------------------
-- Q09  NOTES TEXT PATTERN ANALYSIS
-- Word count and sentiment breakdown by outcome.
-- Cross-validates KPI-10 and KPI-11.
-- ---------------------------------------------------------------------------
-- Q09A_NOTES_STATS_BY_OUTCOME
SELECT
    outcome_3class,
    COUNT(*)                                        AS lead_count,
    ROUND(AVG(notes_word_count), 2)                AS avg_word_count,
    ROUND(AVG(notes_sentiment), 4)                 AS avg_sentiment,
    MIN(notes_word_count)                          AS min_words,
    MAX(notes_word_count)                          AS max_words,
    SUM(CASE WHEN notes_sentiment >  0.1 THEN 1 ELSE 0 END) AS positive_notes,
    SUM(CASE WHEN notes_sentiment < -0.1 THEN 1 ELSE 0 END) AS negative_notes,
    SUM(CASE WHEN notes_sentiment BETWEEN -0.1 AND 0.1
         THEN 1 ELSE 0 END)                        AS neutral_notes
FROM leads
GROUP BY outcome_3class
ORDER BY outcome_3class;

-- Q09B_NOTES_WORD_COUNT_QUARTILES
-- Quartile distribution using ntile-window to approximate quartile boundaries.
-- Returns per-row ntile labels; aggregate in outer query for summary stats.
SELECT
    outcome_3class,
    MIN(notes_word_count)                           AS q0_min,
    MAX(CASE WHEN ntile_4 = 1 THEN notes_word_count END) AS q1_max,
    MAX(CASE WHEN ntile_4 = 2 THEN notes_word_count END) AS q2_max,
    MAX(CASE WHEN ntile_4 = 3 THEN notes_word_count END) AS q3_max,
    MAX(notes_word_count)                           AS q4_max,
    COUNT(*)                                        AS n
FROM (
    SELECT
        outcome_3class,
        notes_word_count,
        NTILE(4) OVER (
            PARTITION BY outcome_3class
            ORDER BY notes_word_count
        ) AS ntile_4
    FROM leads
)
GROUP BY outcome_3class
ORDER BY outcome_3class;


-- ---------------------------------------------------------------------------
-- Q10  SOURCE COMPOSITE PERFORMANCE SCORE
-- Normalised rank combining win rate and volume.
-- (Mirrors KPI-12; values computed in Python due to SQLite window limits.)
-- ---------------------------------------------------------------------------
-- Q10_SOURCE_PERFORMANCE_MATRIX
SELECT
    source,
    source_group,
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS closed_won,
    ROUND(SUM(is_won)*100.0/COUNT(*), 2)            AS win_rate_pct,
    RANK() OVER (ORDER BY SUM(is_won)*1.0/COUNT(*) DESC) AS win_rate_rank,
    RANK() OVER (ORDER BY COUNT(*) DESC)             AS volume_rank
FROM leads
GROUP BY source
ORDER BY win_rate_rank;


-- ---------------------------------------------------------------------------
-- Q11  CROSS-VALIDATION SPOT CHECKS
-- Exact value checks used to confirm zero discrepancy with Phase 3 EDA.
-- ---------------------------------------------------------------------------
-- Q11A_CV_OVERALL_WIN_RATE  [expected: 9.99%]
SELECT ROUND(SUM(is_won)*100.0/COUNT(*), 2) AS win_rate_pct FROM leads;

-- Q11B_CV_PODCAST_WIN_RATE  [expected: 10.69%, 549 won, 5134 total]
SELECT
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS closed_won,
    ROUND(SUM(is_won)*100.0/COUNT(*), 2)           AS win_rate_pct
FROM leads
WHERE source = 'Podcast';

-- Q11C_CV_NETWORKING_EVENT_WIN_RATE  [expected: 9.28%, 452 won, 4870 total]
SELECT
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS closed_won,
    ROUND(SUM(is_won)*100.0/COUNT(*), 2)           AS win_rate_pct
FROM leads
WHERE source = 'Networking Event';

-- Q11D_CV_REFERRAL_PARTNER_GROUP  [expected: 10.63%, 1070 won, 10070 total]
SELECT
    COUNT(*)                                        AS total_leads,
    SUM(is_won)                                     AS closed_won,
    ROUND(SUM(is_won)*100.0/COUNT(*), 2)           AS win_rate_pct
FROM leads
WHERE source_group = 'Referral / Partner';

-- Q11E_CV_SENTIMENT_BY_OUTCOME  [expected: Won=0.0575, Lost=0.0646]
SELECT
    outcome_3class,
    ROUND(AVG(notes_sentiment), 4)                 AS avg_sentiment
FROM leads
GROUP BY outcome_3class
ORDER BY outcome_3class;
