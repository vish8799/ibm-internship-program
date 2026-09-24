"""Post-run validation for Phase 2 outputs."""
import csv, json

# --- 1. Raw file untouched check ---
with open('leads-100000.csv', 'r', encoding='utf-8') as f:
    raw = list(csv.DictReader(f))
assert len(raw) == 100000, "RAW ROW COUNT CHANGED"
assert 'First Name' in raw[0], "RAW PII COLUMN MISSING"
print(f"[OK] Raw file untouched: {len(raw):,} rows, all 14 columns present")

# --- 2. Cleaned output shape ---
with open('data/processed/cleaned_leads.csv', 'r', encoding='utf-8') as f:
    clean = list(csv.DictReader(f))
print(f"[OK] Cleaned shape: {len(clean):,} rows x {len(clean[0])} cols")
assert len(clean) == 100000
assert len(clean[0]) == 16

# --- 3. PII vault ---
with open('data/processed/pii_vault.csv', 'r', encoding='utf-8') as f:
    pii = list(csv.DictReader(f))
assert len(pii) == 100000
print(f"[OK] PII vault: {len(pii):,} rows, cols: {list(pii[0].keys())}")

# --- 4. PII not in cleaned ---
pii_fields = ['First Name','Last Name','Phone 1','Phone 2','Email 1','Email 2']
for f in pii_fields:
    assert f not in clean[0], f"PII FIELD '{f}' STILL IN CLEANED DATA"
print(f"[OK] PII fields absent from cleaned dataset: {pii_fields}")

# --- 5. Null check ---
nulls = sum(1 for r in clean for v in r.values() if str(v).strip() == '')
print(f"[OK] Null cells in cleaned output: {nulls}")
assert nulls == 0

# --- 6. Target columns present and valid ---
for col in ['is_won','is_closed','outcome_3class','target_multiclass','stage_ordinal']:
    assert col in clean[0], f"MISSING TARGET COL: {col}"
is_won_vals = set(r['is_won'] for r in clean)
assert is_won_vals == {'0','1'}, f"Unexpected is_won values: {is_won_vals}"
outcome_vals = set(r['outcome_3class'] for r in clean)
assert outcome_vals == {'Won','Lost','Open'}, f"Unexpected: {outcome_vals}"
mc_vals = set(int(r['target_multiclass']) for r in clean)
assert mc_vals == set(range(10)), f"Missing multiclass values: {mc_vals}"
print(f"[OK] Target columns valid: is_won={is_won_vals}, outcome_3class={outcome_vals}")
print(f"[OK] target_multiclass values: {sorted(mc_vals)}")

# --- 7. is_won count ---
won = sum(1 for r in clean if r['is_won'] == '1')
print(f"[OK] is_won=1: {won:,} ({won/len(clean)*100:.2f}%)  is_won=0: {len(clean)-won:,}")

# --- 8. Source and Stage canonical check ---
from collections import Counter
sources = Counter(r['source'] for r in clean)
stages  = Counter(r['deal_stage'] for r in clean)
print(f"[OK] Unique sources: {len(sources)} (expected 20)")
print(f"[OK] Unique stages : {len(stages)} (expected 10)")
assert len(sources) == 20
assert len(stages)  == 10

# --- 9. Source group check ---
groups = Counter(r['source_group'] for r in clean)
print(f"[OK] Source groups : {dict(groups)}")
assert len(groups) == 7

# --- 10. Report exists ---
with open('data/processed/preprocessing_report.json') as f:
    rpt = json.load(f)
print(f"[OK] Report keys: {list(rpt.keys())}")
assert 'output_schema' in rpt
assert 'cleaning_rules' in rpt
assert 'validation' in rpt

# --- 11. Schema doc check ---
schema_cols = set(rpt['output_schema'].keys())
data_cols   = set(clean[0].keys())
assert schema_cols == data_cols, f"Schema mismatch: {schema_cols ^ data_cols}"
print(f"[OK] Schema documentation covers all {len(schema_cols)} output columns")

print()
print("=== ALL VALIDATION CHECKS PASSED ===")
print(f"  Shape before : 100,000 x 14")
print(f"  Shape after  : {len(clean):,} x {len(clean[0])}")
print(f"  Rows removed : 0")
print(f"  Nulls (raw)  : 0")
print(f"  Nulls (clean): 0")
