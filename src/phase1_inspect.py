import csv
from collections import Counter

with open('leads-100000.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

cols = list(rows[0].keys())
print('=== SCHEMA ===')
print(f'Rows: {len(rows):,}')
print(f'Columns ({len(cols)}): {cols}')

print()
print('=== NULL / EMPTY VALUES PER COLUMN ===')
for c in cols:
    empty = sum(1 for r in rows if not r[c].strip())
    print(f'  {c:<20} {empty}')

print()
print('=== DATA TYPE INFERENCE ===')
for c in cols:
    vals = [r[c].strip() for r in rows[:200] if r[c].strip()]
    numeric = sum(1 for v in vals if v.lstrip('-').replace('.','',1).isdigit())
    dtype = 'numeric' if numeric > len(vals)*0.8 else 'text/categorical'
    sample = vals[0][:50] if vals else ''
    print(f'  {c:<20} {dtype:<20} sample: {sample}')

print()
print('=== UNIQUE VALUES: Deal Stage ===')
stages = Counter(r['Deal Stage'].strip() for r in rows)
for k, v in sorted(stages.items(), key=lambda x: -x[1]):
    print(f'  {k:<25} {v:>7,}  ({v/len(rows)*100:.2f}%)')

print()
print('=== UNIQUE VALUES: Source ===')
sources = Counter(r['Source'].strip() for r in rows)
for k, v in sorted(sources.items(), key=lambda x: -x[1]):
    print(f'  {k:<30} {v:>7,}  ({v/len(rows)*100:.2f}%)')

print()
print('=== CARDINALITY CHECK (all columns) ===')
for c in cols:
    uniq = len(set(r[c].strip() for r in rows))
    print(f'  {c:<20} {uniq:>8,} unique values')

print()
print('=== NOTES FIELD SAMPLE (5 rows) ===')
for r in rows[:5]:
    print(f'  {r["Notes"][:80]}')

print()
print('=== LEAD OWNER DISTRIBUTION (top 5) ===')
owners = Counter(r['Lead Owner'].strip() for r in rows)
print(f'  Unique owners: {len(owners):,}')
print(f'  Max leads/owner: {max(owners.values())}')
print(f'  Owners with 1 lead: {sum(1 for v in owners.values() if v == 1):,}')
top5 = owners.most_common(5)
for name, cnt in top5:
    print(f'  {name:<30} {cnt} leads')

print()
print('=== CROSS-CHECK: Deal Stage x Source (Closed Won rates) ===')
total_by_src = Counter(r['Source'].strip() for r in rows)
won_by_src   = Counter(r['Source'].strip() for r in rows if r['Deal Stage'].strip() == 'Closed Won')
for src in sorted(total_by_src, key=lambda s: -won_by_src.get(s,0)/total_by_src[s]):
    t = total_by_src[src]
    w = won_by_src.get(src, 0)
    print(f'  {src:<30} {w}/{t} = {w/t*100:.2f}%')
