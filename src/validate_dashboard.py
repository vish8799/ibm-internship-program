import re

with open('dashboard/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

chart_ids  = re.findall(r'id="(chart-[^"]+)"', html)
init_calls = re.findall(r"makeChart\('(chart-[^']+)'", html)
print('Chart divs   :', sorted(set(chart_ids)))
print()
print('Init calls   :', sorted(set(init_calls)))
print()
missing = set(chart_ids) - set(init_calls)
print('Missing init :', missing if missing else 'NONE')

pages = re.findall(r'id="page-([^"]+)"', html)
tabs  = re.findall(r"showPage\('([^']+)'\)", html)
print()
print('Pages defined:', pages)
print('Tab targets  :', list(dict.fromkeys(tabs)))
orphan = set(tabs) - set(pages)
print('Orphan tabs  :', orphan if orphan else 'NONE')
print()
print('HTML size    :', len(html), 'chars')
print('VALIDATION PASSED' if not missing and not orphan else 'ISSUES FOUND')
