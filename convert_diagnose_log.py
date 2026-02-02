try:
    content = open('diagnose_shifts.log', encoding='utf-16').read()
except:
    content = open('diagnose_shifts.log', encoding='utf-8').read()

with open('diagnose_shifts_utf8.log', 'w', encoding='utf-8') as f:
    f.write(content)
print("Log conversion complete.")
