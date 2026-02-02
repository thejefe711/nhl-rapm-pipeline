try:
    content = open('verification_results.txt', encoding='utf-16').read()
except:
    content = open('verification_results.txt', encoding='utf-8').read()

with open('verification_results_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(content)
print("Conversion complete.")
