try:
    content = open('debug_rapm_inputs_fixed_2.txt', encoding='utf-16').read()
except:
    content = open('debug_rapm_inputs_fixed_2.txt', encoding='utf-8').read()

with open('debug_rapm_inputs_fixed_2_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(content)
print("Log conversion complete.")
