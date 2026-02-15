try:
    content = open('inspect_output.txt', encoding='utf-16').read()
except:
    content = open('inspect_output.txt', encoding='utf-8').read()

with open('inspect_output_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(content)
print("Log conversion complete.")
