try:
    content = open('test_stints_output.txt', encoding='utf-16').read()
except:
    content = open('test_stints_output.txt', encoding='utf-8').read()

with open('test_stints_output_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(content)
print("Log conversion complete.")
