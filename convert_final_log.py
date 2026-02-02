try:
    content = open('stint_validation_final.txt', encoding='utf-16').read()
except:
    content = open('stint_validation_final.txt', encoding='utf-8').read()

with open('stint_validation_final_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(content)
print("Log conversion complete.")
