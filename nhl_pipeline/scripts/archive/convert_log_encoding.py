try:
    content = open('stint_validation_log.txt', encoding='utf-16').read()
except:
    content = open('stint_validation_log.txt', encoding='utf-8').read()

with open('stint_validation_log_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(content)
print("Log conversion complete.")
