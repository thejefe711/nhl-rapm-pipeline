import sys

def read_log(path):
    encodings = ['utf-16', 'utf-16le', 'utf-8', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
                print(f"Successfully read with {enc}")
                for line in content.splitlines():
                    if "Saved to DuckDB" in line:
                        print(line)
                return
        except Exception:
            continue
    print("Failed to read file with any encoding")

if __name__ == "__main__":
    read_log("debug_rapm_inputs_event_based_v13.txt")
