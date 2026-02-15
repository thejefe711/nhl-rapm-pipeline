import os
import shutil
from pathlib import Path

BASE_DIR = Path("c:/Users/Andrew/Downloads/APM to LLM Web App")
REPORTS_DIR = BASE_DIR / "nhl_pipeline" / "reports"

CATEGORIES = {
    "validation": ["check", "validation", "verification", "audit", "triage", "status", "quality", "pass_rate"],
    "debug": ["debug", "inputs", "log", "stint", "output", "inspect", "discrepancy", "investigation"],
    "research": ["ranks", "profile", "ranks_ascii", "table", "comparison", "results", "analysis", "stats", "metrics_list", "top_block"],
    "archive": ["legacy", "old", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "utf8", "final", "corrected", "raw_counts"]
}

def get_category(filename):
    name = filename.lower()
    # Check for specific patterns first to avoid over-matching to research/debug
    if any(x in name for x in CATEGORIES["validation"]):
        return "validation"
    if any(x in name for x in CATEGORIES["debug"]):
        # Some research files might have 'results' or 'analysis', but if it has 'debug' it's usually debug
        if "debug" in name: return "debug"
        return "debug" # Default for logs/outputs
    if any(x in name for x in CATEGORIES["research"]):
         # Player specific files
        if any(p in name for p in ["mcdavid", "slavin", "jarvis", "bouchard", "fox"]):
            return "research"
        return "research"
    return "archive"

def move_text_files():
    # 1. From Repo Root
    for f in BASE_DIR.glob("*.txt"):
        cat = get_category(f.name)
        dest = REPORTS_DIR / cat / f.name
        print(f"Moving {f.name} to reports/{cat}/")
        shutil.move(str(f), str(dest))
    
    for f in BASE_DIR.glob("*.csv"):
        # Put CSVs in validation or archive
        if "slavin" in f.name.lower():
            cat = "research"
        elif "final" in f.name.lower() or "counts" in f.name.lower():
            cat = "validation"
        else:
            cat = "archive"
        dest = REPORTS_DIR / cat / f.name
        print(f"Moving {f.name} to reports/{cat}/")
        shutil.move(str(f), str(dest))

    # 2. From nhl_pipeline/ root
    p_dir = BASE_DIR / "nhl_pipeline"
    for f in p_dir.glob("*.txt"):
        cat = get_category(f.name)
        dest = REPORTS_DIR / cat / f.name
        print(f"Moving nhl_pipeline/{f.name} to reports/{cat}/")
        shutil.move(str(f), str(dest))

if __name__ == "__main__":
    move_text_files()
