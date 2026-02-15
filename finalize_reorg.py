import os
import shutil
import re
from pathlib import Path

BASE_DIR = Path("c:/Users/Andrew/Downloads/APM to LLM Web App")
SCRIPTS_DIR = BASE_DIR / "nhl_pipeline" / "scripts"

CATEGORIES = ["core", "diagnostic", "research", "utils", "archive"]

def get_category(filename):
    name = filename.lower()
    if any(x in name for x in ["fetch", "parse", "build_on_ice", "load_to_db", "compute", "api_server", "run_pipeline", "workflow"]):
        return "core"
    if any(x in name for x in ["check", "verify", "quality", "report", "inspect", "investigate", "test", "audit", "triage", "monitor"]):
        return "diagnostic"
    if any(x in name for x in ["show", "analyze", "compare", "player_info", "search", "lookup", "skill", "decomposition", "tracking", "roadmap", "demo", "similarity", "profile", "dist"]):
        return "research"
    if any(x in name for x in ["schema", "validation_history", "deploy", "migration", "postgres", "scaling", "bench", "registry", "golden", "clean_seasons", "setup", "retry"]):
        return "utils"
    if any(x in name for x in ["legacy", "old", "trash", "debug", "v2", "optimized", "convert", "recovery", "discrepancy", "fix"]):
        return "archive"
    return "archive"  # Default to archive for excess scripts

def update_paths(file_path, old_level, new_level):
    """Update Path(__file__).parent.parent to match new nesting."""
    if not file_path.exists():
        return
    
    content = file_path.read_text(encoding="utf-8")
    
    # If it was in nhl_pipeline/ root (level 1 from nhl_pipeline) 
    # and moves to nhl_pipeline/scripts/core/ (level 2 from nhl_pipeline)
    # it needs one more .parent
    
    # We'll look for .parent.parent and change to .parent.parent.parent if it was already deep
    # Or change .parent to .parent.parent if it was at root
    
    if ".parent.parent" in content:
        new_content = content.replace(".parent.parent", ".parent.parent.parent")
        file_path.write_text(new_content, encoding="utf-8")
    elif ".parent" in content:
        # Be careful not to replace .parent if it's not and shouldn't be
        # But usually in these scripts .parent refers to the script dir
        # We only want to replace it if it's used to find data folders
        pass

def move_files():
    # 1. From Repo Root
    for f in BASE_DIR.glob("*.py"):
        if f.name in ["verify_players_status.py", "finalize_reorg.py"]: continue # skip itself
        cat = get_category(f.name)
        dest = SCRIPTS_DIR / cat / f.name
        print(f"Moving {f.name} to {cat}/")
        shutil.move(str(f), str(dest))
        update_paths(dest, 0, 2) # from root (0) to scripts/cat (2)

    # 2. From nhl_pipeline/ root
    p_dir = BASE_DIR / "nhl_pipeline"
    for f in p_dir.glob("*.py"):
        if f.name == "__init__.py": continue
        cat = get_category(f.name)
        dest = SCRIPTS_DIR / cat / f.name
        print(f"Moving nhl_pipeline/{f.name} to {cat}/")
        shutil.move(str(f), str(dest))
        update_paths(dest, 1, 2) # from nhl_pipeline (1) to scripts/cat (2)

    # 3. From nhl_pipeline/scripts/ (existing)
    for f in SCRIPTS_DIR.glob("*.py"):
        cat = get_category(f.name)
        dest = SCRIPTS_DIR / cat / f.name
        print(f"Categorizing scripts/{f.name} to {cat}/")
        shutil.move(str(f), str(dest))
        update_paths(dest, 2, 2) # stay at level 2? Wait.
        # Originally in scripts/ was parent.parent
        # Now in scripts/cat/ it needs parent.parent.parent
        # So we ALWAYS update .parent.parent to .parent.parent.parent if moved one level deeper
        # but update_paths already does that.

if __name__ == "__main__":
    move_files()
