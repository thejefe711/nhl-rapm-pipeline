import os
import zipfile
import sys
from pathlib import Path

# Paths relative to this script: nhl_pipeline/scripts/utils/db_manager.py
# ROOT should be the repo root
ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = ROOT / "nhl_pipeline" / "nhl_canonical.duckdb"
ZIP_PATH = ROOT / "nhl_pipeline" / "nhl_canonical.duckdb.zip"

def pack():
    """Compress the DuckDB database into a zip file for GitHub."""
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return False
    
    print(f"Packing {DB_PATH.name} into {ZIP_PATH.name}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(DB_PATH, arcname=DB_PATH.name)
        
        size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
        print(f"Successfully packed! Zip size: {size_mb:.2f} MB")
        print(f"Note: You can now push {ZIP_PATH.name} to GitHub.")
        return True
    except Exception as e:
        print(f"Failed to pack database: {e}")
        return False

def unpack():
    """Extract the DuckDB database from the zip file."""
    if not ZIP_PATH.exists():
        print(f"Error: Zip file not found at {ZIP_PATH}")
        return False
    
    print(f"Unpacking {ZIP_PATH.name} into {DB_PATH.parent}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(DB_PATH.parent)
        
        size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        print(f"Successfully unpacked! Database size: {size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"Failed to unpack database: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python db_manager.py [pack|unpack]")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    if cmd == "pack":
        pack()
    elif cmd == "unpack":
        unpack()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
