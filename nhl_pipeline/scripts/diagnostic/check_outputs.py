import pandas as pd

# Check the pipeline outputs
files = [
    "player_rapm_full.csv",
    "player_categories.csv", 
    "player_clusters.csv",
    "player_similarity.csv",
    "player_flags.csv",
    "player_narratives.csv",
    "validated_profiles.csv"
]

for f in files:
    try:
        df = pd.read_csv(f"profile_data/{f}")
        print(f"\n{f}: {len(df)} rows, {len(df.columns)} cols")
        if "season" in df.columns:
            print(f"  Seasons: {df['season'].unique().tolist()[:3]}...")
        if "position_group" in df.columns:
            print(f"  Position groups: {df['position_group'].value_counts().to_dict()}")
    except Exception as e:
        print(f"{f}: ERROR - {e}")

# Check validated profiles specifically
print("\n=== Validated Profiles ===")
try:
    df = pd.read_csv("profile_data/validated_profiles.csv")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)[:10]}...")
    
    # Check 2024-2025 specifically
    current = df[df["season"] == "20242025"] if "season" in df.columns else df
    print(f"Current season rows: {len(current)}")
except Exception as e:
    print(f"Error: {e}")
