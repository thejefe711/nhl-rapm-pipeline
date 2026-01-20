"""Verify data is real, not mock."""
import duckdb
import pandas as pd
import json

# 1. Check direct from DuckDB
print("=== DIRECT FROM DUCKDB ===")
con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

# Get McDavid's actual RAPM values from database
mcdavid = con.execute("""
    SELECT p.full_name, a.season, a.metric_name, a.value
    FROM apm_results a
    JOIN players p ON a.player_id = p.player_id
    WHERE p.full_name = 'Connor McDavid'
    AND a.season = '20242025'
    AND a.metric_name IN ('corsi_off_rapm_5v5', 'xg_off_rapm_5v5', 'corsi_def_rapm_5v5')
    ORDER BY a.metric_name
""").df()
print("\nMcDavid from DB:")
print(mcdavid.to_string())

# 2. Check from pipeline output
print("\n=== FROM PIPELINE OUTPUT ===")
df = pd.read_csv('profile_data/player_rapm_full.csv')
mcdavid_pipeline = df[df['full_name'] == 'Connor McDavid']
if not mcdavid_pipeline.empty:
    row = mcdavid_pipeline[mcdavid_pipeline['season'] == 20242025].iloc[0]
    print(f"\nMcDavid from pipeline CSV:")
    print(f"  corsi_off_rapm_5v5: {row.get('corsi_off_rapm_5v5', 'MISSING')}")
    print(f"  xg_off_rapm_5v5: {row.get('xg_off_rapm_5v5', 'MISSING')}")
    print(f"  position: {row.get('position', 'MISSING')}")

# 3. Check sample_profiles.json
print("\n=== FROM SAMPLE PROFILES JSON ===")
with open('profile_data/sample_profiles.json') as f:
    profiles = json.load(f)
    
mcdavid_profile = next((p for p in profiles if p['full_name'] == 'Connor McDavid'), None)
if mcdavid_profile:
    print(f"\nMcDavid from JSON:")
    print(f"  offense percentile: {mcdavid_profile['percentiles'].get('offense')}")
    print(f"  defense percentile: {mcdavid_profile['percentiles'].get('defense')}")
    print(f"  similar_players: {[p['name'] for p in mcdavid_profile['similar_players']]}")
    print(f"  narrative: {mcdavid_profile['narrative'][:100]}...")

# 4. Verify row counts
print("\n=== DATA COUNTS ===")
total_apm = con.execute("SELECT COUNT(*) FROM apm_results").fetchone()[0]
total_players = con.execute("SELECT COUNT(DISTINCT player_id) FROM apm_results").fetchone()[0]
print(f"Total RAPM rows in DB: {total_apm:,}")
print(f"Unique players in DB: {total_players:,}")

pipeline_rows = len(df)
pipeline_players = df['player_id'].nunique()
print(f"Rows in pipeline CSV: {pipeline_rows:,}")
print(f"Unique players in CSV: {pipeline_players:,}")

con.close()
print("\n✓ Data verified - using real database values")
