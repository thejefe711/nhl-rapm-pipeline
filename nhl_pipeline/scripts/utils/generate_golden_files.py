import duckdb
import pandas as pd
import os
from pathlib import Path

def generate_golden_files(season: int, output_dir: str = "nhl_pipeline/tests/golden/data"):
    conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating golden files for season {season}...")
    
    # Fetch RAPM results
    query = """
    SELECT player_id, metric_name, value 
    FROM apm_results 
    WHERE season = ? AND metric_name IN ('xg_off_rapm_5v5', 'xg_def_rapm_5v5', 'goals_rapm_5v5')
    """
    
    # Handle 8-digit season format if needed
    # Try both
    df = conn.execute(query, [str(season)]).fetchdf()
    if df.empty:
        season_long = f"{season}{season+1}"
        df = conn.execute(query, [season_long]).fetchdf()
        
    if df.empty:
        print(f"No RAPM results found for season {season}")
        return

    # Pivot to wide format: player_id, xg_off, xg_def, goals
    df_pivot = df.pivot(index='player_id', columns='metric_name', values='value').reset_index()
    
    # Save to parquet
    file_path = output_path / f"rapm_{season}_golden.parquet"
    df_pivot.to_parquet(file_path)
    print(f"Saved golden file to {file_path}")
    
    # Save top 20 for quick sanity check
    if 'xg_off_rapm_5v5' in df_pivot.columns:
        top_20 = df_pivot.nlargest(20, 'xg_off_rapm_5v5')[['player_id', 'xg_off_rapm_5v5']]
        top_20.to_csv(output_path / f"top20_xg_off_{season}.csv", index=False)
        print(f"Saved top 20 to {output_path / f'top20_xg_off_{season}.csv'}")

if __name__ == "__main__":
    generate_golden_files(2024)
