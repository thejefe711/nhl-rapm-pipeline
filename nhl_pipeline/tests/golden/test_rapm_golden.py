import pytest
import pandas as pd
import os
from pathlib import Path

class TestRAPMGolden:
    """Compare RAPM output against manually verified golden files"""
    
    @pytest.fixture
    def golden_data_path(self):
        # Assuming golden files are stored in tests/golden/data
        return Path(__file__).parent / "data"

    def test_rapm_coefficients_match_golden(self, golden_data_path):
        """RAPM coefficients should match verified baseline"""
        golden_file = golden_data_path / "rapm_2024_golden.parquet"
        if not golden_file.exists():
            pytest.skip(f"Golden file not found at {golden_file}")
            
        # Fetch current results from DB
        import duckdb
        conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
        query = """
        SELECT player_id, metric_name, value 
        FROM apm_results 
        WHERE season = '20242025' AND metric_name IN ('xg_off_rapm_5v5', 'xg_def_rapm_5v5', 'goals_rapm_5v5')
        """
        current_df = conn.execute(query).fetchdf()
        if current_df.empty:
             pytest.fail("No current RAPM results found in DB")

        current_pivot = current_df.pivot(index='player_id', columns='metric_name', values='value').reset_index()
        
        golden = pd.read_parquet(golden_file)
        
        # Merge on player_id
        merged = current_pivot.merge(golden, on='player_id', suffixes=('_curr', '_gold'))
        
        # Check correlation and max diff
        for col in ['xg_off_rapm_5v5', 'xg_def_rapm_5v5']:
            if f'{col}_curr' not in merged.columns or f'{col}_gold' not in merged.columns:
                continue
                
            # Correlation check
            corr = merged[f'{col}_curr'].corr(merged[f'{col}_gold'])
            assert corr > 0.99, f"Correlation for {col} dropped to {corr}"
            
            # Max diff check
            diff = (merged[f'{col}_curr'] - merged[f'{col}_gold']).abs()
            # Allow small float diffs
            assert diff.max() < 0.001, f"Max diff for {col} is {diff.max()}"
