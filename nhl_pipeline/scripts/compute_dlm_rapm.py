import duckdb
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import argparse
from typing import Dict, List, Optional

def get_rapm_columns(conn) -> List[str]:
    """Get all RAPM column names from apm_results by pivoting a sample or checking schema"""
    # Since apm_results is long-form, we need to know what metric_names exist
    result = conn.execute("""
        SELECT DISTINCT metric_name 
        FROM apm_results 
        WHERE metric_name LIKE '%rapm%'
    """).fetchall()
    return [row[0] for row in result]

def fit_dlm(values: np.ndarray) -> Optional[Dict]:
    """
    Fit Local Level Model to time series.
    Returns filtered/smoothed estimates and variances.
    """
    if len(values) < 2:
        return None
    
    values = values.reshape(-1, 1)
    
    # Initialize with empirical variance
    initial_var = np.var(values) if np.var(values) > 0 else 0.01
    
    kf = KalmanFilter(
        transition_matrices=np.array([[1]]),
        observation_matrices=np.array([[1]]),
        initial_state_mean=np.array([values[0, 0]]),
        initial_state_covariance=np.array([[initial_var]]),
        observation_covariance=np.array([[initial_var * 0.5]]),
        transition_covariance=np.array([[initial_var * 0.5]]),
        em_vars=['transition_covariance', 'observation_covariance']
    )
    
    # Fit with EM
    try:
        kf = kf.em(values, n_iter=10)
    except Exception as e:
        # Fallback if EM fails
        print(f"  DEBUG: EM failed, using initial params: {e}")
        pass
    
    # Filter and smooth
    filtered_means, filtered_covs = kf.filter(values)
    smoothed_means, smoothed_covs = kf.smooth(values)
    
    return {
        'filtered_means': filtered_means.flatten(),
        'filtered_vars': np.array([c[0, 0] for c in filtered_covs]),
        'smoothed_means': smoothed_means.flatten(),
        'smoothed_vars': np.array([c[0, 0] for c in smoothed_covs]),
        'process_var': float(kf.transition_covariance[0, 0]),
        'obs_var': float(kf.observation_covariance[0, 0])
    }

def main():
    parser = argparse.ArgumentParser(description="Compute DLM estimates for RAPM metrics")
    parser.add_argument("--db", type=str, default="nhl_canonical.duckdb", help="Path to DuckDB")
    args = parser.parse_args()

    conn = duckdb.connect(args.db)
    
    # Get all RAPM metrics
    rapm_metrics = get_rapm_columns(conn)
    print(f"Found {len(rapm_metrics)} RAPM metrics")
    
    # Get all player-season data in long form
    # We pivot it to wide form for easier time-series extraction per player
    print("Fetching and pivoting data...")
    raw_df = conn.execute("""
        SELECT player_id, season, metric_name, value
        FROM apm_results
        ORDER BY player_id, season
    """).fetchdf()
    
    # Pivot to wide: index=(player_id, season), columns=metric_name
    df_wide = raw_df.pivot(index=['player_id', 'season'], columns='metric_name', values='value').reset_index()
    
    # Get players with 2+ seasons
    player_counts = df_wide.groupby('player_id')['season'].count()
    eligible_players = player_counts[player_counts >= 2].index.tolist()
    print(f"Processing {len(eligible_players)} players with 2+ seasons")
    
    results = []
    
    for count, player_id in enumerate(eligible_players):
        if count % 100 == 0:
            print(f"  Progress: {count}/{len(eligible_players)} players")
            
        player_data = df_wide[df_wide['player_id'] == player_id].sort_values('season')
        seasons = player_data['season'].tolist()
        n_seasons = len(seasons)
        
        for metric in rapm_metrics:
            if metric not in player_data.columns:
                continue
                
            values = player_data[metric].values
            
            # Skip if all null or zero
            if pd.isna(values).all() or (values == 0).all():
                continue
            
            # Fill NaN with 0 for DLM
            values_clean = np.nan_to_num(values, nan=0.0)
            
            dlm_result = fit_dlm(values_clean)
            if dlm_result is None:
                continue
            
            # Store results for each season
            for i, season in enumerate(seasons):
                results.append({
                    'player_id': int(player_id),
                    'metric_name': metric,
                    'season': season,
                    'observed_value': float(values_clean[i]),
                    'filtered_mean': float(dlm_result['filtered_means'][i]),
                    'filtered_var': float(dlm_result['filtered_vars'][i]),
                    'smoothed_mean': float(dlm_result['smoothed_means'][i]),
                    'smoothed_var': float(dlm_result['smoothed_vars'][i]),
                    'process_variance': dlm_result['process_var'],
                    'observation_variance': dlm_result['obs_var'],
                    'n_seasons': n_seasons,
                    'is_projected': False
                })
    
    if not results:
        print("No results generated.")
        conn.close()
        return

    # Write to database
    print(f"Writing {len(results)} rows to dlm_rapm_estimates...")
    results_df = pd.DataFrame(results)
    
    conn.execute("DROP TABLE IF EXISTS dlm_rapm_estimates")
    conn.execute("""
        CREATE TABLE dlm_rapm_estimates (
            player_id INTEGER,
            metric_name VARCHAR,
            season VARCHAR,
            observed_value DOUBLE,
            filtered_mean DOUBLE,
            filtered_var DOUBLE,
            smoothed_mean DOUBLE,
            smoothed_var DOUBLE,
            process_variance DOUBLE,
            observation_variance DOUBLE,
            n_seasons INTEGER,
            is_projected BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (player_id, metric_name, season)
        )
    """)
    
    conn.execute("""
        INSERT INTO dlm_rapm_estimates (
            player_id, metric_name, season, observed_value, 
            filtered_mean, filtered_var, smoothed_mean, smoothed_var, 
            process_variance, observation_variance, n_seasons, is_projected
        ) SELECT * FROM results_df
    """)
    
    print(f"DONE. Wrote {len(results_df)} rows to dlm_rapm_estimates")
    conn.close()

if __name__ == "__main__":
    main()
