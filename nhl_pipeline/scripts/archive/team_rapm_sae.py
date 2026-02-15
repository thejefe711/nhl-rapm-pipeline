#!/usr/bin/env python3
"""
Team-Level RAPM and SAE Models - Analyze team performance comprehensively.

Features:
- Team RAPM: How much better/worse is a team than expected based on roster
- Team SAE: Latent team factors (pace, structure, special teams, etc.)
- Team efficiency metrics
- Roster composition analysis
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle

def calculate_team_rapm(season: str = "20242025") -> pd.DataFrame:
    """Calculate team RAPM - how much better/worse teams are than expected."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get team game results
    team_results = con.execute("""
        SELECT
            game_id,
            home_team_id,
            away_team_id,
            home_score,
            away_score,
            CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_win,
            CASE WHEN away_score > home_score THEN 1 ELSE 0 END as away_win
        FROM games
        WHERE season = ?
    """, [season]).df()

    # Get player RAPM ratings
    player_rapm = con.execute("""
        SELECT
            player_id,
            value as rapm_xg,
            team_id
        FROM apm_results ar
        JOIN players p ON ar.player_id = p.player_id
        WHERE season = ?
        AND metric_name = 'xg_off_rapm_5v5'
    """, [season]).df()

    # Get team rosters (players who played significant minutes)
    team_rosters = con.execute("""
        SELECT
            team_id,
            player_id,
            SUM(duration_s) / 3600.0 as hours_played
        FROM stints
        WHERE season = ?
        GROUP BY team_id, player_id
        HAVING hours_played >= 10  -- At least 10 hours played
    """, [season]).df()

    con.close()

    # Calculate expected team strength based on roster RAPM
    team_strength = {}
    for team_id in team_rosters['team_id'].unique():
        team_players = team_rosters[team_rosters['team_id'] == team_id]
        team_rapm_values = []

        for _, player in team_players.iterrows():
            player_rapm_val = player_rapm[player_rapm['player_id'] == player['player_id']]['rapm_xg']
            if not player_rapm_val.empty:
                # Weight by ice time
                weight = player['hours_played'] / team_players['hours_played'].sum()
                team_rapm_values.append(player_rapm_val.iloc[0] * weight)

        if team_rapm_values:
            team_strength[team_id] = np.mean(team_rapm_values)

    # Calculate actual vs expected performance
    team_stats = []

    for team_id in team_strength.keys():
        team_games = team_results[(team_results['home_team_id'] == team_id) | (team_results['away_team_id'] == team_id)]

        if team_games.empty:
            continue

        # Calculate actual points percentage
        wins = 0
        total_games = len(team_games)

        for _, game in team_games.iterrows():
            if game['home_team_id'] == team_id:
                wins += game['home_win']
            else:
                wins += game['away_win']

        actual_win_pct = wins / total_games if total_games > 0 else 0

        # Expected win percentage based on RAPM strength (simplified)
        expected_win_pct = 0.5 + (team_strength[team_id] * 0.1)  # Rough approximation
        expected_win_pct = np.clip(expected_win_pct, 0.1, 0.9)  # Keep reasonable bounds

        team_stats.append({
            'team_id': team_id,
            'total_games': total_games,
            'actual_win_pct': actual_win_pct,
            'expected_win_pct': expected_win_pct,
            'rapm_overperformance': actual_win_pct - expected_win_pct,
            'roster_rapm_strength': team_strength[team_id]
        })

    return pd.DataFrame(team_stats).sort_values('rapm_overperformance', ascending=False)

def build_team_sae_model(season: str = "20242025") -> Tuple[object, np.ndarray, List[str]]:
    """Build SAE model for team latent factors."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get team-level aggregated stats
    team_stats = con.execute("""
        SELECT
            CASE WHEN s.home_team_id = s.away_team_id THEN NULL ELSE s.home_team_id END as team_id,
            AVG(s.net_corsi) as avg_corsi,
            AVG(s.net_xg) as avg_xg,
            AVG(s.net_pen_taken) as avg_penalties,
            SUM(s.duration_s) as total_duration,
            COUNT(*) as num_stints
        FROM stints s
        WHERE s.season = ?
        AND s.duration_s >= 60
        GROUP BY team_id
        HAVING total_duration >= 3600  -- At least 1 hour total
    """, [season]).df()

    # Add game-level aggregates
    game_stats = con.execute("""
        SELECT
            g.home_team_id as team_id,
            AVG(CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END) as home_win_pct,
            AVG(g.home_score - g.away_score) as avg_goal_diff,
            COUNT(*) as games_played
        FROM games g
        WHERE g.season = ?
        GROUP BY g.home_team_id
    """, [season]).df()

    con.close()

    # Merge team stats
    team_features = team_stats.merge(game_stats, on='team_id', how='left').fillna(0)

    if team_features.empty:
        return None, None, []

    # Prepare features for SAE
    feature_cols = ['avg_corsi', 'avg_xg', 'avg_penalties', 'home_win_pct', 'avg_goal_diff']
    X = team_features[feature_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SAE (using sklearn's DictionaryLearning for sparse coding)
    from sklearn.decomposition import DictionaryLearning

    n_components = min(8, X_scaled.shape[0] - 1)  # Don't overfit
    sae = DictionaryLearning(n_components=n_components, alpha=1, random_state=42)
    sae.fit(X_scaled)

    # Get latent representations
    latent_codes = sae.transform(X_scaled)

    # Save model
    model_path = Path(__file__).parent.parent.parent / f"models/team_sae_{season}.pkl"
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump({
            'sae': sae,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'latent_codes': latent_codes,
            'team_ids': team_features['team_id'].values
        }, f)

    return sae, latent_codes, feature_cols

def analyze_team_latent_factors(season: str = "20242025"):
    """Analyze what team latent factors represent."""

    # Try to load existing model
    model_path = Path(__file__).parent.parent.parent / f"models/team_sae_{season}.pkl"

    if not model_path.exists():
        print(f"Building team SAE model for {season}...")
        sae, latent_codes, feature_cols = build_team_sae_model(season)
        if sae is None:
            print("Could not build team SAE model - insufficient data")
            return
    else:
        print(f"Loading existing team SAE model for {season}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            sae = model_data['sae']
            latent_codes = model_data['latent_codes']
            feature_cols = model_data['feature_cols']
            team_ids = model_data['team_ids']

    print(f"\nTEAM LATENT FACTORS ANALYSIS - {season}")
    print("=" * 50)

    # Analyze each latent dimension
    for dim_idx in range(latent_codes.shape[1]):
        print(f"\nLatent Dimension {dim_idx + 1}:")

        # Find teams with high/low activation
        activations = latent_codes[:, dim_idx]
        top_team_indices = np.argsort(activations)[-3:][::-1]  # Top 3
        bottom_team_indices = np.argsort(activations)[:3]  # Bottom 3

        print("  Top teams:")
        for idx in top_team_indices:
            team_id = team_ids[idx] if 'team_ids' in locals() else f"Team {idx}"
            activation = activations[idx]
            print(".3f"
        print("  Bottom teams:")
        for idx in bottom_team_indices:
            team_id = team_ids[idx] if 'team_ids' in locals() else f"Team {idx}"
            activation = activations[idx]
            print(".3f"
        # Try to interpret the dimension
        # Look at correlation with original features
        if 'model_data' in locals() and hasattr(model_data['sae'], 'components_'):
            components = model_data['sae'].components_
            dim_components = components[dim_idx]

            # Find strongest correlations
            top_features = np.argsort(np.abs(dim_components))[-3:][::-1]
            print("  Associated features:")
            for feat_idx in top_features:
                feature_name = feature_cols[feat_idx]
                weight = dim_components[feat_idx]
                print(".3f"
def get_team_efficiency_metrics(season: str = "20242025") -> pd.DataFrame:
    """Calculate comprehensive team efficiency metrics."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get team-level stats
    team_metrics = con.execute("""
        SELECT
            CASE WHEN home_team_id < away_team_id THEN home_team_id ELSE away_team_id END as team_id,
            AVG(CASE WHEN home_team_id < away_team_id THEN s.net_corsi ELSE -s.net_corsi END) as avg_corsi_for,
            AVG(CASE WHEN home_team_id < away_team_id THEN s.net_xg ELSE -s.net_xg END) as avg_xg_for,
            AVG(CASE WHEN home_team_id < away_team_id THEN s.net_pen_taken ELSE -s.net_pen_taken END) as avg_penalties_for,
            SUM(s.duration_s) / 3600.0 as total_hours
        FROM stints s
        JOIN games g ON s.game_id = g.game_id
        WHERE s.season = ?
        AND s.duration_s >= 60
        GROUP BY team_id
        HAVING total_hours >= 5
        ORDER BY avg_xg_for DESC
    """, [season]).df()

    con.close()

    if team_metrics.empty:
        return pd.DataFrame()

    # Calculate efficiency ratios
    team_metrics['xg_efficiency'] = team_metrics['avg_xg_for'] / (team_metrics['avg_corsi_for'] + 1e-6)
    team_metrics['penalty_efficiency'] = team_metrics['avg_penalties_for'] / team_metrics['total_hours']

    return team_metrics.sort_values('avg_xg_for', ascending=False)

def analyze_team_rapm(season: str = "20242025"):
    """Analyze team RAPM over/underperformance."""

    team_rapm_df = calculate_team_rapm(season)

    if team_rapm_df.empty:
        print("No team RAPM data available")
        return

    print(f"\nTEAM RAPM ANALYSIS - {season}")
    print("=" * 40)

    # Get team names (simplified - using team IDs for now)
    team_names = {team_id: f"Team {team_id}" for team_id in team_rapm_df['team_id']}

    print("\nOVERPERFORMING TEAMS (better than RAPM suggests):")
    for _, row in team_rapm_df.head(5).iterrows():
        team_name = team_names.get(row['team_id'], f"Team {row['team_id']}")
        overperf = row['rapm_overperformance']
        actual = row['actual_win_pct']
        expected = row['expected_win_pct']
        print(".3f"
    print("\nUNDERPERFORMING TEAMS (worse than RAPM suggests):")
    for _, row in team_rapm_df.tail(5).iterrows():
        team_name = team_names.get(row['team_id'], f"Team {row['team_id']}")
        overperf = row['rapm_overperformance']
        actual = row['actual_win_pct']
        expected = row['expected_win_pct']
        print(".3f"
if __name__ == "__main__":
    # Analyze team RAPM
    analyze_team_rapm("20242025")

    # Analyze team latent factors
    analyze_team_latent_factors("20242025")

    # Show team efficiency metrics
    efficiency_df = get_team_efficiency_metrics("20242025")
    if not efficiency_df.empty:
        print("
TEAM EFFICIENCY LEADERS:")
        print("=" * 30)
        for _, row in efficiency_df.head(5).iterrows():
            print(".3f"            print(".3f"