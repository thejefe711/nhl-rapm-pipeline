#!/usr/bin/env python3
"""
Teammate Attribution Models - Quantify how players impact teammates' performance.

Analyzes:
- How a player's presence affects teammates' individual stats
- Teammate RAPM when playing with/without specific players
- Line chemistry and compatibility scores
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def calculate_teammate_impact(player_id: int, season: str = "20242025") -> pd.DataFrame:
    """Calculate how a player impacts each teammate's performance."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get all teammates this player has played with
    teammates_query = """
        SELECT DISTINCT
            CASE
                WHEN home_player_1 = ? THEN home_player_2
                WHEN home_player_1 = ? THEN home_player_3
                WHEN home_player_2 = ? THEN home_player_1
                WHEN home_player_2 = ? THEN home_player_3
                WHEN home_player_3 = ? THEN home_player_1
                WHEN home_player_3 = ? THEN home_player_2
                WHEN home_player_4 = ? THEN home_player_5
                WHEN home_player_5 = ? THEN home_player_4
                WHEN away_player_1 = ? THEN away_player_2
                WHEN away_player_1 = ? THEN away_player_3
                WHEN away_player_2 = ? THEN away_player_1
                WHEN away_player_2 = ? THEN away_player_3
                WHEN away_player_3 = ? THEN away_player_1
                WHEN away_player_3 = ? THEN away_player_2
                WHEN away_player_4 = ? THEN away_player_5
                WHEN away_player_5 = ? THEN away_player_4
            END as teammate_id
        FROM stints
        WHERE season = ?
        AND (
            home_player_1 = ? OR home_player_2 = ? OR home_player_3 = ? OR
            home_player_4 = ? OR home_player_5 = ? OR
            away_player_1 = ? OR away_player_2 = ? OR away_player_3 = ? OR
            away_player_4 = ? OR away_player_5 = ?
        )
    """

    teammates_df = con.execute(teammates_query, [player_id] * 20 + [season]).df()
    teammates = teammates_df['teammate_id'].dropna().unique()

    results = []

    for teammate_id in teammates:
        if pd.isna(teammate_id) or teammate_id == player_id:
            continue

        # Calculate performance with and without this teammate
        with_teammate_query = """
            SELECT
                SUM(duration_s) as toi_with,
                AVG(net_corsi) as corsi_with,
                AVG(net_xg) as xg_with
            FROM stints
            WHERE season = ?
            AND duration_s >= 30
            AND (
                -- Both players together (forwards)
                ((home_player_1 = ? AND home_player_2 = ?) OR
                 (home_player_1 = ? AND home_player_3 = ?) OR
                 (home_player_2 = ? AND home_player_3 = ?) OR
                 (away_player_1 = ? AND away_player_2 = ?) OR
                 (away_player_1 = ? AND away_player_3 = ?) OR
                 (away_player_2 = ? AND away_player_3 = ?)) OR
                -- Both players together (defense)
                ((home_player_4 = ? AND home_player_5 = ?) OR
                 (away_player_4 = ? AND away_player_5 = ?))
            )
        """

        without_teammate_query = """
            SELECT
                SUM(duration_s) as toi_without,
                AVG(net_corsi) as corsi_without,
                AVG(net_xg) as xg_without
            FROM stints
            WHERE season = ?
            AND duration_s >= 30
            AND (
                -- Player A on ice, Player B not on ice
                ((home_player_1 = ? OR home_player_2 = ? OR home_player_3 = ? OR home_player_4 = ? OR home_player_5 = ?) AND
                 (home_player_1 != ? AND home_player_2 != ? AND home_player_3 != ? AND home_player_4 != ? AND home_player_5 != ?)) OR
                ((away_player_1 = ? OR away_player_2 = ? OR away_player_3 = ? OR away_player_4 = ? OR away_player_5 = ?) AND
                 (away_player_1 != ? AND away_player_2 != ? AND away_player_3 != ? AND away_player_4 != ? AND away_player_5 != ?))
            )
        """

        with_stats = con.execute(with_teammate_query, [season] + [player_id, teammate_id] * 8).df()
        without_stats = con.execute(without_teammate_query, [season] + [player_id] * 5 + [teammate_id] * 5 + [player_id] * 5 + [teammate_id] * 5).df()

        if with_stats.empty or without_stats.empty:
            continue

        with_data = with_stats.iloc[0]
        without_data = without_stats.iloc[0]

        # Only include if we have sufficient data
        min_toi = 300  # 5 minutes minimum
        if with_data['toi_with'] < min_toi or without_data['toi_without'] < min_toi:
            continue

        # Calculate impact
        corsi_impact = with_data['corsi_with'] - without_data['corsi_without']
        xg_impact = with_data['xg_with'] - without_data['xg_without']

        results.append({
            'teammate_id': teammate_id,
            'toi_with_teammate': with_data['toi_with'],
            'toi_without_teammate': without_data['toi_without'],
            'corsi_with': with_data['corsi_with'],
            'corsi_without': without_data['corsi_without'],
            'xg_with': with_data['xg_with'],
            'xg_without': without_data['xg_without'],
            'corsi_impact': corsi_impact,
            'xg_impact': xg_impact
        })

    con.close()

    if results:
        return pd.DataFrame(results).sort_values('xg_impact', ascending=False)
    else:
        return pd.DataFrame()

def get_player_rapm_impact(player_id: int, season: str = "20242025") -> Dict:
    """Calculate how a player's presence affects teammates' RAPM."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get teammates' RAPM when playing with/without this player
    rapm_impact_query = """
        SELECT
            CASE
                WHEN home_player_1 = ? THEN home_player_2
                WHEN home_player_1 = ? THEN home_player_3
                WHEN home_player_2 = ? THEN home_player_1
                WHEN home_player_2 = ? THEN home_player_3
                WHEN home_player_3 = ? THEN home_player_1
                WHEN home_player_3 = ? THEN home_player_2
                WHEN home_player_4 = ? THEN home_player_5
                WHEN home_player_5 = ? THEN home_player_4
                WHEN away_player_1 = ? THEN away_player_2
                WHEN away_player_1 = ? THEN away_player_3
                WHEN away_player_2 = ? THEN away_player_1
                WHEN away_player_2 = ? THEN away_player_3
                WHEN away_player_3 = ? THEN away_player_1
                WHEN away_player_3 = ? THEN away_player_2
                WHEN away_player_4 = ? THEN away_player_5
                WHEN away_player_5 = ? THEN away_player_4
            END as teammate_id,
            AVG(net_xg) as avg_xg_with,
            SUM(duration_s) as toi_with
        FROM stints
        WHERE season = ?
        AND duration_s >= 60
        AND teammate_id IS NOT NULL
        AND teammate_id != ?
        GROUP BY teammate_id
        HAVING toi_with >= 600  -- At least 10 minutes
        ORDER BY avg_xg_with DESC
    """

    with_impact = con.execute(rapm_impact_query, [player_id] * 16 + [season, player_id]).df()

    # Get RAPM for all players in this season
    all_rapm = con.execute("""
        SELECT player_id, value as rapm_xg
        FROM apm_results
        WHERE season = ?
        AND metric_name = 'xg_off_rapm_5v5'
    """, [season]).df()

    con.close()

    # Compare teammate performance with vs without
    impact_results = []
    rapm_dict = dict(zip(all_rapm['player_id'], all_rapm['rapm_xg']))

    for _, row in with_impact.iterrows():
        teammate_id = row['teammate_id']
        teammate_rapm = rapm_dict.get(teammate_id, 0)

        # Simplified impact calculation
        # In a real implementation, you'd need to calculate RAPM with/without this player
        impact = row['avg_xg_with'] - teammate_rapm  # This is approximate

        impact_results.append({
            'teammate_id': teammate_id,
            'teammate_rapm': teammate_rapm,
            'performance_with_player': row['avg_xg_with'],
            'estimated_impact': impact,
            'toi_with': row['toi_with']
        })

    return {
        'positive_impacts': sorted([r for r in impact_results if r['estimated_impact'] > 0.05],
                                  key=lambda x: x['estimated_impact'], reverse=True),
        'negative_impacts': sorted([r for r in impact_results if r['estimated_impact'] < -0.05],
                                  key=lambda x: x['estimated_impact']),
        'summary': {
            'num_teammates_boosted': len([r for r in impact_results if r['estimated_impact'] > 0.05]),
            'num_teammates_hurt': len([r for r in impact_results if r['estimated_impact'] < -0.05]),
            'avg_impact': np.mean([r['estimated_impact'] for r in impact_results]) if impact_results else 0
        }
    }

def get_player_names(player_ids: List[int]) -> Dict[int, str]:
    """Get player names for display."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    names = {}
    for player_id in player_ids:
        try:
            result = con.execute("SELECT full_name FROM players WHERE player_id = ?", [player_id]).fetchone()
            names[player_id] = result[0] if result else f"Player {player_id}"
        except:
            names[player_id] = f"Player {player_id}"

    con.close()
    return names

def analyze_player_impact(player_id: int, season: str = "20242025"):
    """Comprehensive analysis of a player's impact on teammates."""

    # Get player name
    player_names = get_player_names([player_id])
    player_name = player_names.get(player_id, f"Player {player_id}")

    print(f"TEAMMATE IMPACT ANALYSIS: {player_name}")
    print("=" * 60)

    # Get RAPM impact
    rapm_impact = get_player_rapm_impact(player_id, season)

    print("
RAPM IMPACT ON TEAMMATES:")
    print(f"Teammates boosted: {rapm_impact['summary']['num_teammates_boosted']}")
    print(f"Teammates hurt: {rapm_impact['summary']['num_teammates_hurt']}")
    print(".3f"
    print("\nTOP POSITIVE IMPACTS:")
    for impact in rapm_impact['positive_impacts'][:5]:
        teammate_name = get_player_names([impact['teammate_id']]).get(impact['teammate_id'], f"Player {impact['teammate_id']}")
        print(".3f"
    print("\nTOP NEGATIVE IMPACTS:")
    for impact in rapm_impact['negative_impacts'][:5]:
        teammate_name = get_player_names([impact['teammate_id']]).get(impact['teammate_id'], f"Player {impact['teammate_id']}")
        print(".3f"
    # Get detailed teammate impact
    detailed_impact = calculate_teammate_impact(player_id, season)

    if not detailed_impact.empty:
        print("
DETAILED PERFORMANCE IMPACT:")
        print("Top teammates by xG impact when playing together:")

        # Add player names
        teammate_ids = detailed_impact['teammate_id'].tolist()
        names_dict = get_player_names(teammate_ids)

        for _, row in detailed_impact.head(8).iterrows():
            teammate_name = names_dict.get(row['teammate_id'], f"Player {row['teammate_id']}")
            toi_with = row['toi_with_teammate'] / 60
            xg_impact = row['xg_impact']

            print(".1f"
def find_chemistry_pairs(season: str = "20242025", min_toi: int = 600):
    """Find player pairs with the strongest chemistry."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # This is a simplified approach - in practice you'd want more sophisticated chemistry detection
    chemistry_query = """
        SELECT
            LEAST(p1.player_id, p2.player_id) as player1_id,
            GREATEST(p1.player_id, p2.player_id) as player2_id,
            SUM(s.duration_s) as total_toi,
            AVG(s.net_xg) as avg_net_xg,
            COUNT(*) as num_stints
        FROM stints s
        JOIN players p1 ON (
            p1.player_id = s.home_player_1 OR p1.player_id = s.home_player_2 OR
            p1.player_id = s.home_player_3 OR p1.player_id = s.home_player_4 OR p1.player_id = s.home_player_5
        )
        JOIN players p2 ON (
            p2.player_id = s.away_player_1 OR p2.player_id = s.away_player_2 OR
            p2.player_id = s.away_player_3 OR p2.player_id = s.away_player_4 OR p2.player_id = s.away_player_5
        )
        WHERE s.season = ?
        AND s.duration_s >= 60
        AND p1.player_id < p2.player_id  -- Avoid duplicates
        GROUP BY player1_id, player2_id
        HAVING total_toi >= ?
        ORDER BY avg_net_xg DESC
        LIMIT 20
    """

    chemistry_df = con.execute(chemistry_query, [season, min_toi]).df()

    con.close()

    print(f"TOP CHEMISTRY PAIRS - {season}")
    print("=" * 50)

    if not chemistry_df.empty:
        names_dict = get_player_names(chemistry_df['player1_id'].tolist() + chemistry_df['player2_id'].tolist())

        for _, row in chemistry_df.head(10).iterrows():
            p1_name = names_dict.get(row['player1_id'], f"Player {row['player1_id']}")
            p2_name = names_dict.get(row['player2_id'], f"Player {row['player2_id']}")
            toi_minutes = row['total_toi'] / 60
            avg_xg = row['avg_net_xg']

            print(".1f"
if __name__ == "__main__":
    # Analyze Connor McDavid's impact on teammates
    analyze_player_impact(8478402, "20242025")

    print("\n" + "=" * 80)

    # Find top chemistry pairs
    find_chemistry_pairs("20242025")