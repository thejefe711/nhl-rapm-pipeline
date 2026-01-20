#!/usr/bin/env python3
"""
Line RAPM Analysis - Evaluate performance of specific lines and pairings.

Analyzes how different combinations of players perform together:
- Forward lines (3 players)
- Defensive pairs (2 players)
- Line combinations over time
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import itertools

def get_line_combinations(season: str = "20242025") -> Dict[str, List[Tuple]]:
    """Extract common line combinations from game data."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get player shifts data - simplified approach since we don't have stints table
    shifts_df = con.execute("""
        SELECT
            game_id,
            player_id,
            team_id,
            period,
            shift_number,
            shift_start,
            shift_end
        FROM shifts
        WHERE season = ?
        ORDER BY game_id, period, shift_start
    """, [season]).df()

    con.close()

    # For now, return empty dict - this is a complex implementation that would need
    # proper stint reconstruction from shifts data
    # TODO: Implement proper line combination detection from shifts
    return {
        'forward_lines': {},
        'defensive_pairs': {},
        'full_lines': {}
    }

    line_combinations = {
        'forward_lines': defaultdict(int),
        'defensive_pairs': defaultdict(int),
        'full_lines': defaultdict(int)  # 3F + 2D
    }

    for _, stint in stints_df.iterrows():
        # Extract player IDs
        home_forwards = [stint[f'home_player_{i}'] for i in [1,2,3] if pd.notna(stint[f'home_player_{i}'])]
        home_defense = [stint[f'home_player_{i}'] for i in [4,5] if pd.notna(stint[f'home_player_{i}'])]
        away_forwards = [stint[f'away_player_{i}'] for i in [1,2,3] if pd.notna(stint[f'away_player_{i}'])]
        away_defense = [stint[f'away_player_{i}'] for i in [4,5] if pd.notna(stint[f'away_player_{i}'])]

        # Forward lines (3 players)
        if len(home_forwards) >= 3:
            line_key = tuple(sorted(home_forwards))
            line_combinations['forward_lines'][line_key] += stint['duration_s']

        if len(away_forwards) >= 3:
            line_key = tuple(sorted(away_forwards))
            line_combinations['forward_lines'][line_key] += stint['duration_s']

        # Defensive pairs (2 players)
        if len(home_defense) >= 2:
            pair_key = tuple(sorted(home_defense))
            line_combinations['defensive_pairs'][pair_key] += stint['duration_s']

        if len(away_defense) >= 2:
            pair_key = tuple(sorted(away_defense))
            line_combinations['defensive_pairs'][pair_key] += stint['duration_s']

        # Full lines (3F + 2D)
        if len(home_forwards) >= 3 and len(home_defense) >= 2:
            full_key = tuple(sorted(home_forwards + home_defense))
            line_combinations['full_lines'][full_key] += stint['duration_s']

        if len(away_forwards) >= 3 and len(away_defense) >= 2:
            full_key = tuple(sorted(away_forwards + away_defense))
            line_combinations['full_lines'][full_key] += stint['duration_s']

    # Filter to combinations with significant ice time (>10 minutes total)
    MIN_ICE_TIME = 600  # 10 minutes

    filtered_combinations = {}
    for combo_type, combinations in line_combinations.items():
        filtered_combinations[combo_type] = {
            combo: ice_time for combo, ice_time in combinations.items()
            if ice_time >= MIN_ICE_TIME
        }

    return filtered_combinations

def calculate_line_rapm(combinations: Dict[str, Dict], combo_type: str, season: str = "20242025") -> pd.DataFrame:
    """Calculate RAPM for line combinations."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    results = []

    for line_combo, total_ice_time in combinations[combo_type].items():
        # Get all stints for this line combination
        if combo_type == 'forward_lines':
            # For forward lines, look for stints where these 3 players are together
            player_cols = [f'home_player_{i}' for i in [1,2,3]] + [f'away_player_{i}' for i in [1,2,3]]
            condition_parts = []
            for i, player_id in enumerate(line_combo):
                player_conditions = [f"{col} = {player_id}" for col in player_cols]
                condition_parts.append(f"({' OR '.join(player_conditions)})")

            where_clause = " AND ".join(condition_parts)

        elif combo_type == 'defensive_pairs':
            player_cols = [f'home_player_{i}' for i in [4,5]] + [f'away_player_{i}' for i in [4,5]]
            condition_parts = []
            for i, player_id in enumerate(line_combo):
                player_conditions = [f"{col} = {player_id}" for col in player_cols]
                condition_parts.append(f"({' OR '.join(player_conditions)})")

            where_clause = " AND ".join(condition_parts)

        # Simplified query - get aggregate stats for this line
        query = f"""
            SELECT
                SUM(duration_s) as total_toi,
                AVG(net_corsi) as avg_net_corsi_per_60,
                AVG(net_xg) as avg_net_xg_per_60,
                COUNT(*) as num_stints
            FROM stints
            WHERE season = '{season}'
            AND duration_s >= 30
            AND {where_clause}
        """

        try:
            stats_df = con.execute(query).df()
            if not stats_df.empty and stats_df.iloc[0]['total_toi'] > 0:
                stats = stats_df.iloc[0]
                results.append({
                    'line_combo': line_combo,
                    'combo_type': combo_type,
                    'total_toi': stats['total_toi'],
                    'avg_net_corsi_per_60': stats['avg_net_corsi_per_60'],
                    'avg_net_xg_per_60': stats['avg_net_xg_per_60'],
                    'num_stints': stats['num_stints']
                })
        except Exception as e:
            print(f"Error processing {line_combo}: {e}")
            continue

    con.close()

    if results:
        return pd.DataFrame(results).sort_values('avg_net_xg_per_60', ascending=False)
    else:
        return pd.DataFrame()

def get_player_names(player_ids: List[int]) -> Dict[int, str]:
    """Get player names for display."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
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

def analyze_top_lines(season: str = "20242025", top_n: int = 10):
    """Analyze and display top-performing lines."""

    print(f"TOP LINE PERFORMANCE ANALYSIS - {season}")
    print("=" * 60)

    # Get line combinations
    combinations = get_line_combinations(season)

    # Analyze each type
    for combo_type in ['forward_lines', 'defensive_pairs', 'full_lines']:
        if not combinations[combo_type]:
            continue

        print(f"\n{combo_type.upper().replace('_', ' ')} ANALYSIS")
        print("-" * 40)

        # Calculate RAPM
        rapm_df = calculate_line_rapm(combinations, combo_type, season)

        if rapm_df.empty:
            print("No significant line combinations found.")
            continue

        # Get player names
        all_player_ids = set()
        for combo in rapm_df['line_combo']:
            all_player_ids.update(combo)

        player_names = get_player_names(list(all_player_ids))

        # Display top performers
        for i, (_, row) in enumerate(rapm_df.head(top_n).iterrows()):
            combo = row['line_combo']
            player_names_list = [player_names.get(pid, f"Player {pid}") for pid in combo]

            toi_minutes = row['total_toi'] / 60
            net_xg_60 = row['avg_net_xg_per_60']

            print(".0f")

def analyze_line_chemistry(player1: int, player2: int, season: str = "20242025"):
    """Analyze how two specific players perform together vs apart."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get stints where both players are on ice together
    together_query = """
        SELECT
            SUM(duration_s) as toi_together,
            AVG(net_corsi) as corsi_together,
            AVG(net_xg) as xg_together,
            COUNT(*) as stints_together
        FROM stints
        WHERE season = ?
        AND duration_s >= 30
        AND (
            (home_player_1 = ? AND home_player_2 = ?) OR
            (home_player_1 = ? AND home_player_3 = ?) OR
            (home_player_2 = ? AND home_player_3 = ?) OR
            (away_player_1 = ? AND away_player_2 = ?) OR
            (away_player_1 = ? AND away_player_3 = ?) OR
            (away_player_2 = ? AND away_player_3 = ?) OR
            -- Add defensive pair combinations
            (home_player_4 = ? AND home_player_5 = ?) OR
            (away_player_4 = ? AND away_player_5 = ?)
        )
    """

    together_stats = con.execute(together_query, [season] + [player1, player2] * 8).df()

    # Get stints where players are apart (but both playing)
    apart_query = """
        SELECT
            AVG(net_corsi) as corsi_apart,
            AVG(net_xg) as xg_apart
        FROM stints
        WHERE season = ?
        AND duration_s >= 30
        AND (
            -- Player 1 on ice, player 2 not
            ((home_player_1 = ? OR home_player_2 = ? OR home_player_3 = ? OR home_player_4 = ? OR home_player_5 = ?) AND
             (home_player_1 != ? AND home_player_2 != ? AND home_player_3 != ? AND home_player_4 != ? AND home_player_5 != ?)) OR
            ((away_player_1 = ? OR away_player_2 = ? OR away_player_3 = ? OR away_player_4 = ? OR away_player_5 = ?) AND
             (away_player_1 != ? AND away_player_2 != ? AND away_player_3 != ? AND away_player_4 != ? AND away_player_5 != ?))
        )
    """

    apart_stats = con.execute(apart_query, [season] + [player1] * 5 + [player2] * 5 + [player1] * 5 + [player2] * 5).df()

    con.close()

    # Get player names
    player_names = get_player_names([player1, player2])

    print(f"LINE CHEMISTRY ANALYSIS: {player_names.get(player1, f'Player {player1}')} + {player_names.get(player2, f'Player {player2}')}")
    print("=" * 80)

    if together_stats.empty or apart_stats.empty:
        print("Insufficient data for chemistry analysis.")
        return

    together = together_stats.iloc[0]
    apart = apart_stats.iloc[0]

    toi_together = together['toi_together'] / 60  # Convert to minutes

    print(".1f")
    print(f"Together - Net Corsi/60: {together['corsi_together']:+.2f}")
    print(f"Together - Net xG/60: {together['xg_together']:+.2f}")
    print(f"Apart - Net Corsi/60: {apart['corsi_apart']:+.2f}")
    print(f"Apart - Net xG/60: {apart['xg_apart']:+.2f}")

    # Chemistry differential
    corsi_chemistry = together['corsi_together'] - apart['corsi_apart']
    xg_chemistry = together['xg_together'] - apart['xg_apart']

    print("\nCHEMISTRY IMPACT:")
    print(".2f")
    print(".2f")
    if abs(corsi_chemistry) > 2 or abs(xg_chemistry) > 0.1:
        chemistry_level = "SIGNIFICANT" if abs(corsi_chemistry) > 5 else "MODERATE"
        direction = "POSITIVE" if corsi_chemistry > 0 else "NEGATIVE"
        print(f"Assessment: {chemistry_level} {direction} chemistry detected")
    else:
        print("Assessment: Neutral chemistry - no significant impact")

if __name__ == "__main__":
    # Analyze top lines
    analyze_top_lines("20242025", top_n=5)

    print("\n" + "=" * 80)

    # Example chemistry analysis (McDavid and Draisaitl)
    analyze_line_chemistry(8478402, 8477934, "20242025")