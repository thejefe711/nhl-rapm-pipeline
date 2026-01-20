#!/usr/bin/env python3
"""
Advanced Attribution Models - Beyond traditional goals/assists.

Features:
- xG attribution (expected goals created/assisted)
- Shot contribution attribution
- Zone entry/exit attribution
- Puck possession attribution
- Micro-attribution for secondary assists
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def calculate_xg_attribution(season: str = "20242025") -> pd.DataFrame:
    """Calculate expected goals created and allowed attribution."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get play-by-play events with xG
    pbp_data = con.execute("""
        SELECT
            game_id,
            period,
            period_time,
            event_type,
            event_team_id,
            home_team_id,
            away_team_id,
            player_id_1 as shooter,
            player_id_2 as assister1,
            player_id_3 as assister2,
            xg as shot_xg,
            goal
        FROM events
        WHERE season = ?
        AND event_type IN ('SHOT', 'GOAL')
        ORDER BY game_id, period, period_time
    """, [season]).df()

    con.close()

    # Attribution logic
    player_xg_created = defaultdict(float)
    player_xg_allowed = defaultdict(float)
    player_goals_created = defaultdict(int)
    player_assists = defaultdict(int)

    for _, event in pbp_data.iterrows():
        shooter = event['shooter']
        assister1 = event['assister1']
        assister2 = event['assister2']
        xg = event['shot_xg'] if not pd.isna(event['shot_xg']) else 0.0
        is_goal = event['goal']

        if pd.notna(shooter):
            if event['event_team_id'] == event['home_team_id']:
                # Home team shot - shooter creates xG, away team allows it
                player_xg_created[shooter] += xg
                if is_goal:
                    player_goals_created[shooter] += 1

                # Find players on ice for away team (allowing the xG)
                # This is simplified - in practice you'd need to join with stint data
                pass  # Would need stint lookup here

            else:
                # Away team shot
                player_xg_created[shooter] += xg
                if is_goal:
                    player_goals_created[shooter] += 1

        # Primary assist attribution
        if pd.notna(assister1):
            player_assists[assister1] += 1 if is_goal else 0
            # Attribute portion of xG to assister
            player_xg_created[assister1] += xg * 0.3  # 30% credit for primary assist

        # Secondary assist attribution
        if pd.notna(assister2):
            player_assists[assister2] += 1 if is_goal else 0
            # Attribute portion of xG to secondary assister
            player_xg_created[assister2] += xg * 0.1  # 10% credit for secondary assist

    # Convert to DataFrame
    results = []
    all_players = set(player_xg_created.keys()) | set(player_xg_allowed.keys()) | set(player_goals_created.keys()) | set(player_assists.keys())

    for player_id in all_players:
        results.append({
            'player_id': player_id,
            'xg_created': player_xg_created[player_id],
            'xg_allowed': player_xg_allowed[player_id],
            'goals_created': player_goals_created[player_id],
            'assists': player_assists[player_id]
        })

    return pd.DataFrame(results).sort_values('xg_created', ascending=False)

def calculate_zone_transition_attribution(season: str = "20242025") -> pd.DataFrame:
    """Calculate attribution for zone entries and exits."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get zone entry events
    zone_entries = con.execute("""
        SELECT
            game_id,
            period,
            period_time,
            event_type,
            event_team_id,
            player_id_1 as entry_player,
            zone_start,
            zone_end
        FROM events
        WHERE season = ?
        AND event_type = 'ZONE_ENTRY'
        ORDER BY game_id, period, period_time
    """, [season]).df()

    # Get subsequent xG in the zone
    # This is simplified - in practice you'd need to look at events 10-20 seconds after entry
    zone_xg = con.execute("""
        SELECT
            game_id,
            period,
            period_time,
            event_team_id,
            xg
        FROM events
        WHERE season = ?
        AND event_type IN ('SHOT', 'GOAL')
        AND xg IS NOT NULL
    """, [season]).df()

    con.close()

    # Attribution logic for zone entries
    player_entries_created = defaultdict(int)
    player_successful_entries = defaultdict(int)

    for _, entry in zone_entries.iterrows():
        player = entry['entry_player']
        if pd.notna(player):
            player_entries_created[player] += 1

            # Check if entry led to offensive zone success
            # Simplified: if they entered OZ, count as successful
            if entry['zone_end'] == 'O':  # Offensive zone
                player_successful_entries[player] += 1

    # Calculate entry success rates
    results = []
    for player_id in player_entries_created.keys():
        total_entries = player_entries_created[player_id]
        successful_entries = player_successful_entries[player_id]
        success_rate = successful_entries / total_entries if total_entries > 0 else 0

        results.append({
            'player_id': player_id,
            'total_zone_entries': total_entries,
            'successful_entries': successful_entries,
            'entry_success_rate': success_rate
        })

    return pd.DataFrame(results).sort_values('successful_entries', ascending=False)

def calculate_possession_attribution(season: str = "20242025") -> pd.DataFrame:
    """Calculate puck possession attribution."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get stint-level possession data
    possession_data = con.execute("""
        SELECT
            home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
            away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
            net_corsi,
            duration_s,
            home_team_id,
            away_team_id
        FROM stints
        WHERE season = ?
        AND duration_s >= 60
    """, [season]).df()

    con.close()

    # Attribution logic
    player_possession_time = defaultdict(float)
    player_corsi_contribution = defaultdict(float)

    for _, stint in possession_data.iterrows():
        duration = stint['duration_s']
        net_corsi = stint['net_corsi']

        # Home team possession
        home_players = []
        for i in [1,2,3,4,5]:
            player = stint[f'home_player_{i}']
            if pd.notna(player):
                home_players.append(player)

        if home_players:
            possession_per_player = duration / len(home_players)
            corsi_per_player = net_corsi / len(home_players)

            for player_id in home_players:
                player_possession_time[player_id] += possession_per_player
                player_corsi_contribution[player_id] += corsi_per_player

        # Away team possession (negative since corsi is net)
        away_players = []
        for i in [1,2,3,4,5]:
            player = stint[f'away_player_{i}']
            if pd.notna(player):
                away_players.append(player)

        if away_players:
            possession_per_player = duration / len(away_players)
            corsi_per_player = -net_corsi / len(away_players)  # Negative for away team

            for player_id in away_players:
                player_possession_time[player_id] += possession_per_player
                player_corsi_contribution[player_id] += corsi_per_player

    # Convert to DataFrame
    results = []
    for player_id in player_possession_time.keys():
        results.append({
            'player_id': player_id,
            'total_possession_seconds': player_possession_time[player_id],
            'corsi_contribution': player_corsi_contribution[player_id],
            'possession_minutes': player_possession_time[player_id] / 60
        })

    return pd.DataFrame(results).sort_values('corsi_contribution', ascending=False)

def calculate_secondary_assists(season: str = "20242025") -> pd.DataFrame:
    """Calculate secondary assist attribution beyond primary assists."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get all goals with assist chains
    goals_data = con.execute("""
        SELECT
            game_id,
            period,
            period_time,
            player_id_1 as goal_scorer,
            player_id_2 as primary_assister,
            player_id_3 as secondary_assister
        FROM events
        WHERE season = ?
        AND event_type = 'GOAL'
        AND player_id_1 IS NOT NULL
    """, [season]).df()

    # Get stint data to find all players on ice
    stints_data = con.execute("""
        SELECT
            game_id,
            period,
            period_time,
            home_player_1, home_player_2, home_player_3, home_player_4, home_player_5,
            away_player_1, away_player_2, away_player_3, away_player_4, away_player_5,
            home_team_id,
            away_team_id
        FROM stints
        WHERE season = ?
    """, [season]).df()

    con.close()

    # Attribution logic for secondary assists
    secondary_assists = defaultdict(int)

    for _, goal in goals_data.iterrows():
        goal_scorer = goal['goal_scorer']
        primary_assister = goal['primary_assister']
        secondary_assister = goal['secondary_assister']

        # Find the stint at the time of the goal
        goal_stint = stints_data[
            (stints_data['game_id'] == goal['game_id']) &
            (stints_data['period'] == goal['period']) &
            (stints_data['period_time'] <= goal['period_time']) &
            (stints_data['period_time'] + 60 >= goal['period_time'])  # Within 1 minute window
        ]

        if goal_stint.empty:
            continue

        stint = goal_stint.iloc[0]
        goal_team = None

        # Determine which team scored
        # This is simplified - you'd need to check the event_team_id
        if pd.notna(goal_scorer):
            # Assume goal_scorer determines the team
            if goal_scorer in [stint[f'home_player_{i}'] for i in [1,2,3,4,5]]:
                goal_team = 'home'
                teammates = [stint[f'home_player_{i}'] for i in [1,2,3,4,5] if pd.notna(stint[f'home_player_{i}'])]
            elif goal_scorer in [stint[f'away_player_{i}'] for i in [1,2,3,4,5]]:
                goal_team = 'away'
                teammates = [stint[f'away_player_{i}'] for i in [1,2,3,4,5] if pd.notna(stint[f'away_player_{i}'])]
            else:
                continue

            # Remove goal scorer, primary assister, and secondary assister from consideration
            excluded_players = {goal_scorer}
            if pd.notna(primary_assister):
                excluded_players.add(primary_assister)
            if pd.notna(secondary_assister):
                excluded_players.add(secondary_assister)

            # Any remaining teammate gets a "secondary assist" attribution
            for teammate in teammates:
                if pd.notna(teammate) and teammate not in excluded_players:
                    secondary_assists[teammate] += 1

    # Convert to DataFrame
    results = []
    for player_id, assists in secondary_assists.items():
        results.append({
            'player_id': player_id,
            'secondary_assists': assists
        })

    return pd.DataFrame(results).sort_values('secondary_assists', ascending=False)

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

def analyze_advanced_attribution(season: str = "20242025"):
    """Comprehensive advanced attribution analysis."""

    print(f"ADVANCED ATTRIBUTION ANALYSIS - {season}")
    print("=" * 50)

    # xG Attribution
    print("\nXG CREATION LEADERS:")
    xg_df = calculate_xg_attribution(season)
    if not xg_df.empty:
        names_dict = get_player_names(xg_df['player_id'].tolist())
        for _, row in xg_df.head(10).iterrows():
            player_name = names_dict.get(row['player_id'], f"Player {row['player_id']}")
            print(".2f")
    # Zone Entry Attribution
    print("\nZONE ENTRY LEADERS:")
    zone_df = calculate_zone_transition_attribution(season)
    if not zone_df.empty:
        names_dict = get_player_names(zone_df['player_id'].tolist())
        for _, row in zone_df.head(10).iterrows():
            player_name = names_dict.get(row['player_id'], f"Player {row['player_id']}")
            success_rate = row['entry_success_rate']
            print(".1f")
    # Possession Attribution
    print("\nPOSSESSION LEADERS (by Corsi contribution):")
    possession_df = calculate_possession_attribution(season)
    if not possession_df.empty:
        names_dict = get_player_names(possession_df['player_id'].tolist())
        for _, row in possession_df.head(10).iterrows():
            player_name = names_dict.get(row['player_id'], f"Player {row['player_id']}")
            corsi = row['corsi_contribution']
            minutes = row['possession_minutes']
            print(".1f")
    # Secondary Assists
    print("\nSECONDARY ASSIST LEADERS:")
    secondary_df = calculate_secondary_assists(season)
    if not secondary_df.empty:
        names_dict = get_player_names(secondary_df['player_id'].tolist())
        for _, row in secondary_df.head(10).iterrows():
            player_name = names_dict.get(row['player_id'], f"Player {row['player_id']}")
            assists = row['secondary_assists']
            print(f"  {player_name}: {assists} secondary assists")

if __name__ == "__main__":
    analyze_advanced_attribution("20242025")