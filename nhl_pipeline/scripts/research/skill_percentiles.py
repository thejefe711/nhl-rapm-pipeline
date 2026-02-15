#!/usr/bin/env python3
"""
Calculate skill percentiles for players using latent dimensions.

This gives interpretable skill rankings like:
- Playmaking: 95th percentile (elite)
- Shooting: 78th percentile (above average)
- Defense: 23rd percentile (below average)
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Map SAE dimensions to interpretable skill categories
SKILL_CATEGORIES = {
    "offensive": {
        "name": "Offensive Skill",
        "dimensions": ["Play driver", "Transition killer", "PP quarterback"],
        "description": "Play creation, puck movement, power play effectiveness"
    },
    "defensive": {
        "name": "Defensive Skill",
        "dimensions": ["Elite shutdown (HD)", "PK stopper", "Two-way profile"],
        "description": "Shot suppression, penalty killing, overall defensive impact"
    },
    "shooting": {
        "name": "Shooting",
        "dimensions": ["HD xG Offense"],  # Would need to map properly
        "description": "High-danger shot generation and finishing"
    },
    "transition": {
        "name": "Transition Game",
        "dimensions": ["Transition killer"],
        "description": "Puck movement and zone transitions"
    },
    "special_teams": {
        "name": "Special Teams",
        "dimensions": ["PP quarterback", "PK stopper"],
        "description": "Power play and penalty kill effectiveness"
    }
}

def load_latent_data(db_path: Path, season: str = "20242025") -> pd.DataFrame:
    """Load all latent skill data for a season."""
    con = duckdb.connect(str(db_path))

    df = con.execute("""
        SELECT
            player_id,
            rls.dim_idx,
            label,
            value as skill_score,
            CASE WHEN stable_seasons >= 3 THEN 1 ELSE 0 END as is_stable
        FROM rolling_latent_skills rls
        LEFT JOIN latent_dim_meta ldm ON rls.model_name = ldm.model_name AND rls.dim_idx = ldm.dim_idx
        WHERE rls.model_name = 'sae_apm_v1_k12_a1'
        AND rls.season = ?
        QUALIFY ROW_NUMBER() OVER (PARTITION BY player_id, rls.dim_idx ORDER BY window_end_game_id DESC) = 1
        ORDER BY player_id, rls.dim_idx
    """, [season]).df()

    con.close()
    return df

def calculate_skill_percentiles(df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    """Calculate percentiles for each skill category."""

    percentiles = {}

    # Group by dimension to get all player scores for each skill
    for dim_label in df['label'].unique():
        dim_data = df[df['label'] == dim_label]

        if len(dim_data) < 10:  # Need minimum sample size
            continue

        # Sort by skill score (higher = better)
        sorted_data = dim_data.sort_values('skill_score', ascending=False).reset_index(drop=True)

        # Calculate percentiles
        player_percentiles = {}
        n_players = len(sorted_data)

        for i, (_, row) in enumerate(sorted_data.iterrows()):
            player_id = int(row['player_id'])
            # Percentile rank (0-100, higher = better)
            percentile = 100 * (n_players - i) / n_players
            player_percentiles[player_id] = percentile

        percentiles[dim_label] = player_percentiles

    return percentiles

def get_player_skill_profile(player_id: int, percentiles: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
    """Get skill profile for a specific player."""

    profile = {}

    for skill_category, category_info in SKILL_CATEGORIES.items():
        category_scores = []

        for dim_label in category_info["dimensions"]:
            if dim_label in percentiles and player_id in percentiles[dim_label]:
                percentile = percentiles[dim_label][player_id]
                category_scores.append(percentile)

        if category_scores:
            # Average percentile across dimensions in this category
            avg_percentile = np.mean(category_scores)

            # Classify performance level
            if avg_percentile >= 90:
                level = "Elite"
                description = "Top 10% of NHL players"
            elif avg_percentile >= 75:
                level = "Above Average"
                description = "Top 25% of NHL players"
            elif avg_percentile >= 50:
                level = "Average"
                description = "Middle 50% of NHL players"
            elif avg_percentile >= 25:
                level = "Below Average"
                description = "Bottom 25% of NHL players"
            else:
                level = "Poor"
                description = "Bottom 10% of NHL players"

            profile[skill_category] = {
                "percentile": avg_percentile,
                "level": level,
                "description": description,
                "category_name": category_info["name"],
                "category_description": category_info["description"]
            }

    return profile

def format_skill_profile(profile: Dict[str, Dict]) -> str:
    """Format skill profile into readable text."""
    lines = []

    for skill_name, data in profile.items():
        percentile = data["percentile"]
        level = data["level"]
        desc = data["description"]
        category_name = data["category_name"]

        lines.append(f"• {category_name}: {percentile:.0f}th percentile ({level}) - {desc}")

    return "\n".join(lines)

# Example usage
if __name__ == "__main__":
    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"

    print("LOADING LATENT SKILL DATA...")
    df = load_latent_data(db_path, "20242025")
    print(f"Loaded {len(df)} skill measurements for {df['player_id'].nunique()} players")

    print("\nCALCULATING PERCENTILES...")
    percentiles = calculate_skill_percentiles(df)
    print(f"Calculated percentiles for {len(percentiles)} skill dimensions")

    # Example: Connor McDavid
    player_id = 8478402
    profile = get_player_skill_profile(player_id, percentiles)

    print(f"\nSKILL PROFILE FOR PLAYER {player_id}")
    print("=" * 50)
    print(format_skill_profile(profile))

    # Show raw percentiles for each dimension
    print("\nRAW DIMENSION PERCENTILES:")
    print("-" * 30)
    for dim_label, player_percentiles in percentiles.items():
        if player_id in player_percentiles:
            pct = player_percentiles[player_id]
            print(".0f")

    print("\nThis percentile approach gives objective, interpretable skill rankings!")
    print("No expert labels needed - just statistical comparison to peers.")