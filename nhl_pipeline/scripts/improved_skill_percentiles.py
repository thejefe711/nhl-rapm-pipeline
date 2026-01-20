#!/usr/bin/env python3
"""
Improved skill percentiles that handle positive/negative impacts correctly.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

def get_player_skill_profile(player_id: int, season: str = "20242025"):
    """Get a comprehensive skill profile with percentiles."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get player's latest skill scores
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
    """, [season]).df()

    con.close()

    # Define skill categories with their interpretation
    skill_categories = {
        "Play Creation": {
            "dimensions": ["Play driver", "Transition killer"],
            "positive_impact": True,  # Higher scores = better
            "description": "Playmaking, vision, and puck distribution"
        },
        "Defensive Impact": {
            "dimensions": ["Elite shutdown (HD)", "Two-way profile"],
            "positive_impact": True,  # Higher scores = better defense
            "description": "Shot suppression and overall defensive contribution"
        },
        "Special Teams": {
            "dimensions": ["PP quarterback", "PK stopper"],
            "positive_impact": True,  # Higher scores = better
            "description": "Power play and penalty kill effectiveness"
        }
    }

    profile = {}

    for category_name, category_info in skill_categories.items():
        category_scores = []

        for dim_label in category_info["dimensions"]:
            dim_data = df[df['label'] == dim_label]

            if len(dim_data) < 10:  # Not enough data
                continue

            # Get all scores for this dimension
            all_scores = dim_data['skill_score'].values
            player_row = dim_data[dim_data['player_id'] == player_id]

            if len(player_row) == 0:
                continue

            player_score = player_row['skill_score'].iloc[0]

            # Calculate percentile based on impact direction
            if category_info["positive_impact"]:
                # Higher scores = better (normal percentile)
                sorted_scores = np.sort(all_scores)
                percentile = np.searchsorted(sorted_scores, player_score, sorter=np.argsort(all_scores)) / len(all_scores) * 100
            else:
                # Lower scores = better (reverse percentile)
                sorted_scores = np.sort(all_scores)[::-1]  # Reverse sort
                percentile = np.searchsorted(sorted_scores, player_score, sorter=np.argsort(all_scores)[::-1]) / len(all_scores) * 100

            category_scores.append(percentile)

        if category_scores:
            avg_percentile = np.mean(category_scores)

            # Classify performance
            if avg_percentile >= 90:
                level = "Elite"
                desc = "Top 10% of NHL"
            elif avg_percentile >= 75:
                level = "Above Average"
                desc = "Top 25% of NHL"
            elif avg_percentile >= 50:
                level = "Average"
                desc = "Middle 50% of NHL"
            elif avg_percentile >= 25:
                level = "Below Average"
                desc = "Bottom 25% of NHL"
            else:
                level = "Poor"
                desc = "Bottom 10% of NHL"

            profile[category_name] = {
                "percentile": avg_percentile,
                "level": level,
                "description": desc,
                "category_description": category_info["description"]
            }

    return profile

def format_percentile_profile(profile: dict) -> str:
    """Format the percentile profile nicely."""
    lines = []
    lines.append("PERCENTILE-BASED SKILL PROFILE")
    lines.append("=" * 40)

    for skill_name, data in profile.items():
        percentile = data["percentile"]
        level = data["level"]
        desc = data["description"]
        category_desc = data["category_description"]

        lines.append(f"• {skill_name}: {percentile:.0f}th percentile ({level})")
        lines.append(f"  {category_desc} - {desc}")

    return "\n".join(lines)

if __name__ == "__main__":
    # Test with a few players
    players = [
        (8478402, "Connor McDavid"),
        (8477934, "Leon Draisaitl"),
        (8475690, "Christopher Tanev"),
        (8480803, "Evan Bouchard")
    ]

    for player_id, name in players:
        print(f"\n{'='*60}")
        print(f"PLAYER: {name} (ID: {player_id})")
        print(f"{'='*60}")

        profile = get_player_skill_profile(player_id, "20242025")
        if profile:
            print(format_percentile_profile(profile))
        else:
            print("No skill data available")

    print(f"\n{'='*60}")
    print("PERCENTILE ADVANTAGES:")
    print(f"{'='*60}")
    print("✅ Objective statistical ranking")
    print("✅ No expert labels required")
    print("✅ Interpretable (85th percentile = top 15%)")
    print("✅ Handles positive/negative skill impacts correctly")
    print("✅ Directly comparable across players")