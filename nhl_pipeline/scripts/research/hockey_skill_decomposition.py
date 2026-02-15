#!/usr/bin/env python3
"""
Prototype: Decompose player skills into traditional hockey attributes.

This demonstrates how we could map latent dimensions to specific hockey skills
like playmaker, sniper, net front presence, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Traditional hockey skill categories
HOCKEY_SKILLS = {
    "playmaker": {
        "description": "Play creation, passing, vision",
        "indicators": ["play_driver", "pp_quarterback", "transition_killer"],
        "weights": {"play_driver": 0.6, "pp_quarterback": 0.3, "transition_killer": 0.1}
    },
    "sniper": {
        "description": "High-danger shooting, scoring efficiency",
        "indicators": ["shooting_efficiency", "hd_xg_off", "finishing"],
        "weights": {"hd_xg_off": 0.5, "finishing": 0.3, "shooting_efficiency": 0.2}
    },
    "net_front_presence": {
        "description": "Screening, rebounding, traffic work",
        "indicators": ["net_front", "screening", "rebound"],
        "weights": {"net_front": 0.4, "screening": 0.3, "rebound": 0.3}
    },
    "defensive_specialist": {
        "description": "Shot suppression, gap control, PK ability",
        "indicators": ["hd_suppression", "pk_stopper", "gap_control"],
        "weights": {"hd_suppression": 0.4, "pk_stopper": 0.4, "gap_control": 0.2}
    },
    "transition_game": {
        "description": "Puck movement, zone exits/entries",
        "indicators": ["transition_killer", "puck_movement", "zone_transitions"],
        "weights": {"transition_killer": 0.5, "puck_movement": 0.3, "zone_transitions": 0.2}
    },
    "physicality": {
        "description": "Hitting, physical play, intimidation",
        "indicators": ["hitting", "physical_presence", "board_battles"],
        "weights": {"hitting": 0.4, "physical_presence": 0.3, "board_battles": 0.3}
    }
}

# Mapping from our SAE dimensions to hockey skills
DIMENSION_TO_SKILL_MAPPING = {
    "Play driver": ["playmaker"],
    "Elite shutdown (HD)": ["defensive_specialist"],
    "Transition killer": ["transition_game", "playmaker"],
    "PP quarterback": ["playmaker"],
    "PK stopper": ["defensive_specialist"],
    "Two-way profile": ["defensive_specialist", "transition_game"]
}

def decompose_player_skills(forecast_data: dict) -> Dict[str, float]:
    """
    Decompose player latent dimensions into traditional hockey skill percentages.

    Args:
        forecast_data: Player forecast data from API

    Returns:
        Dict mapping skill names to percentage contributions
    """
    rows = forecast_data.get("rows", [])

    # Initialize skill scores
    skill_scores = {skill: 0.0 for skill in HOCKEY_SKILLS.keys()}
    total_weight = 0.0

    # Map each dimension to skills
    for row in rows:
        dim_label = row.get("label", "")
        forecast_mean = row["forecast_mean"]
        is_stable = row.get("is_stable", False)

        # Weight stable skills more heavily
        weight = 2.0 if is_stable else 1.0

        # Map dimension to skills
        mapped_skills = DIMENSION_TO_SKILL_MAPPING.get(dim_label, [])

        for skill in mapped_skills:
            # Only count positive contributions (skills player excels at)
            if forecast_mean > 0.1:
                skill_scores[skill] += abs(forecast_mean) * weight
                total_weight += weight

    # Normalize to percentages
    if total_weight > 0:
        skill_percentages = {}
        for skill, score in skill_scores.items():
            skill_percentages[skill] = (score / total_weight) * 100
    else:
        # Default equal distribution if no positive skills
        skill_percentages = {skill: 100.0 / len(HOCKEY_SKILLS) for skill in HOCKEY_SKILLS.keys()}

    return skill_percentages

def format_skill_breakdown(percentages: Dict[str, float]) -> str:
    """Format skill percentages into readable string."""
    sorted_skills = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

    breakdown_parts = []
    for skill, pct in sorted_skills:
        if pct >= 5:  # Only show skills with meaningful contribution
            breakdown_parts.append(f"{pct:.0f}% {skill.replace('_', ' ')}")

    return ", ".join(breakdown_parts)

def get_top_skills(percentages: Dict[str, float], top_n: int = 3) -> List[Tuple[str, float]]:
    """Get top N skills for a player."""
    return sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Example usage
if __name__ == "__main__":
    # Mock data for demonstration
    mock_forecast_data = {
        "rows": [
            {"label": "Play driver", "forecast_mean": 1.585, "is_stable": False},
            {"label": "Transition killer", "forecast_mean": 0.828, "is_stable": False},
            {"label": "PP quarterback", "forecast_mean": 0.048, "is_stable": True},
            {"label": "Elite shutdown (HD)", "forecast_mean": -0.559, "is_stable": False}
        ]
    }

    skill_breakdown = decompose_player_skills(mock_forecast_data)
    formatted = format_skill_breakdown(skill_breakdown)
    top_skills = get_top_skills(skill_breakdown, 3)

    print("Skill Decomposition Example:")
    print("=" * 40)
    print(f"Player skill profile: {formatted}")
    print("\nTop 3 skills:")
    for skill, pct in top_skills:
        desc = HOCKEY_SKILLS[skill]["description"]
        print(".0f")

    print(f"\nTotal skills analyzed: {len(HOCKEY_SKILLS)}")
    print("Note: This is a prototype - actual implementation would require")
    print("expert mapping of latent dimensions to traditional hockey skills.")