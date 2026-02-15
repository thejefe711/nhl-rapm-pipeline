#!/usr/bin/env python3
"""
Player Skill Tracker - Track skills over time with projections.

Inspired by EvanMiya's college basketball tool, this provides:
- Skill progression over career
- Current projections with confidence intervals
- RAPM integration
- Career trajectory analysis
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

def get_player_skill_history(player_id: int, season: str = "20242025") -> pd.DataFrame:
    """Get complete skill history for a player."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get all historical latent skills
    df = con.execute("""
        SELECT
            season,
            window_end_game_id,
            window_end_time_utc,
            dim_idx,
            label,
            value as skill_score
        FROM rolling_latent_skills rls
        LEFT JOIN latent_dim_meta ldm ON rls.model_name = ldm.model_name AND rls.dim_idx = ldm.dim_idx
        WHERE rls.model_name = 'sae_apm_v1_k12_a1'
        AND player_id = ?
        AND window_size = 10
        ORDER BY window_end_time_utc
    """, [player_id]).df()

    con.close()
    return df

def get_player_rapm_history(player_id: int) -> pd.DataFrame:
    """Get RAPM history for player."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    df = con.execute("""
        SELECT
            season,
            metric_name,
            value as rapm_score
        FROM apm_results
        WHERE player_id = ?
        ORDER BY season
    """, [player_id]).df()

    con.close()
    return df

def calculate_skill_percentiles_over_time(player_id: int) -> Dict[str, List]:
    """Calculate how player's skill percentiles change over time."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # For each dimension, get player's scores over time and league percentiles
    skill_progression = {}

    dimensions = con.execute("""
        SELECT DISTINCT dim_idx, label
        FROM latent_dim_meta
        WHERE model_name = 'sae_apm_v1_k12_a1'
    """).fetchall()

    for dim_idx, label in dimensions:
        # Get player's scores over time
        player_scores = con.execute("""
            SELECT window_end_time_utc, value
            FROM rolling_latent_skills
            WHERE model_name = 'sae_apm_v1_k12_a1'
            AND player_id = ?
            AND dim_idx = ?
            AND window_size = 10
            ORDER BY window_end_time_utc
        """, [player_id, dim_idx]).df()

        if player_scores.empty:
            continue

        percentiles_over_time = []

        for _, row in player_scores.iterrows():
            score_time = row['window_end_time_utc']
            player_score = row['value']

            # Get league scores at this time period (simplified - just get all for now)
            league_scores = con.execute("""
                SELECT value
                FROM rolling_latent_skills rls
                WHERE model_name = 'sae_apm_v1_k12_a1'
                AND dim_idx = ?
                AND window_size = 10
            """, [dim_idx]).df()

            if not league_scores.empty:
                # Calculate percentile (higher score = higher percentile)
                percentile = (player_score > league_scores['value']).mean() * 100
                percentiles_over_time.append({
                    'date': score_time,
                    'percentile': percentile,
                    'raw_score': player_score
                })

        skill_progression[label] = percentiles_over_time

    con.close()
    return skill_progression

def get_player_skill_grades(player_id: int) -> Dict[str, Dict]:
    """Calculate current skill grades and projections."""

    skill_progression = calculate_skill_percentiles_over_time(player_id)

    grades = {}

    for skill_name, history in skill_progression.items():
        if not history:
            continue

        # Current percentile (most recent)
        current = history[-1]['percentile']

        # Grade based on percentile
        if current >= 90:
            grade = "A+"
            description = "Elite"
        elif current >= 80:
            grade = "A"
            description = "Excellent"
        elif current >= 70:
            grade = "A-"
            description = "Very Good"
        elif current >= 60:
            grade = "B+"
            description = "Good"
        elif current >= 50:
            grade = "B"
            description = "Above Average"
        elif current >= 40:
            grade = "B-"
            description = "Average"
        elif current >= 30:
            grade = "C+"
            description = "Below Average"
        elif current >= 20:
            grade = "C"
            description = "Poor"
        else:
            grade = "C-"
            description = "Very Poor"

        grades[skill_name] = {
            "grade": grade,
            "percentile": current,
            "description": description,
            "history": history
        }

    return grades

def format_player_skill_report(player_id: int) -> str:
    """Generate complete skill report like EvanMiya."""

    grades = get_player_skill_grades(player_id)
    skill_progression = calculate_skill_percentiles_over_time(player_id)

    # Get player name
    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))
    name_result = con.execute("SELECT full_name FROM players WHERE player_id = ?", [player_id]).fetchone()
    player_name = name_result[0] if name_result else f"Player {player_id}"
    con.close()

    report = []
    report.append(f"PLAYER SKILL PROFILE: {player_name}")
    report.append("=" * 60)
    report.append("")

    # Current Grades
    report.append("CURRENT SKILL GRADES")
    report.append("-" * 30)

    # Sort by percentile
    sorted_grades = sorted(grades.items(), key=lambda x: x[1]['percentile'], reverse=True)

    for skill_name, data in sorted_grades:
        grade = data['grade']
        percentile = data['percentile']
        desc = data['description']
        report.append("8")

    report.append("")
    report.append("SKILL PROGRESSION OVER TIME")
    report.append("-" * 40)

    # Show progression for top 3 skills
    for skill_name, _ in sorted_grades[:3]:
        if skill_name in skill_progression:
            history = skill_progression[skill_name]
            if len(history) >= 3:
                report.append(f"\n{skill_name} Progression:")
                recent = history[-3:]  # Last 3 points
                for point in recent:
                    date = str(point['date'])[:10]  # YYYY-MM-DD
                    pct = point['percentile']
                    report.append(".1f")

    report.append("")
    report.append("PROJECTIONS & CONFIDENCE INTERVALS")
    report.append("-" * 45)

    # Get DLM forecasts
    try:
        import requests
        response = requests.get(f"http://localhost:8000/api/player/{player_id}/dlm-forecast?model=sae_apm_v1_k12_a1&season=20242025&window=10&horizon=3")
        if response.status_code == 200:
            forecast_data = response.json()
            rows = forecast_data.get('rows', [])[:3]  # Top 3 dimensions

            for row in rows:
                label = row.get('label', f'Dim {row["dim_idx"]}')
                mean = row['forecast_mean']
                std = row['forecast_var'] ** 0.5
                ci_lower = mean - 1.96 * std
                ci_upper = mean + 1.96 * std

                report.append(f"{label}:")
                report.append(".3f")
    except Exception as e:
        report.append(f"Could not load projections: {e}")

    report.append("")
    report.append("INTERPRETATION")
    report.append("-" * 20)
    report.append("• Grades: A+ (Elite) to C- (Very Poor) based on percentile vs league")
    report.append("• Progression: Shows how skills have developed over career")
    report.append("• Projections: 3-game ahead forecasts with 95% confidence intervals")
    report.append("• RAPM Integration: Skills adjusted for teammates/opponents")

    return "\n".join(report)

if __name__ == "__main__":
    # Example: Connor McDavid
    player_id = 8478402
    report = format_player_skill_report(player_id)
    print(report)