#!/usr/bin/env python3
"""
Career Timeline Charts - DLM-based career progression and forecasts.

Creates career timeline visualizations for:
- RAPM metrics over time with DLM forecasts
- Latent skills progression with confidence intervals
- Similar to EvanMiya's college basketball tool
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

def get_player_rapm_timeline(player_id: int) -> pd.DataFrame:
    """Get RAPM metrics timeline for career progression."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get RAPM data across all seasons
    rapm_timeline = con.execute("""
        SELECT
            season,
            metric_name,
            value as rapm_value
        FROM apm_results
        WHERE player_id = ?
        ORDER BY season, metric_name
    """, [player_id]).fetchall()

    con.close()

    # Convert to DataFrame
    df = pd.DataFrame(rapm_timeline, columns=['season', 'metric_name', 'rapm_value'])

    # Pivot to have seasons as rows, metrics as columns
    timeline_df = df.pivot(index='season', columns='metric_name', values='rapm_value').reset_index()

    return timeline_df

def get_player_latent_timeline(player_id: int) -> pd.DataFrame:
    """Get latent skills timeline for career progression."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get latent skills across rolling windows
    latent_timeline = con.execute("""
        SELECT
            window_end_time_utc as date,
            ldm.label as skill_name,
            rls.value as skill_value,
            ldm.stable_seasons
        FROM rolling_latent_skills rls
        JOIN latent_dim_meta ldm ON rls.model_name = ldm.model_name AND rls.dim_idx = ldm.dim_idx
        WHERE rls.player_id = ?
        AND rls.model_name = 'sae_apm_v1_k12_a1'
        AND rls.window_size = 10
        ORDER BY date, skill_name
    """, [player_id]).fetchall()

    con.close()

    # Convert to DataFrame
    df = pd.DataFrame(latent_timeline, columns=['date', 'skill_name', 'skill_value', 'stable_seasons'])

    # Convert date string to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Derive stability
    df['is_stable'] = df['stable_seasons'] >= 2

    return df

def get_dlm_forecasts(player_id: int) -> pd.DataFrame:
    """Get DLM forecasts for future projections."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get forecasts for multiple horizons
    forecasts = con.execute("""
        SELECT
            ldm.label as skill_name,
            horizon_games,
            forecast_mean,
            forecast_var,
            filtered_mean,
            filtered_var
        FROM dlm_forecasts df
        JOIN latent_dim_meta ldm ON df.model_name = ldm.model_name AND df.dim_idx = ldm.dim_idx
        WHERE df.player_id = ?
        AND df.model_name = 'sae_apm_v1_k12_a1'
        ORDER BY skill_name, horizon_games
    """, [player_id]).fetchall()

    con.close()

    df = pd.DataFrame(forecasts, columns=['skill_name', 'horizon_games', 'forecast_mean', 'forecast_var', 'filtered_mean', 'filtered_var'])

    return df

def create_rapm_timeline_chart_data(player_id: int, metric_name: str = "xg_off_rapm_5v5") -> Dict:
    """Create chart data for RAPM career timeline with DLM forecasts."""

    # Get historical RAPM data
    rapm_timeline = get_player_rapm_timeline(player_id)

    if rapm_timeline.empty or metric_name not in rapm_timeline.columns:
        return {"error": f"No data available for {metric_name}"}

    # Filter to the specific metric
    metric_data = rapm_timeline[['season', metric_name]].dropna()

    # Create timeline data points
    historical_data = []
    for _, row in metric_data.iterrows():
        # Convert season to approximate date (use October 1st of season)
        season_year = int(row['season'][:4])
        date = f"{season_year}-10-01"

        historical_data.append({
            "date": date,
            "value": round(row[metric_name], 4),
            "type": "historical"
        })

    # Add RAPM DLM forecasts
    from .rapm_dlm_forecasts import create_rapm_forecast_chart_data
    forecast_chart = create_rapm_forecast_chart_data(player_id, metric_name)

    if "error" not in forecast_chart:
        forecast_data = forecast_chart.get("forecast_data", [])
        confidence_intervals = forecast_chart.get("confidence_intervals", [])
        subtitle = forecast_chart.get("subtitle", "")
    else:
        forecast_data = []
        confidence_intervals = []
        subtitle = ""

    return {
        "chart_type": "line_with_confidence",
        "title": f"{metric_name.replace('_', ' ').title()} Career Timeline",
        "subtitle": subtitle,
        "x_axis_label": "Season",
        "y_axis_label": "RAPM Value",
        "data": historical_data,
        "forecast_data": forecast_data,
        "confidence_intervals": confidence_intervals,
        "has_forecasts": len(forecast_data) > 0
    }

def create_latent_skill_timeline_chart_data(player_id: int, skill_name: str) -> Dict:
    """Create chart data for latent skill career timeline with DLM forecasts."""

    # Get historical latent skill data
    latent_timeline = get_player_latent_timeline(player_id)

    if latent_timeline.empty:
        return {"error": "No latent skill data available"}

    # Filter to the specific skill
    skill_data = latent_timeline[latent_timeline['skill_name'] == skill_name].copy()

    if skill_data.empty:
        return {"error": f"No data available for skill: {skill_name}"}

    # Sort by date
    skill_data = skill_data.sort_values('date')

    # Create historical data points
    historical_data = []
    for _, row in skill_data.iterrows():
        historical_data.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "value": round(row['skill_value'], 4),
            "type": "historical",
            "is_stable": row['is_stable']
        })

    # Get DLM forecasts for this skill
    forecasts = get_dlm_forecasts(player_id)
    skill_forecasts = forecasts[forecasts['skill_name'] == skill_name]

    # Create forecast data points
    forecast_data = []
    confidence_intervals = []

    if not skill_forecasts.empty:
        # Get the last historical date
        last_date = skill_data['date'].max()

        for _, forecast in skill_forecasts.iterrows():
            # Project forward from last date
            horizon_days = forecast['horizon_games'] * 2  # Rough estimate: 2 days per game
            forecast_date = last_date + pd.Timedelta(days=horizon_days)

            forecast_mean = forecast['forecast_mean']
            forecast_std = np.sqrt(forecast['forecast_var'])

            forecast_data.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "value": round(forecast_mean, 4),
                "type": "forecast"
            })

            # 95% confidence interval
            ci_lower = forecast_mean - 1.96 * forecast_std
            ci_upper = forecast_mean + 1.96 * forecast_std

            confidence_intervals.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "lower": round(ci_lower, 4),
                "upper": round(ci_upper, 4)
            })

    return {
        "chart_type": "line_with_confidence",
        "title": f"{skill_name} Career Timeline",
        "x_axis_label": "Date",
        "y_axis_label": "Skill Score",
        "data": historical_data,
        "forecast_data": forecast_data,
        "confidence_intervals": confidence_intervals,
        "has_forecasts": len(forecast_data) > 0
    }

def generate_player_career_timeline(player_id: int, output_format: str = "json") -> Dict:
    """Generate complete career timeline data for a player."""

    # Get player name
    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))
    player_name_result = con.execute("SELECT full_name FROM players WHERE player_id = ?", [player_id]).fetchone()
    player_name = player_name_result[0] if player_name_result else f"Player {player_id}"
    con.close()

    # RAPM timelines for key metrics
    rapm_charts = {}
    key_rapm_metrics = [
        "xg_off_rapm_5v5",
        "corsi_off_rapm_5v5",
        "hd_xg_off_rapm_5v5",
        "finishing_residual_rapm_5v5"
    ]

    for metric in key_rapm_metrics:
        chart_data = create_rapm_timeline_chart_data(player_id, metric)
        if "error" not in chart_data:
            rapm_charts[metric] = chart_data

    # Latent skill timelines
    latent_charts = {}
    latent_timeline = get_player_latent_timeline(player_id)

    if not latent_timeline.empty:
        # Get top skills by recent performance
        recent_skills = latent_timeline.sort_values('date').groupby('skill_name').last().reset_index()
        top_skills = recent_skills.nlargest(6, 'skill_value')['skill_name'].tolist()

        for skill_name in top_skills:
            chart_data = create_latent_skill_timeline_chart_data(player_id, skill_name)
            if "error" not in chart_data:
                latent_charts[skill_name] = chart_data

    result = {
        "player_id": player_id,
        "player_name": player_name,
        "generated_at": datetime.now().isoformat(),
        "rapm_timelines": rapm_charts,
        "latent_skill_timelines": latent_charts,
        "summary": {
            "total_rapm_charts": len(rapm_charts),
            "total_latent_charts": len(latent_charts),
            "has_forecasts": any(chart.get("has_forecasts", False) for chart in latent_charts.values())
        }
    }

    if output_format == "json":
        return result
    else:
        return json.dumps(result, indent=2)

def create_chart_html(chart_data: Dict) -> str:
    """Create HTML for a timeline chart (simplified example)."""

    title = chart_data.get("title", "Timeline Chart")

    html = f"""
    <div class="timeline-chart">
        <h3>{title}</h3>
        <div class="chart-container" style="width: 100%; height: 300px;">
            <!-- Chart would be rendered here with a JS library like Chart.js or D3.js -->
            <p>Chart data: {len(chart_data.get('data', []))} historical points, {len(chart_data.get('forecast_data', []))} forecast points</p>
        </div>
    </div>
    """

    return html

def demo_career_timeline(player_id: int = 8478402):
    """Demo career timeline generation."""

    print(f"Generating career timeline for Player {player_id}...")
    print("=" * 60)

    timeline_data = generate_player_career_timeline(player_id)

    print(f"Player: {timeline_data['player_name']}")
    print(f"Generated: {timeline_data['generated_at']}")
    print()

    print("RAPM TIMELINES:")
    for metric, chart in timeline_data['rapm_timelines'].items():
        historical_points = len(chart['data'])
        forecast_points = len(chart.get('forecast_data', []))
        print(f"  {metric}: {historical_points} seasons, {forecast_points} forecast points")

    print()
    print("LATENT SKILL TIMELINES:")
    for skill, chart in timeline_data['latent_skill_timelines'].items():
        historical_points = len(chart['data'])
        forecast_points = len(chart.get('forecast_data', []))
        has_forecasts = chart.get('has_forecasts', False)
        print(f"  {skill}: {historical_points} data points, forecasts: {has_forecasts}")

    print()
    print(f"Summary: {timeline_data['summary']['total_rapm_charts']} RAPM charts, {timeline_data['summary']['total_latent_charts']} latent skill charts")
    print(f"Has DLM forecasts: {timeline_data['summary']['has_forecasts']}")

    return timeline_data

if __name__ == "__main__":
    # Demo with Connor McDavid
    timeline_data = demo_career_timeline(8478402)

    # Example: Get data for a specific chart
    if timeline_data['latent_skill_timelines']:
        first_skill = list(timeline_data['latent_skill_timelines'].keys())[0]
        skill_chart = timeline_data['latent_skill_timelines'][first_skill]

        print(f"\nExample {first_skill} chart data:")
        print(f"  Historical points: {len(skill_chart['data'])}")
        print(f"  Forecast points: {len(skill_chart['forecast_data'])}")
        if skill_chart['confidence_intervals']:
            print(f"  Confidence intervals: {len(skill_chart['confidence_intervals'])}")