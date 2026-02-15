#!/usr/bin/env python3
"""
RAPM DLM Forecasts - Add DLM forecasting to RAPM metrics.

Similar to EvanMiya's approach, this creates predictive models for RAPM metrics
to forecast future performance with confidence intervals.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def fit_rapm_dlm(player_id: int, metric_name: str) -> Dict:
    """Fit a simple DLM-like model to RAPM time series."""

    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get RAPM data across seasons
    rapm_data = con.execute("""
        SELECT
            season,
            value as rapm_value
        FROM apm_results
        WHERE player_id = ? AND metric_name = ?
        ORDER BY season
    """, [player_id, metric_name]).fetchall()

    con.close()

    if len(rapm_data) < 3:
        return {"error": "Insufficient data for forecasting"}

    # Convert to DataFrame
    df = pd.DataFrame(rapm_data, columns=['season', 'rapm_value'])

    # Create time index (season number)
    df['season_num'] = range(len(df))

    # Simple linear trend model (basic DLM approximation)
    X = df[['season_num']]
    y = df['rapm_value']

    model = LinearRegression()
    model.fit(X, y)

    # Calculate residuals for uncertainty estimation
    predictions = model.predict(X)
    residuals = y - predictions
    residual_std = np.std(residuals)

    # Generate forecasts for next 3 seasons
    future_season_nums = [len(df) + i for i in range(1, 4)]
    future_X = pd.DataFrame({'season_num': future_season_nums})
    future_predictions = model.predict(future_X)

    # Calculate confidence intervals
    forecasts = []
    for i, (season_num, pred) in enumerate(zip(future_season_nums, future_predictions)):
        # Confidence interval (simplified - increases with distance)
        horizon = i + 1
        ci_width = residual_std * np.sqrt(horizon) * 1.96  # Rough approximation

        forecasts.append({
            'horizon_seasons': horizon,
            'forecast_mean': round(pred, 4),
            'forecast_std': round(residual_std * np.sqrt(horizon), 4),
            'ci_lower': round(pred - ci_width, 4),
            'ci_upper': round(pred + ci_width, 4)
        })

    return {
        'model_type': 'linear_trend',
        'historical_data': df[['season', 'rapm_value']].to_dict('records'),
        'trend_slope': round(model.coef_[0], 4),
        'trend_intercept': round(model.intercept_, 4),
        'residual_std': round(residual_std, 4),
        'forecasts': forecasts
    }

def create_rapm_forecast_chart_data(player_id: int, metric_name: str) -> Dict:
    """Create RAPM timeline chart with DLM forecasts."""

    # Fit DLM model
    dlm_model = fit_rapm_dlm(player_id, metric_name)

    if "error" in dlm_model:
        return dlm_model

    # Create historical data points
    historical_data = []
    for point in dlm_model['historical_data']:
        # Convert season to approximate date
        season_year = int(point['season'][:4])
        date = f"{season_year}-10-01"

        historical_data.append({
            "date": date,
            "value": point['rapm_value'],
            "type": "historical"
        })

    # Create forecast data points
    forecast_data = []
    confidence_intervals = []

    for forecast in dlm_model['forecasts']:
        # Project future dates
        last_season_year = int(dlm_model['historical_data'][-1]['season'][:4])
        future_year = last_season_year + forecast['horizon_seasons']
        date = f"{future_year}-10-01"

        forecast_data.append({
            "date": date,
            "value": forecast['forecast_mean'],
            "type": "forecast"
        })

        confidence_intervals.append({
            "date": date,
            "lower": forecast['ci_lower'],
            "upper": forecast['ci_upper']
        })

    return {
        "chart_type": "line_with_confidence",
        "title": f"{metric_name.replace('_', ' ').title()} Career Timeline",
        "subtitle": f"Trend: {dlm_model['trend_slope']:+.4f} per season",
        "x_axis_label": "Season",
        "y_axis_label": "RAPM Value",
        "data": historical_data,
        "forecast_data": forecast_data,
        "confidence_intervals": confidence_intervals,
        "model_stats": {
            "r_squared": None,  # Would need more sophisticated model
            "residual_std": dlm_model['residual_std'],
            "seasons_analyzed": len(dlm_model['historical_data'])
        }
    }

def demo_rapm_forecasts(player_id: int = 8478402):
    """Demo RAPM forecasting for key metrics."""

    print(f"RAPM DLM FORECASTS - Player {player_id}")
    print("=" * 50)

    # Get player name
    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))
    player_name_result = con.execute("SELECT full_name FROM players WHERE player_id = ?", [player_id]).fetchone()
    player_name = player_name_result[0] if player_name_result else f"Player {player_id}"
    con.close()

    print(f"Player: {player_name}")
    print()

    # Key RAPM metrics to forecast
    key_metrics = [
        "xg_off_rapm_5v5",
        "corsi_off_rapm_5v5",
        "hd_xg_off_rapm_5v5",
        "finishing_residual_rapm_5v5"
    ]

    forecast_results = {}

    for metric in key_metrics:
        print(f"Forecasting {metric}...")
        chart_data = create_rapm_forecast_chart_data(player_id, metric)

        if "error" in chart_data:
            print(f"  {chart_data['error']}")
            continue

        forecast_results[metric] = chart_data

        # Show key forecast info
        last_forecast_value = chart_data['forecast_data'][-1]['value'] if chart_data.get('forecast_data') else None
        last_ci = chart_data['confidence_intervals'][-1] if chart_data.get('confidence_intervals') else None

        if last_forecast_value is not None and last_ci:
            trend = chart_data.get('subtitle', 'No trend info')
            print(f"  Historical seasons: {len(chart_data['data'])}")
            print(f"  {trend}")
            print(f"  3-season forecast: {last_forecast_value:+.3f} ({last_ci['lower']:+.3f} to {last_ci['upper']:+.3f})")
        print()

    return forecast_results

if __name__ == "__main__":
    # Demo RAPM forecasting
    forecasts = demo_rapm_forecasts(8478402)

    # Example: Show detailed forecast for one metric
    if 'xg_off_rapm_5v5' in forecasts:
        xg_forecast = forecasts['xg_off_rapm_5v5']
        print("XG OFF RAPM FORECAST DETAILS:")
        print("=" * 40)
        print(f"Model: {xg_forecast.get('model_stats', {}).get('seasons_analyzed', 0)} seasons analyzed")
        print(f"Trend: {xg_forecast.get('subtitle', 'Unknown')}")

        print("\n3-Year Forecast:")
        for i, forecast_data in enumerate(xg_forecast.get('forecast_data', [])):
            horizon = i + 1
            ci_data = xg_forecast.get('confidence_intervals', [])[i] if i < len(xg_forecast.get('confidence_intervals', [])) else {}
            print(f"  Year +{horizon}: {forecast_data['value']:+.3f} ({ci_data.get('lower', 0):+.3f} to {ci_data.get('upper', 0):+.3f})")