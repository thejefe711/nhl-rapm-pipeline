#!/usr/bin/env python3
"""Display a formatted player profile from DLM forecasts."""

import requests
import json
import sys

def format_player_profile(player_id: int, model: str = "sae_apm_v1_k12_a1", season: str = "20242025"):
    """Fetch and format player profile."""
    url = f"http://localhost:8000/api/player/{player_id}/dlm-forecast?model={model}&season={season}&window=10&horizon=3"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching player data: {e}")
        return

    print('=' * 80)
    print(f'PLAYER PROFILE: {data.get("player_id", "Unknown")} - {data.get("season", "")}')
    print('=' * 80)
    print(f'Model: {data["model"]} | Window: {data["window"]} games | Horizon: {data["horizon"]} games ahead')
    print(f'Latest window ends: {data["window_end_game_id"]}')
    print()

    print('STABLE SKILLS (3+ seasons consistent):')
    print('-' * 50)
    stable_dims = [r for r in data['rows'] if r.get('is_stable', False)]
    for dim in sorted(stable_dims, key=lambda x: abs(x['forecast_mean']), reverse=True):
        label = dim.get('label', f'Dim {dim["dim_idx"]}')
        mean = dim['forecast_mean']
        var = dim['forecast_var']
        direction = 'HIGH' if mean > 0.1 else 'LOW' if mean < -0.1 else 'NEUTRAL'
        print(f'{direction:<8} {label:<20} | {mean:+.3f} (±{var**.5:.3f}) | {dim["stable_seasons"]} seasons')

    print()
    print('EMERGING SKILLS (unstable, developing):')
    print('-' * 50)
    emerging_dims = [r for r in data['rows'] if not r.get('is_stable', False)]
    for dim in sorted(emerging_dims, key=lambda x: abs(x['forecast_mean']), reverse=True):
        label = dim.get('label', f'Dim {dim["dim_idx"]}')
        mean = dim['forecast_mean']
        var = dim['forecast_var']
        direction = 'HIGH' if mean > 0.1 else 'LOW' if mean < -0.1 else 'NEUTRAL'
        print(f'{direction:<8} {label:<20} | {mean:+.3f} (±{var**.5:.3f}) | {dim["stable_seasons"]} seasons')

    print()
    print('SUMMARY:')
    print('-' * 50)
    stable_count = len(stable_dims)
    emerging_count = len(emerging_dims)
    high_stable = len([d for d in stable_dims if d['forecast_mean'] > 0.1])
    print(f'Stable skills: {stable_count}/12 dimensions')
    print(f'Emerging skills: {emerging_count}/12 dimensions')
    print(f'High stable skills: {high_stable}')
    print()
    print('INTERPRETATION:')
    print('- Higher forecast_mean = stronger in that skill dimension')
    print('- Lower forecast_var = more confident prediction')
    print('- Stable dimensions = consistent across seasons')
    print('- Emerging dimensions = developing or inconsistent skills')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        player_id = int(sys.argv[1])
    else:
        player_id = 8478402  # Default to Connor McDavid

    format_player_profile(player_id)