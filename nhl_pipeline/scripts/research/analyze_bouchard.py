#!/usr/bin/env python3
"""Detailed analysis of Evan Bouchard."""

import requests
import json

def analyze_bouchard():
    url = "http://localhost:8000/api/player/8480803/dlm-forecast?model=sae_apm_v1_k12_a1&season=20242025&window=10&horizon=3"

    response = requests.get(url)
    data = response.json()

    print('EVAN BOUCHARD: DETAILED SKILL ANALYSIS')
    print('=' * 60)
    print()

    print('ELITE OFFENSIVE STRENGTHS (Stable):')
    print('-' * 40)
    for row in data['rows']:
        if row['forecast_mean'] > 0.3 and row.get('is_stable', False):
            label = row.get('label', f'Dim {row["dim_idx"]}')
            mean = row['forecast_mean']
            seasons = row.get('stable_seasons', 0)
            print(f'{label:<20} | {mean:+.3f} | {seasons} seasons consistent')

    print()
    print('DEVELOPING OFFENSIVE SKILLS (Emerging):')
    print('-' * 40)
    for row in data['rows']:
        if row['forecast_mean'] > 0.3 and not row.get('is_stable', False):
            label = row.get('label', f'Dim {row["dim_idx"]}')
            mean = row['forecast_mean']
            seasons = row.get('stable_seasons', 0)
            print(f'{label:<20} | {mean:+.3f} | Emerging ({seasons} seasons)')

    print()
    print('DEFENSIVE WEAKNESSES:')
    print('-' * 40)
    for row in data['rows']:
        if row['forecast_mean'] < -0.2:
            label = row.get('label', f'Dim {row["dim_idx"]}')
            mean = row['forecast_mean']
            stable = 'STABLE' if row.get('is_stable', False) else 'EMERGING'
            seasons = row.get('stable_seasons', 0)
            print(f'{label:<20} | {mean:+.3f} | {stable} ({seasons} seasons)')

    print()
    print('NEUTRAL SKILLS:')
    print('-' * 40)
    for row in data['rows']:
        mean = row['forecast_mean']
        if -0.2 <= mean <= 0.3:
            label = row.get('label', f'Dim {row["dim_idx"]}')
            stable = 'STABLE' if row.get('is_stable', False) else 'EMERGING'
            seasons = row.get('stable_seasons', 0)
            print(f'{label:<20} | {mean:+.3f} | {stable} ({seasons} seasons)')

    print()
    print('CORRECTED ASSESSMENT:')
    print('-' * 40)
    print('Bouchard is NOT declining - he is a high-end offensive defenseman!')
    print()
    print('ELITE STRENGTHS:')
    print('  • Elite transition offense (+0.816 stable)')
    print('  • Strong secondary transition (+0.439 stable)')
    print('  • Developing playmaking (+0.910 emerging)')
    print()
    print('DEVELOPMENTAL CONCERNS:')
    print('  • Significant defensive weaknesses emerging')
    print('  • Struggles with two-way play (-0.291 stable)')
    print('  • High-danger shutdown challenges (-0.576, -1.972 emerging)')
    print()
    print('NHL ROLE: Premium offensive defenseman, not a two-way defender')

if __name__ == "__main__":
    analyze_bouchard()