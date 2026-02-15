#!/usr/bin/env python3
"""Display formatted LLM explanation for a player."""

import requests
import json
import sys

def show_explanation(player_id: int):
    url = f"http://localhost:8000/api/explanations/player/{player_id}?model=sae_apm_v1_k12_a1&season=20242025&window=10&horizon=3"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error: {e}")
        return

    print("=" * 80)
    print(f"LLM EXPLANATION: {data['player_name']} (ID: {data['player_id']})")
    print("=" * 80)
    print(f"Model: {data['model']} | Season: {data['season']} | Quality: {data['data_quality']}")
    print(f"Stable Skills: {data['stable_skills']} | Emerging Skills: {data['emerging_skills']}")
    print()
    print("NATURAL LANGUAGE ANALYSIS:")
    print("-" * 50)
    import textwrap
    wrapped = textwrap.fill(data['explanation'], width=80)
    print(wrapped)
    print()
    print("TECHNICAL SUMMARY:")
    print("-" * 50)
    print(f"Analysis based on {data['stable_skills']} stable + {data['emerging_skills']} emerging skill dimensions")
    print(f"Data quality: {data['data_quality']} (good = 8+ dimensions)")
    print(f"Latest data: Window ending {data['analysis_timestamp']}")

if __name__ == "__main__":
    player_id = int(sys.argv[1]) if len(sys.argv) > 1 else 8478402
    show_explanation(player_id)