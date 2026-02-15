#!/usr/bin/env python3
"""Test different shift chart API query formats."""

import requests
import json
from pathlib import Path

# Load failed games from progress
progress = json.load(open(Path(__file__).parent / "data" / "fetch_progress.json"))
failed_games = progress.get("20252026", {}).get("failed_games", [])[:5]

print("Testing different API query formats:")
print("=" * 70)

for game_id in failed_games:
    print(f"\nGame {game_id}:")
    
    # Simple query (what we use now)
    simple_url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
    r1 = requests.get(simple_url, timeout=30)
    data1 = r1.json()
    simple_count = len(data1.get("data", []))
    
    # Complex query (from nhl-api-py)
    expr = f"gameId={game_id} and ((duration != '00:00' and typeCode = 517) or typeCode != 517 )"
    complex_url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp={expr}&exclude=eventDetails"
    r2 = requests.get(complex_url, timeout=30)
    data2 = r2.json()
    complex_count = len(data2.get("data", []))
    
    print(f"  Simple query: {simple_count} shifts")
    print(f"  Complex query: {complex_count} shifts")
    
    if simple_count != complex_count:
        print(f"  *** DIFFERENCE FOUND! ***")

# Also test a known good game
print("\n\nTesting known good game 2025020002:")
game_id = 2025020002

simple_url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
r1 = requests.get(simple_url, timeout=30)
simple_count = len(r1.json().get("data", []))

expr = f"gameId={game_id} and ((duration != '00:00' and typeCode = 517) or typeCode != 517 )"
complex_url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp={expr}&exclude=eventDetails"
r2 = requests.get(complex_url, timeout=30)
complex_count = len(r2.json().get("data", []))

print(f"  Simple query: {simple_count} shifts")
print(f"  Complex query: {complex_count} shifts")
