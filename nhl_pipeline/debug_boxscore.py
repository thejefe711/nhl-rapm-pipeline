import json
from pathlib import Path

def debug_boxscore():
    path = Path('raw/20242025/2024020001/boxscore.json')
    with open(path, 'r') as f:
        data = json.load(f)
    
    pbgs = data.get('playerByGameStats', {})
    home = pbgs.get('homeTeam', {})
    forwards = home.get('forwards', [])
    if forwards:
        print("Sample Player Structure:")
        print(json.dumps(forwards[0], indent=2))

if __name__ == "__main__":
    debug_boxscore()
