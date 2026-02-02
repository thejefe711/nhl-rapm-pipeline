import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_game(boxscore_path):
    try:
        with open(boxscore_path, 'r') as f:
            data = json.load(f)
        
        game_id = data.get('id')
        season = str(game_id)[:4] + str(int(str(game_id)[:4]) + 1)
        
        stats = []
        for team_key in ['homeTeam', 'awayTeam']:
            team_data = data.get(team_key, {})
            team_id = team_data.get('id')
            
            # Players are in playerByGameStats
            # But wait, the structure might vary. Let's check a sample or use a safe path.
            # In NHL API v2, skaters are often under 'skaters' and goalies under 'goalies'
            # Let's look at the playerByGameStats section
            pbgs = data.get('playerByGameStats', {})
            team_stats = pbgs.get('homeTeam' if team_key == 'homeTeam' else 'awayTeam', {})
            
            # Skaters are split into forwards and defense
            for group in ['forwards', 'defense']:
                for player in team_stats.get(group, []):
                    stats.append({
                        'player_id': player.get('playerId'),
                        'name': f"{player.get('firstName', {}).get('default', '')} {player.get('lastName', {}).get('default', '')}",
                        'games': 1,
                        'toi_seconds': time_to_seconds(player.get('toi', '00:00')),
                        'season': season
                    })
        return stats
    except Exception:
        return []

def time_to_seconds(t_str):
    try:
        m, s = map(int, t_str.split(':'))
        return m * 60 + s
    except:
        return 0

def aggregate_stats():
    raw_dir = Path('raw')
    boxscore_files = list(raw_dir.glob('*/20*/boxscore.json'))
    
    # Map IDs to names for reporting
    target_players = {
        8478427: 'Sebastian Aho (F)',
        8480222: 'Sebastian Aho (D)',
        8482093: 'Seth Jarvis',
        8471675: 'Sidney Crosby',
        8478402: 'Connor McDavid',
        8473533: 'Jordan Staal'
    }
    target_ids = set(target_players.keys())
    
    print(f"Scanning {len(boxscore_files)} boxscores for target IDs...")
    
    player_stats = {pid: {'name': name, 'games': 0, 'toi_seconds': 0, 'seasons': set()} for pid, name in target_players.items()}
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_game, boxscore_files), total=len(boxscore_files)))
    
    for game_stats in results:
        for p in game_stats:
            pid = p['player_id']
            if pid in target_ids:
                player_stats[pid]['games'] += 1
                player_stats[pid]['toi_seconds'] += p['toi_seconds']
                player_stats[pid]['seasons'].add(p['season'])

    print("\n=== Fast Career Stats Audit ===")
    for pid, data in player_stats.items():
        print(f"\n{data['name']} (ID: {pid}):")
        print(f"  Career Games: {data['games']}")
        print(f"  Total TOI:   {data['toi_seconds']/60:.1f} min")
        print(f"  Seasons:     {sorted(list(data['seasons']))}")

if __name__ == "__main__":
    aggregate_stats()
