#!/usr/bin/env python3
"""Count games per season for rolling embeddings estimation."""

from pathlib import Path

root = Path(__file__).parent.parent
canonical_dir = root / "canonical"

seasons = sorted([p.name for p in canonical_dir.glob("*") if p.is_dir()])

print("=" * 60)
print("GAMES PER SEASON (for rolling embeddings)")
print("=" * 60)

total_games = 0
total_windows = 0
window_size = 10
stride = 1

for season in seasons:
    games = list((canonical_dir / season).glob("*_event_on_ice.parquet"))
    n_games = len(games)
    if n_games >= window_size:
        n_windows = (n_games - window_size) // stride + 1
    else:
        n_windows = 0
    total_games += n_games
    total_windows += n_windows
    print(f"{season}: {n_games:3d} games -> {n_windows:3d} windows (window={window_size}, stride={stride})")

print("=" * 60)
print(f"TOTAL: {total_games} games → {total_windows} windows")
print("=" * 60)
print(f"\nTime estimates (per window ~60-90s):")
print(f"  CPU (optimized): {total_windows * 75 / 3600:.1f} hours ({total_windows * 75 / 60:.0f} minutes)")
print(f"  GPU (batched 5x): {total_windows * 20 / 3600:.1f} hours ({total_windows * 20 / 60:.0f} minutes)")
