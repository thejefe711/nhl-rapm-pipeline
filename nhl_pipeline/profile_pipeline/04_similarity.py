#!/usr/bin/env python3
"""
Step 04: Compute player similarity using cosine similarity.

Inputs:
  - profile_data/player_clusters.csv

Outputs:
  - profile_data/player_similarity.csv: Top 5 similar players for each player
  - profile_data/similarity_report.txt: Diagnostic report
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from profile_pipeline.config import DATA_DIR, METRIC_CATEGORIES, CURRENT_SEASON


def compute_similarity(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Compute top N similar players for each player.
    
    Uses category scores for similarity (not raw metrics).
    Compares only within same position group.
    """
    # Convert season to int for consistent comparison
    df = df.copy()
    df["season"] = df["season"].astype(int)
    
    # Use current season
    current_season_str = str(CURRENT_SEASON)
    current = df[df["season"].astype(str) == current_season_str].copy()
    
    if current.empty:
        # Fall back to most recent season
        max_season = df["season"].max()
        current = df[df["season"] == max_season].copy()
        current_season_str = str(max_season)
    
    # IMPORTANT: Deduplicate by player_id - take first row per player
    current = current.drop_duplicates(subset=["player_id"], keep="first")
    
    print(f"Computing similarity for {len(current)} unique players in season {current_season_str}")
    
    # Features for similarity - prefer Signal scores
    signal_cols = [f"{cat}_signal_score" for cat in METRIC_CATEGORIES.keys()]
    signal_cols = [c for c in signal_cols if c in current.columns]
    
    # Define weights for categories
    # High weight for Offense, Defense, and Usage context
    category_weights = {
        "OFFENSE": 2.0,
        "DEFENSE": 2.0,
        "USAGE": 2.0,
        "TRANSITION": 1.0,
        "SPECIAL_TEAMS": 1.0,
        "DISCIPLINE": 0.5,
        "FINISHING": 0.5,
    }
    
    weights = np.array([category_weights.get(cat, 1.0) for cat in METRIC_CATEGORIES.keys()])
    
    similarity_records = []
    
    for pos_group in ["F", "D"]:
        # Only compare to qualified players
        pos_data = current[(current["position_group"] == pos_group) & (current["is_qualified"] == True)].copy().reset_index(drop=True)
        
        # If the target player is NOT qualified, we still want to find similarities for them, 
        # but only compare them TO qualified players.
        targets = current[current["position_group"] == pos_group].copy().reset_index(drop=True)
        
        if len(pos_data) < 2:
            continue
        
        print(f"  {pos_group}: {len(targets)} targets, {len(pos_data)} qualified peers")
        
        # Prepare features
        X_peers = pos_data[signal_cols].fillna(0).values
        X_targets = targets[signal_cols].fillna(0).values
        
        # Apply weights to features
        X_peers_weighted = X_peers * weights
        X_targets_weighted = X_targets * weights
        
        # Compute cosine similarity matrix (targets vs peers)
        sim_matrix = cosine_similarity(X_targets_weighted, X_peers_weighted)
        
        # For each target player, find top N similar (excluding self)
        target_ids = targets["player_id"].values
        target_names = targets["full_name"].values
        peer_ids = pos_data["player_id"].values
        peer_names = pos_data["full_name"].values
        
        for i, (pid, name) in enumerate(zip(target_ids, target_names)):
            similarities = sim_matrix[i].copy()
            
            # Find index of self in peers if it exists
            self_idx = np.where(peer_ids == pid)[0]
            if len(self_idx) > 0:
                similarities[self_idx[0]] = -1
            
            # Sort by similarity descending
            sorted_indices = np.argsort(similarities)[::-1]
            
            similar_players = []
            similar_scores = []
            
            for j in sorted_indices[:top_n]:
                if similarities[j] < 0:
                    continue
                similar_players.append(peer_names[j])
                similar_scores.append(float(similarities[j]))
            
            similarity_records.append({
                "player_id": pid,
                "full_name": name,
                "position_group": pos_group,
                "similar_1": similar_players[0] if len(similar_players) > 0 else None,
                "similar_1_score": similar_scores[0] if len(similar_scores) > 0 else None,
                "similar_2": similar_players[1] if len(similar_players) > 1 else None,
                "similar_2_score": similar_scores[1] if len(similar_scores) > 1 else None,
                "similar_3": similar_players[2] if len(similar_players) > 2 else None,
                "similar_3_score": similar_scores[2] if len(similar_scores) > 2 else None,
                "similar_4": similar_players[3] if len(similar_players) > 3 else None,
                "similar_4_score": similar_scores[3] if len(similar_scores) > 3 else None,
                "similar_5": similar_players[4] if len(similar_players) > 4 else None,
                "similar_5_score": similar_scores[4] if len(similar_scores) > 4 else None,
            })
    
    sim_df = pd.DataFrame(similarity_records)
    
    # Merge similarity info back to main dataframe
    result = df.merge(
        sim_df[["player_id", "similar_1", "similar_1_score", 
                "similar_2", "similar_2_score", "similar_3", "similar_3_score",
                "similar_4", "similar_4_score", "similar_5", "similar_5_score"]],
        on="player_id",
        how="left"
    )
    
    return result


def generate_report(df: pd.DataFrame) -> str:
    """Generate similarity report."""
    lines = [
        "=" * 60,
        "SIMILARITY REPORT (SIGNAL FOCUS)",
        "=" * 60,
        "",
    ]
    
    max_season = df["season"].max()
    current = df[df["season"] == max_season]
    
    # Check some notable players
    notable_players = [
        "Connor McDavid",
        "Auston Matthews", 
        "Cale Makar",
        "Sidney Crosby",
        "Nathan MacKinnon",
    ]
    
    for player_name in notable_players:
        player = current[current["full_name"] == player_name]
        
        if player.empty:
            continue
        
        row = player.iloc[0]
        lines.append(f"\n{player_name} ({max_season}):")
        
        for i in range(1, 4):
            sim_name = row.get(f"similar_{i}")
            sim_score = row.get(f"similar_{i}_score")
            
            if pd.notna(sim_name):
                lines.append(f"  {i}. {sim_name} ({sim_score:.3f})")
    
    # Summary stats
    lines.append("\n\nSimilarity score statistics:")
    if "similar_1_score" in current.columns:
        scores = current["similar_1_score"].dropna()
        lines.append(f"  Mean top-1 similarity: {scores.mean():.3f}")
        lines.append(f"  Min top-1 similarity: {scores.min():.3f}")
        lines.append(f"  Max top-1 similarity: {scores.max():.3f}")
    
    return "\n".join(lines)


def main():
    print("Step 04: Computing player similarity...")
    
    # Load data
    input_file = DATA_DIR / "player_clusters.csv"
    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run 03_cluster.py first.")
        return None
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    
    # Compute similarity
    df = compute_similarity(df)
    print("Computed similarity")
    
    # Generate report
    report = generate_report(df)
    print(report)
    
    # Save outputs
    output_csv = DATA_DIR / "player_similarity.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    report_file = DATA_DIR / "similarity_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"Saved: {report_file}")
    
    return df


if __name__ == "__main__":
    main()
