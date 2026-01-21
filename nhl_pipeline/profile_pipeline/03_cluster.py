#!/usr/bin/env python3
"""
Step 03: K-means clustering to create player archetypes.

Inputs:
  - profile_data/player_categories.csv

Outputs:
  - profile_data/player_clusters.csv: With cluster assignments
  - profile_data/cluster_report.txt: Cluster descriptions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from profile_pipeline.config import (
    DATA_DIR, METRIC_CATEGORIES, CLUSTERS_FORWARD, CLUSTERS_DEFENSE, CURRENT_SEASON
)


# Cluster name templates based on dominant characteristics
ARCHETYPE_NAMES = {
    "F": {
        # Will be assigned based on cluster centroids
        "high_offense_high_finish": "Sniper",
        "high_offense_low_finish": "Playmaker",
        "balanced": "Two-Way Forward",
        "high_defense": "Defensive Forward",
        "high_transition": "Transition Specialist",
        "low_all": "Depth Forward",
    },
    "D": {
        "high_offense": "Offensive Defenseman",
        "high_defense": "Shutdown Defenseman",
        "balanced": "Two-Way Defenseman",
        "high_transition": "Puck-Moving Defenseman",
        "low_all": "Depth Defenseman",
    }
}


def cluster_players(df: pd.DataFrame, position_group: str, n_clusters: int) -> pd.DataFrame:
    """
    Cluster players within a position group.
    
    Uses category scores as features.
    """
    # Filter to position group and qualified players
    mask = (df["position_group"] == position_group) & (df["is_qualified"] == True)
    subset = df[mask].copy()
    
    if len(subset) < n_clusters * 3:
        print(f"WARNING: Not enough qualified {position_group} players for {n_clusters} clusters. Falling back to all players.")
        mask = df["position_group"] == position_group
        subset = df[mask].copy()
        
    if len(subset) < n_clusters:
        subset["cluster"] = 0
        subset["archetype"] = f"Unknown {position_group}"
        return subset
    
    # Features for clustering - Use Signal for more stable archetypes
    feature_cols = [f"{cat}_signal_score" for cat in METRIC_CATEGORIES.keys()]
    feature_cols = [c for c in feature_cols if c in subset.columns]
    
    # Use only current season for clustering
    current_season_str = str(CURRENT_SEASON)
    current = subset[subset["season"] == current_season_str].copy()
    
    if len(current) < n_clusters * 5:
        # Fall back to most recent season
        max_season = subset["season"].max()
        current = subset[subset["season"] == max_season].copy()
    
    # Prepare features
    X = current[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    current["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Analyze cluster centroids to assign names
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_cols
    )
    
    # Assign archetype names based on dominant characteristics
    archetype_map = {}
    for cluster_id in range(n_clusters):
        centroid = centroids.iloc[cluster_id]
        # Map back to generic names for _assign_archetype_name
        generic_centroid = {cat: centroid.get(f"{cat}_signal_score", 0) for cat in METRIC_CATEGORIES.keys()}
        archetype_map[cluster_id] = _assign_archetype_name(
            pd.Series(generic_centroid), position_group, cluster_id
        )
    
    current["archetype"] = current["cluster"].map(archetype_map)
    
    # Propagate cluster assignments to all seasons for same players
    player_clusters = current[["player_id", "cluster", "archetype"]].drop_duplicates()
    
    result = subset.merge(
        player_clusters,
        on="player_id",
        how="left",
        suffixes=("", "_new")
    )
    
    # Fill any remaining NaN clusters
    if "cluster_new" in result.columns:
        result["cluster"] = result["cluster_new"].fillna(result.get("cluster", 0))
        result["archetype"] = result["archetype_new"].fillna(result.get("archetype", "Unknown"))
        result = result.drop(columns=["cluster_new", "archetype_new"])
    
    return result


def _assign_archetype_name(centroid: pd.Series, position_group: str, cluster_id: int) -> str:
    """Assign an archetype name based on cluster centroid characteristics."""
    offense = centroid.get("OFFENSE", 0)
    defense = centroid.get("DEFENSE", 0)
    finishing = centroid.get("FINISHING", 0)
    transition = centroid.get("TRANSITION", 0)
    special = centroid.get("SPECIAL_TEAMS", 0)
    
    if position_group == "F":
        if offense > 0.5 and finishing > 0.3:
            return "Sniper"
        elif offense > 0.5 and finishing < 0:
            return "Playmaker"
        elif defense > 0.5:
            return "Defensive Forward"
        elif abs(offense) < 0.3 and abs(defense) < 0.3:
            return "Two-Way Forward"
        elif transition > 0.3:
            return "Transition Forward"
        elif special > 0.5:
            return "Special Teams Forward"
        else:
            return f"Forward Type {cluster_id + 1}"
    else:  # Defenseman
        if offense > 0.5:
            return "Offensive Defenseman"
        elif defense > 0.5:
            return "Shutdown Defenseman"
        elif transition > 0.3:
            return "Puck-Moving Defenseman"
        elif abs(offense) < 0.3 and abs(defense) < 0.3:
            return "Two-Way Defenseman"
        else:
            return f"Defenseman Type {cluster_id + 1}"


def generate_report(df: pd.DataFrame) -> str:
    """Generate cluster report."""
    lines = [
        "=" * 60,
        "CLUSTERING REPORT (SIGNAL FOCUS)",
        "=" * 60,
        "",
    ]
    
    # Use most recent season for report
    max_season = df["season"].max()
    current = df[df["season"] == max_season]
    
    for pos_group in ["F", "D"]:
        pos_data = current[current["position_group"] == pos_group]
        
        lines.append(f"\n{'FORWARDS' if pos_group == 'F' else 'DEFENSEMEN'} ({max_season}):")
        lines.append("-" * 40)
        
        if "archetype" not in pos_data.columns:
            lines.append("  (no clusters computed)")
            continue
        
        for archetype in sorted(pos_data["archetype"].unique()):
            arch_players = pos_data[pos_data["archetype"] == archetype]
            lines.append(f"\n  {archetype} (n={len(arch_players)}):")
            
            # Show category averages (Signal)
            for cat in METRIC_CATEGORIES.keys():
                pct_col = f"{cat}_signal_percentile"
                if pct_col in arch_players.columns:
                    avg_pct = arch_players[pct_col].mean()
                    lines.append(f"    Avg {cat}: {avg_pct:.1f}%")
            
            # Show top 3 examples
            if "OFFENSE_signal_percentile" in arch_players.columns:
                top = arch_players.nlargest(3, "OFFENSE_signal_percentile")
                examples = ", ".join(top["full_name"].tolist())
                lines.append(f"    Examples: {examples}")
    
    return "\n".join(lines)


def main():
    print("Step 03: Clustering players into archetypes...")
    
    # Load data
    input_file = DATA_DIR / "player_categories.csv"
    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run 02_categorize.py first.")
        return None
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    
    # Cluster forwards
    forwards = cluster_players(df, "F", CLUSTERS_FORWARD)
    print(f"Clustered {len(forwards[forwards['season'] == '20242025']):,} forwards into {CLUSTERS_FORWARD} archetypes")
    
    # Cluster defensemen
    defensemen = cluster_players(df, "D", CLUSTERS_DEFENSE)
    print(f"Clustered {len(defensemen[defensemen['season'] == '20242025']):,} defensemen into {CLUSTERS_DEFENSE} archetypes")
    
    # Combine
    result = pd.concat([forwards, defensemen], ignore_index=True)
    
    # Generate report
    report = generate_report(result)
    print(report)
    
    # Save outputs
    output_csv = DATA_DIR / "player_clusters.csv"
    result.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    report_file = DATA_DIR / "cluster_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"Saved: {report_file}")
    
    return result


if __name__ == "__main__":
    main()
