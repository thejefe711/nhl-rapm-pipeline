"""
Configuration for player profile pipeline.
"""
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "nhl_canonical.duckdb"
DATA_DIR = ROOT / "profile_data"
DATA_DIR.mkdir(exist_ok=True)

# Position handling
FORWARD_POSITIONS = {"C", "LW", "RW", "F", "L", "R"}
DEFENSE_POSITIONS = {"D", "LD", "RD"}

def is_forward(position: str) -> bool:
    return position.upper() in FORWARD_POSITIONS

def is_defenseman(position: str) -> bool:
    return position.upper() in DEFENSE_POSITIONS

def position_group(position: str) -> str:
    if is_forward(position):
        return "F"
    elif is_defenseman(position):
        return "D"
    return "UNKNOWN"

# Metric categorization - 26 RAPM metrics grouped into 6 categories
METRIC_CATEGORIES = {
    "OFFENSE": [
        "corsi_off_rapm_5v5",
        "xg_off_rapm_5v5",
        "hd_xg_off_rapm_5v5_ge020",
    ],
    "DEFENSE": [
        "corsi_def_rapm_5v5",
        "xg_def_rapm_5v5",
        "hd_xg_def_rapm_5v5_ge020",
    ],
    "TRANSITION": [
        "turnover_to_xg_swing_rapm_5v5_w10",
        "takeaway_to_xg_swing_rapm_5v5_w10",
        "giveaway_to_xg_swing_rapm_5v5_w10",
        "blocked_shot_to_xg_swing_rapm_5v5_w10",
        "faceoff_loss_to_xg_swing_rapm_5v5_w10",
    ],
    "SPECIAL_TEAMS": [
        "corsi_pp_off_rapm",
        "xg_pp_off_rapm",
        "corsi_pk_def_rapm",
        "xg_pk_def_rapm",
    ],
    "DISCIPLINE": [
        "penalties_taken_rapm_5v5",
        "penalties_drawn_rapm_5v5",
    ],
    "FINISHING": [
        "finishing_residual_rapm_5v5",
        "goals_rapm_5v5",
        "primary_assist_rapm_5v5",
        "secondary_assist_rapm_5v5",
    ],
    "USAGE": [
        "toi_per_game",
        "pp_time_pct",
        "pk_time_pct",
    ],
}

# Recommendation A: Key metrics for the "Signal" feature
SIGNAL_METRICS = [
    "corsi_off_rapm_5v5",
    "xg_off_rapm_5v5",
    "goals_rapm_5v5",
    "corsi_def_rapm_5v5",
    "xg_def_rapm_5v5",
]

# Flatten for easy lookup
ALL_METRICS = []
METRIC_TO_CATEGORY = {}
for cat, metrics in METRIC_CATEGORIES.items():
    for m in metrics:
        ALL_METRICS.append(m)
        METRIC_TO_CATEGORY[m] = cat

# Clustering
CLUSTERS_FORWARD = 6
CLUSTERS_DEFENSE = 5

# Narrative generation
MAX_PLAYERS_FOR_NARRATIVES = 200

# For discipline metrics, lower is better (fewer penalties taken)
# For most others, higher is better
LOWER_IS_BETTER = {"penalties_taken_rapm_5v5", "giveaway_to_xg_swing_rapm_5v5_w10"}

# Season ordering for trend analysis
SEASON_ORDER = [
    "20202021",
    "20212022",
    "20222023",
    "20232024",
    "20242025",
    "20252026",
]

# Current season for analysis
CURRENT_SEASON = 20252026


def get_current_season_data(df, season_col="season"):
    """
    Filter dataframe to current season, handling int/string formats.
    Falls back to most recent season if current not found.
    """
    import pandas as pd
    
    # Convert to consistent format
    df = df.copy()
    df[season_col] = df[season_col].astype(int)
    
    current = df[df[season_col] == CURRENT_SEASON]
    
    if current.empty:
        max_season = df[season_col].max()
        current = df[df[season_col] == max_season]
        print(f"Note: Using season {max_season} (current season {CURRENT_SEASON} not found)")
    
    return current
