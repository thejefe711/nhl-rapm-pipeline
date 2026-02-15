
import pandas as pd
import duckdb
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import RidgeCV
from typing import Dict, List, Optional, Tuple

# --- Helper Functions from compute_corsi_apm.py ---

def _skater_cols(prefix: str) -> List[str]:
    return [f"{prefix}_skater_{i}" for i in range(1, 7)]

def _collect_players_from_onice(df: pd.DataFrame, home_cols: List[str], away_cols: List[str]) -> List[int]:
    players = pd.concat([df[home_cols], df[away_cols]], axis=0).stack().dropna().astype(int).unique().tolist()
    return players

def _build_sparse_X_off_def(
    df: pd.DataFrame,
    player_to_col: Dict[int, int],
    off_cols: List[str],
    def_cols: List[str],
) -> csr_matrix:
    """
    Build sparse design matrix with separate offense/defense coefficients per player.
    Vectorized implementation.
    """
    # Create a mapping Series
    pid_map = pd.Series(player_to_col)
    
    # Process Offense
    # Melt off_cols to long format: index -> player_id
    off_long = df[off_cols].reset_index().melt(id_vars="index", value_name="player_id").dropna()
    off_long["player_id"] = off_long["player_id"].astype(int)
    
    # Map player_id to column index
    # We need to filter out players not in player_to_col (though they should be there if we built it from data)
    off_long["col_base"] = off_long["player_id"].map(pid_map)
    off_long = off_long.dropna(subset=["col_base"])
    
    # Offense columns are at 2 * base
    off_long["col_idx"] = (2 * off_long["col_base"]).astype(int)
    
    # Add scale (vectorized lookup)
    # If off_scale is in df, map it. Otherwise 1.0.
    if "off_scale" in df.columns:
        off_long["val"] = df.loc[off_long["index"], "off_scale"].values
    else:
        off_long["val"] = 1.0
        
    # Process Defense
    def_long = df[def_cols].reset_index().melt(id_vars="index", value_name="player_id").dropna()
    def_long["player_id"] = def_long["player_id"].astype(int)
    
    def_long["col_base"] = def_long["player_id"].map(pid_map)
    def_long = def_long.dropna(subset=["col_base"])
    
    # Defense columns are at 2 * base + 1
    def_long["col_idx"] = (2 * def_long["col_base"] + 1).astype(int)
    
    if "def_scale" in df.columns:
        def_long["val"] = df.loc[def_long["index"], "def_scale"].values
    else:
        def_long["val"] = 1.0
        
    # Combine
    combined = pd.concat([off_long, def_long], ignore_index=True)
    
    # Build CSR
    row_idx = combined["index"].values
    col_idx = combined["col_idx"].values
    data = combined["val"].values
    
    return csr_matrix((data, (row_idx, col_idx)), shape=(len(df), 2 * len(player_to_col)))

from sklearn.linear_model import Ridge

def _ridge_fit(X: csr_matrix, y: np.ndarray, sample_weight: Optional[np.ndarray], alphas: List[float]) -> Tuple[np.ndarray, float]:
    # Use fixed alpha=100.0 and lsqr solver as per spec
    alpha = 100.0
    if alphas and len(alphas) > 0:
         alpha = 100.0
    
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(X, y, sample_weight=sample_weight)
    print(f"DEBUG: _ridge_fit used Ridge(solver='lsqr', alpha={alpha})")
    return model.coef_, float(model.alpha)

def _write_apm_results(
    conn: "duckdb.DuckDBPyConnection",
    season: str,
    metric_name: str,
    coefs: Dict[int, float],
    toi_seconds: Dict[int, int],
    events_count: int,
    games_count: int,
) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS apm_results (
            season VARCHAR NOT NULL,
            metric_name VARCHAR NOT NULL,
            player_id INTEGER NOT NULL,
            value DOUBLE NOT NULL,
            games_count INTEGER,
            toi_seconds INTEGER,
            events_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (season, metric_name, player_id)
        );
        """
    )

    rows = []
    for pid, val in coefs.items():
        rows.append(
            {
                "season": season,
                "metric_name": metric_name,
                "player_id": int(pid),
                "value": float(val),
                "games_count": int(games_count),
                "toi_seconds": int(toi_seconds.get(pid, 0)),
                "events_count": int(events_count),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)
    print(f"DEBUG: _write_apm_results writing {len(df)} rows for {metric_name}")
    conn.execute(
        """
        INSERT OR REPLACE INTO apm_results
            (season, metric_name, player_id, value, games_count, toi_seconds, events_count)
        SELECT
            season, metric_name, player_id, value, games_count, toi_seconds, events_count
        FROM df
        """
    )

def _time_based_alpha_search(
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    alphas: np.ndarray,
    n_splits: int = 1,
    val_ratio: float = 0.2
) -> float:
    """
    Perform time-based split cross-validation to find optimal alpha.
    Assumes X, y are sorted chronologically.
    """
    n_samples = X.shape[0]
    split_idx = int(n_samples * (1 - val_ratio))
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    w_train = weights[:split_idx]
    
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    w_val = weights[split_idx:]
    
    best_alpha = 100.0
    best_mse = float("inf")
    
    print(f"  Alpha Search: Train={X_train.shape[0]}, Val={X_val.shape[0]}")
    
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        model.fit(X_train, y_train, sample_weight=w_train)
        
        preds = model.predict(X_val)
        mse = np.average((y_val - preds) ** 2, weights=w_val)
        
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            
    print(f"  Best Alpha: {best_alpha:.4f} (MSE: {best_mse:.6e})")
    return best_alpha

def recover_rapm(season="20242025"):
    # User requested alphas: 0.01 to 100,000 (50 steps)
    alphas = np.logspace(-2, 5, 50)
    
    staging_dir = Path("staging")
    season_dir = staging_dir / season
    # Use absolute path to be sure
    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    print(f"Using DB path: {db_path.resolve()}")
    
    print(f"Recovering RAPM for season {season} from {season_dir}...")
    
    # Find all partial files
    partial_files = sorted(season_dir.glob("rapm_partial_*.parquet"))
    if not partial_files:
        print("No partial files found!")
        return

    print(f"Found {len(partial_files)} partial files.")
    
    dfs = []
    for p in partial_files:
        print(f"  Loading {p}...")
        dfs.append(pd.read_parquet(p))
        
    if not dfs:
        print("No data loaded.")
        return
        
    data = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(data)}")
    
    # Filter for valid duration
    data = data[data["duration_s"] > 0].copy()
    
    # Re-calculate TOI
    print("Re-calculating TOI from stints...")
    home_cols = _skater_cols("home")
    away_cols = _skater_cols("away")
    
    # We need to sum duration for each player
    # This is a bit slow if we iterate. Let's use melt.
    
    # Melt home players
    home_melt = data.melt(id_vars=["duration_s"], value_vars=home_cols, value_name="player_id").dropna()
    home_toi = home_melt.groupby("player_id")["duration_s"].sum()
    
    # Melt away players
    away_melt = data.melt(id_vars=["duration_s"], value_vars=away_cols, value_name="player_id").dropna()
    away_toi = away_melt.groupby("player_id")["duration_s"].sum()
    
    # Combine
    combined_toi = home_toi.add(away_toi, fill_value=0).to_dict()
    toi_seconds = {int(k): int(v) for k, v in combined_toi.items()}
    print(f"Re-calculated TOI for {len(toi_seconds)} players.")
    
    # Build player mapping
    players_sorted = sorted(toi_seconds.keys())
    player_to_col = {pid: i for i, pid in enumerate(players_sorted)}
    
    conn = duckdb.connect(str(db_path))
    
    # --- Run Regression for Corsi Off/Def ---
    print("Running Ridge Regression for Corsi Off/Def...")
    
    off_cols = [f"off_skater_{i}" for i in range(1, 7)]
    def_cols = [f"def_skater_{i}" for i in range(1, 7)]
    
    # Vectorized observation construction
    # We need to create two rows for each stint:
    # 1. Home Offense vs Away Defense
    # 2. Away Offense vs Home Defense
    
    # Prepare base DataFrame
    # We need columns: duration_s, corsi_home, corsi_away, xg_home, xg_away
    # And the player columns.
    
    # Create a long-form DataFrame for players to map them to off/def columns
    # This is tricky with variable number of players.
    # But we know we have exactly 6 columns for home and 6 for away.
    
    # Strategy:
    # 1. Create obs_home: Home is Offense, Away is Defense
    # 2. Create obs_away: Away is Offense, Home is Defense
    # 3. Concatenate
    
    # 1. obs_home
    obs_home = data.copy()
    obs_home["y_corsi"] = obs_home["corsi_home"].astype(float) / obs_home["duration_s"]
    obs_home["y_xg"] = obs_home["xg_home"].astype(float) / obs_home["duration_s"]
    obs_home["weight"] = obs_home["duration_s"]
    
    # Rename columns to off_skater_X and def_skater_X
    rename_home = {c: c.replace("home_skater", "off_skater") for c in home_cols}
    rename_home.update({c: c.replace("away_skater", "def_skater") for c in away_cols})
    obs_home = obs_home.rename(columns=rename_home)
    
    # Keep only relevant columns
    keep_cols = off_cols + def_cols + ["y_corsi", "y_xg", "weight"]
    obs_home = obs_home[keep_cols]
    
    # 2. obs_away
    obs_away = data.copy()
    obs_away["y_corsi"] = obs_away["corsi_away"].astype(float) / obs_away["duration_s"]
    obs_away["y_xg"] = obs_away["xg_away"].astype(float) / obs_away["duration_s"]
    obs_away["weight"] = obs_away["duration_s"]
    
    # Rename columns: Away is Offense, Home is Defense
    rename_away = {c: c.replace("away_skater", "off_skater") for c in away_cols}
    rename_away.update({c: c.replace("home_skater", "def_skater") for c in home_cols})
    obs_away = obs_away.rename(columns=rename_away)
    
    obs_away = obs_away[keep_cols]
    
    # 3. Concatenate
    obs_df = pd.concat([obs_home, obs_away], ignore_index=True)
    
    # Drop rows with missing players (should be none if data is clean)
    # Actually, if a stint has < 6 players, the columns are None/NaN.
    # The sparse builder handles NaNs by dropping them.
    # So we don't need to drop rows, just ensure NaNs are handled.
    # The original script did `dropna(subset=[off_cols[0], def_cols[0]])` which implies at least 1 player.
    
    print(f"Observation rows: {len(obs_df)}")
    
    X = _build_sparse_X_off_def(obs_df, player_to_col, off_cols, def_cols)
    
    # Corsi
    print("Fitting Corsi...")
    best_alpha = _time_based_alpha_search(X, obs_df["y_corsi"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
    coefs, alpha_used = _ridge_fit(X, obs_df["y_corsi"].values.astype(float), obs_df["weight"].values.astype(float), [best_alpha])
    coefs = coefs * 3600.0
    off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
    def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
    
    _write_apm_results(conn, season, "corsi_off_rapm_5v5", off_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), 650)
    _write_apm_results(conn, season, "corsi_def_rapm_5v5", def_map, toi_seconds, int(data["corsi_home"].sum() + data["corsi_away"].sum()), 650)
    
    # xG
    print("Fitting xG...")
    best_alpha = _time_based_alpha_search(X, obs_df["y_xg"].values.astype(float), obs_df["weight"].values.astype(float), alphas)
    coefs, alpha_used = _ridge_fit(X, obs_df["y_xg"].values.astype(float), obs_df["weight"].values.astype(float), [best_alpha])
    coefs = coefs * 3600.0
    off_map = {pid: float(coefs[2 * player_to_col[pid]]) for pid in players_sorted}
    def_map = {pid: float(-coefs[2 * player_to_col[pid] + 1]) for pid in players_sorted}
    
    _write_apm_results(conn, season, "xg_off_rapm_5v5", off_map, toi_seconds, int(len(data)), 650)
    _write_apm_results(conn, season, "xg_def_rapm_5v5", def_map, toi_seconds, int(len(data)), 650)
    
    # Verify immediately
    print("Verifying write...")
    res = conn.execute(f"SELECT COUNT(*) FROM apm_results WHERE season='{season}'").fetchone()
    print(f"Rows in DB for {season}: {res[0]}")
    
    conn.close()
    print("Done!")

if __name__ == "__main__":
    recover_rapm()
