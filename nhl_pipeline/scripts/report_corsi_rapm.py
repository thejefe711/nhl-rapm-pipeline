#!/usr/bin/env python3
"""
Report: Top/bottom players by RAPM metrics (5v5).

Reads DuckDB `nhl_canonical.duckdb` table `apm_results` and prints per-season
top/bottom leaderboards with player names. Can also show a correlation matrix
across metrics for a given season (complete cases only).

Usage:
  python report_corsi_rapm.py --top 20
  python report_corsi_rapm.py --metrics corsi_rapm_5v5,corsi_off_rapm_5v5,corsi_def_rapm_5v5 --corr-season 20252026
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    raise


def load_player_names(con: "duckdb.DuckDBPyConnection", root: Path) -> pd.DataFrame:
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    frames = []

    # 1) DuckDB players table (if it exists and is populated)
    if "players" in tables:
        df = con.execute("SELECT player_id, full_name FROM players").df()
        df = df.dropna(subset=["player_id"]).copy()
        df["player_id"] = df["player_id"].astype(int)
        df = df.dropna(subset=["full_name"])
        df = df[df["full_name"].astype(str).str.strip() != ""]
        if not df.empty:
            frames.append(df.drop_duplicates(subset=["player_id"]))

    # Fallback: derive names from staging shifts parquet
    rows = []
    for p in glob.glob(str(root / "staging" / "*" / "*_shifts.parquet")):
        df = pd.read_parquet(p, columns=["player_id", "first_name", "last_name"])
        df = df.dropna(subset=["player_id"]).copy()
        df["player_id"] = df["player_id"].astype(int)
        df["first_name"] = df["first_name"].fillna("")
        df["last_name"] = df["last_name"].fillna("")
        df["full_name"] = (df["first_name"].str.strip() + " " + df["last_name"].str.strip()).str.strip()
        df = df[df["full_name"] != ""]
        rows.append(df[["player_id", "full_name"]].drop_duplicates())

    if rows:
        frames.append(pd.concat(rows, ignore_index=True).drop_duplicates(subset=["player_id"]))

    # Final fallback: derive names from raw boxscore JSON (often most reliable)
    box_rows = []
    for p in glob.glob(str(root / "raw" / "*" / "*" / "boxscore.json")):
        try:
            data = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue

        pbgs = data.get("playerByGameStats", {})
        for team_key in ("homeTeam", "awayTeam"):
            team = pbgs.get(team_key, {}) or {}
            for group_key in ("forwards", "defense", "goalies"):
                for pl in team.get(group_key, []) or []:
                    pid = pl.get("playerId")
                    name = pl.get("name", {})
                    # Typical structure: {"default": "First Last"}
                    full = None
                    if isinstance(name, dict):
                        full = name.get("default") or name.get("fullName")
                    elif isinstance(name, str):
                        full = name
                    if pid is not None and full:
                        full = str(full).strip()
                        if full:
                            box_rows.append({"player_id": int(pid), "full_name": full})

    if box_rows:
        frames.append(pd.DataFrame(box_rows).drop_duplicates(subset=["player_id"]))

    if not frames:
        return pd.DataFrame(columns=["player_id", "full_name"])

    # Prefer earlier sources (DuckDB > shifts > boxscore) by concatenation order.
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["player_id", "full_name"])
    merged["full_name"] = merged["full_name"].astype(str).str.strip()
    merged = merged[merged["full_name"] != ""]
    merged = merged.drop_duplicates(subset=["player_id"], keep="first")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Report RAPM metrics (5v5)")
    parser.add_argument("--top", type=int, default=20, help="Top N players to show per season")
    parser.add_argument("--metric", type=str, default="corsi_rapm_5v5", help="Metric name in apm_results")
    parser.add_argument("--metrics", type=str, default=None, help="Comma-separated metrics to report (overrides --metric)")
    parser.add_argument("--bottom", type=int, default=None, help="Bottom N players to show per season (default: same as --top)")
    parser.add_argument("--corr-season", type=str, default=None, help="Season to compute correlations for (requires --metrics with 2+ metrics)")
    args = parser.parse_args()
    bottom_n = int(args.bottom) if args.bottom is not None else int(args.top)

    root = Path(__file__).parent.parent
    db_path = root / "nhl_canonical.duckdb"

    con = duckdb.connect(str(db_path))

    metrics = [args.metric]
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    placeholders = ", ".join(["?"] * len(metrics))
    apm = con.execute(
        f"SELECT season, metric_name, player_id, value FROM apm_results WHERE metric_name IN ({placeholders})",
        metrics,
    ).df()

    if apm.empty:
        print(f"No rows found in apm_results for metrics={metrics!r}")
        return

    apm["player_id"] = apm["player_id"].astype(int)
    names = load_player_names(con, root=root)
    con.close()

    out = apm.merge(names, on="player_id", how="left")
    out["full_name"] = out["full_name"].fillna(out["player_id"].astype(str))

    for metric in metrics:
        mdf = out[out["metric_name"] == metric].copy()
        if mdf.empty:
            continue

        for season in sorted(mdf["season"].unique(), reverse=True):
            sdf = mdf[mdf["season"] == season].copy()
            if sdf.empty:
                continue

            top_df = sdf.sort_values("value", ascending=False).head(int(args.top))
            bot_df = sdf.sort_values("value", ascending=True).head(int(bottom_n))

            print(f"\n=== {season} Top {args.top} {metric} ===")
            print(top_df[["full_name", "player_id", "value"]].to_string(index=False, formatters={"value": lambda x: f"{x: .3f}"}))

            print(f"\n=== {season} Bottom {bottom_n} {metric} ===")
            print(bot_df[["full_name", "player_id", "value"]].to_string(index=False, formatters={"value": lambda x: f"{x: .3f}"}))

    if args.corr_season and len(metrics) >= 2:
        season = args.corr_season
        sdf = out[out["season"] == season].copy()
        if sdf.empty:
            print(f"\nNo rows for corr-season={season!r}")
            return
        piv = sdf.pivot_table(index="player_id", columns="metric_name", values="value", aggfunc="mean")
        piv = piv.dropna(axis=0, how="any")
        if piv.empty:
            print(f"\nNo complete cases for corr-season={season!r} across metrics={metrics!r}")
            return
        corr = piv.corr()
        print(f"\n=== Correlations (season={season}, complete_cases={len(piv)}) ===")
        print(corr.to_string())


if __name__ == "__main__":
    main()

