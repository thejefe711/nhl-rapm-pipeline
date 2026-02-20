"""
Minimal FastAPI server for serving computed NHL metrics from DuckDB.

Run (from nhl_pipeline/):
  pip install -r requirements.txt
  python -m uvicorn api_server:app --reload --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).parent.parent.parent
DUCKDB_PATH = ROOT / "nhl_canonical.duckdb"
STAGING_DIR = ROOT / "staging"
RAW_DIR = ROOT / "raw"


# ---------------------------------------------------------------------------
# Connection singleton — one read-only connection shared across all requests.
# Replaces the previous per-request duckdb.connect() pattern.
# ---------------------------------------------------------------------------
_DB: dict[str, Any] = {}  # populated by lifespan


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open one read-only DuckDB connection at startup; close on shutdown."""
    if DUCKDB_PATH.exists():
        _DB["con"] = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    else:
        _DB["con"] = None
    try:
        yield
    finally:
        con = _DB.pop("con", None)
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _connect() -> "duckdb.DuckDBPyConnection":
    """Return the shared read-only DuckDB connection.

    Raises HTTP 503 if the database file was not found at startup.
    Individual endpoint con.close() calls are harmless no-ops since the
    lifespan context manager owns the actual teardown.
    """
    con = _DB.get("con")
    if con is None:
        raise HTTPException(
            status_code=503,
            detail=f"DuckDB not found at {str(DUCKDB_PATH)}. Run the pipeline first.",
        )
    return con


def _players_table_exists(con: "duckdb.DuckDBPyConnection") -> bool:
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        return "players" in tables
    except Exception:
        return False


def _get_player_name(con: "duckdb.DuckDBPyConnection", player_id: int) -> Optional[str]:
    """Efficiently fetch player name from the players table."""
    try:
        row = con.execute("SELECT full_name FROM players WHERE player_id = ?", [int(player_id)]).fetchone()
        return str(row[0]) if row else None
    except Exception:
        return None


app = FastAPI(title="NHL Pipeline API", version="0.1.0", lifespan=_lifespan)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "duckdb_path": str(DUCKDB_PATH), "duckdb_exists": DUCKDB_PATH.exists()}


@app.get("/api/seasons")
def seasons() -> dict[str, Any]:
    con = _connect()
    try:
        rows = con.execute("SELECT DISTINCT season FROM apm_results ORDER BY season").fetchall()
        return {"seasons": [r[0] for r in rows]}
    finally:
        con.close()


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    con = _connect()
    try:
        rows = con.execute("SELECT metric_name, COUNT(*) AS n FROM apm_results GROUP BY 1 ORDER BY 1").fetchall()
        return {"metrics": [{"metric_name": r[0], "rows": int(r[1])} for r in rows]}
    finally:
        con.close()


@app.get("/api/latent-models")
def latent_models() -> dict[str, Any]:
    con = _connect()
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "latent_models" not in tables:
            return {"models": []}
        df = con.execute(
            "SELECT model_name, model_type, n_components, alpha, trained_at, n_samples FROM latent_models ORDER BY trained_at DESC"
        ).df()
        return {"models": df.to_dict(orient="records")}
    finally:
        con.close()


@app.get("/api/latent-models/{model_name}/dimensions")
def latent_model_dimensions(
    model_name: str,
    stable_threshold: int = Query(default=3, ge=1, le=20, description="Minimum stable_seasons to flag is_stable=true"),
) -> dict[str, Any]:
    """
    Dimension metadata for a latent model (sticky labels + stability), if available.

    Produced by: scripts/train_sae_apm.py (writes latent_dim_meta).
    """
    con = _connect()
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "latent_dim_meta" not in tables:
            return {"model": model_name, "rows": []}

        df = con.execute(
            """
            SELECT model_name, dim_idx, label, top_features_json, stable_seasons, seasons_active_json
            FROM latent_dim_meta
            WHERE model_name = ?
            ORDER BY dim_idx
            """,
            [model_name],
        ).df()

        rows = df.to_dict(orient="records")
        for r in rows:
            try:
                r["is_stable"] = int(r.get("stable_seasons") or 0) >= int(stable_threshold)
            except Exception:
                r["is_stable"] = False

        return {"model": model_name, "stable_threshold": int(stable_threshold), "rows": rows}
    finally:
        con.close()


@app.get("/api/metric-catalog")
def metric_catalog() -> dict[str, Any]:
    """
    Small hand-curated catalog for UI toggles.

    This lets the frontend offer human-friendly choices while still using raw metric_name strings underneath.
    """
    return {
        "defense": {
            "volume_suppression": {"metric_name": "corsi_def_rapm_5v5", "label": "Defense: Volume suppression (Corsi)"},
            "hd_suppression": {"metric_name": "hd_xg_def_rapm_5v5_ge020", "label": "Defense: HD suppression (HD xG >= 0.20)"},
        },
        "special_teams": {
            "pp_quarterback": {"metric_name": "xg_pp_off_rapm", "label": "PP impact (xG offense)"},
            "pk_stopper": {"metric_name": "xg_pk_def_rapm", "label": "PK impact (xG defense)"},
        },
    }


@app.get("/api/player/{player_id}/latent-skills")
def player_latent_skills(
    player_id: int,
    model: str = Query(..., description="Model name from /api/latent-models"),
    season: Optional[str] = Query(default=None, description="Optional season like 20252026 (if omitted: all seasons)"),
    include_dim_meta: bool = Query(default=False, description="If true, attach label/stability for each dim when available"),
    stable_threshold: int = Query(default=3, ge=1, le=20, description="Minimum stable_seasons to flag is_stable=true (when include_dim_meta=true)"),
) -> dict[str, Any]:
    con = _connect()
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "latent_skills" not in tables:
            raise HTTPException(status_code=404, detail="latent_skills table not found; train SAE first")

        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        has_meta = include_dim_meta and ("latent_dim_meta" in tables)

        if season is None:
            if has_meta:
                df = con.execute(
                    """
                    SELECT s.season, s.dim_idx, s.value, m.label, m.stable_seasons
                    FROM latent_skills s
                    LEFT JOIN latent_dim_meta m
                      ON m.model_name = s.model_name AND m.dim_idx = s.dim_idx
                    WHERE s.model_name = ? AND s.player_id = ?
                    ORDER BY s.season, s.dim_idx
                    """,
                    [model, int(player_id)],
                ).df()
            else:
                df = con.execute(
                    "SELECT season, dim_idx, value FROM latent_skills WHERE model_name = ? AND player_id = ? ORDER BY season, dim_idx",
                    [model, int(player_id)],
                ).df()
        else:
            if has_meta:
                df = con.execute(
                    """
                    SELECT s.season, s.dim_idx, s.value, m.label, m.stable_seasons
                    FROM latent_skills s
                    LEFT JOIN latent_dim_meta m
                      ON m.model_name = s.model_name AND m.dim_idx = s.dim_idx
                    WHERE s.model_name = ? AND s.player_id = ? AND s.season = ?
                    ORDER BY s.dim_idx
                    """,
                    [model, int(player_id), season],
                ).df()
            else:
                df = con.execute(
                    "SELECT season, dim_idx, value FROM latent_skills WHERE model_name = ? AND player_id = ? AND season = ? ORDER BY dim_idx",
                    [model, int(player_id), season],
                ).df()

        rows = df.to_dict(orient="records")
        if has_meta:
            for r in rows:
                try:
                    r["is_stable"] = int(r.get("stable_seasons") or 0) >= int(stable_threshold)
                except Exception:
                    r["is_stable"] = False

        return {
            "player_id": int(player_id),
            "model": model,
            "season": season,
            "include_dim_meta": bool(include_dim_meta),
            "stable_threshold": int(stable_threshold) if has_meta else None,
            "rows": rows,
        }
    finally:
        con.close()


@app.get("/api/player/{player_id}/dlm-forecast")
def player_dlm_forecast(
    player_id: int,
    model: str = Query(..., description="Latent model name (same as rolling_latent_skills.model_name)"),
    season: str = Query(..., description="Season like 20242025"),
    window: int = Query(default=10, ge=1, le=82, description="Rolling window size used for rolling_latent_skills"),
    horizon: int = Query(default=3, ge=1, le=10, description="Forecast horizon in games"),
    stable_threshold: int = Query(default=3, ge=1, le=20, description="Minimum stable_seasons to flag is_stable=true (if dim meta present)"),
) -> dict[str, Any]:
    """
    Return latest DLM/Kalman forecasts for a player's latent dims.

    Requires scripts/compute_rolling_latents.py + scripts/compute_dlm_forecasts.py to have been run.
    """
    con = _connect()
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "dlm_forecasts" not in tables:
            raise HTTPException(status_code=404, detail="dlm_forecasts table not found; run compute_dlm_forecasts.py first")

        # latest window_end_game_id for this player/season/window
        latest = con.execute(
            """
            SELECT window_end_game_id
            FROM dlm_forecasts
            WHERE model_name = ? AND season = ? AND window_size = ? AND player_id = ? AND horizon_games = ?
            ORDER BY window_end_time_utc DESC NULLS LAST, window_end_game_id DESC
            LIMIT 1
            """,
            [model, season, int(window), int(player_id), int(horizon)],
        ).fetchone()
        if not latest:
            return {"player_id": int(player_id), "model": model, "season": season, "window": int(window), "horizon": int(horizon), "rows": []}

        window_end_game_id = str(latest[0])

        df = con.execute(
            """
            SELECT dim_idx, forecast_mean, forecast_var, filtered_mean, filtered_var, n_obs, q, r, window_end_game_id, window_end_time_utc
            FROM dlm_forecasts
            WHERE model_name = ? AND season = ? AND window_size = ? AND player_id = ? AND horizon_games = ? AND window_end_game_id = ?
            ORDER BY dim_idx
            """,
            [model, season, int(window), int(player_id), int(horizon), window_end_game_id],
        ).df()

        rows = df.to_dict(orient="records")

        # Attach dim labels/stability if available
        if "latent_dim_meta" in tables and rows:
            meta = con.execute(
                """
                SELECT dim_idx, label, stable_seasons
                FROM latent_dim_meta
                WHERE model_name = ?
                """,
                [model],
            ).df()
            meta_map = {int(r["dim_idx"]): {"label": r["label"], "stable_seasons": int(r["stable_seasons"])} for _, r in meta.iterrows()}
            for r in rows:
                k = int(r["dim_idx"])
                m = meta_map.get(k)
                if m:
                    r["label"] = m["label"]
                    r["stable_seasons"] = m["stable_seasons"]
                    r["is_stable"] = int(m["stable_seasons"]) >= int(stable_threshold)

        return {
            "player_id": int(player_id),
            "model": model,
            "season": season,
            "window": int(window),
            "horizon": int(horizon),
            "window_end_game_id": window_end_game_id,
            "stable_threshold": int(stable_threshold),
            "rows": rows,
        }
    finally:
        con.close()


@app.get("/api/players/search")
def player_search(
    q: str = Query(..., min_length=2, description="Case-insensitive substring match (e.g. 'mcdavid')"),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    """
    Lightweight player search for UI selection.

    Sources:
    - DuckDB players table if available
    - Fallback: cached scan from staging/raw
    """
    qn = q.strip().lower()
    if not qn:
        return {"q": q, "rows": []}

    con = _connect()
    try:
        rows: list[dict[str, Any]] = []

        if _players_table_exists(con):
            df = con.execute(
                "SELECT player_id, full_name FROM players WHERE LOWER(full_name) LIKE ? ORDER BY full_name LIMIT ?",
                [f"%{qn}%", int(limit)],
            ).df()
            rows = df.to_dict(orient="records")
        else:
            nm = _cached_name_map()
            # simple substring filter over cached names
            hits = [(pid, name) for pid, name in nm.items() if qn in str(name).lower()]
            hits.sort(key=lambda x: x[1])
            rows = [{"player_id": int(pid), "full_name": name} for pid, name in hits[: int(limit)]]

        return {"q": q, "limit": int(limit), "rows": rows}
    finally:
        con.close()


@app.get("/api/players/{player_id}")
def player_detail(player_id: int) -> dict[str, Any]:
    """
    Player metadata (best effort).
    """
    con = _connect()
    try:
        player_data: dict[str, Any] = {"player_id": int(player_id)}
        
        # Get basic player info from players table
        if _players_table_exists(con):
            df = con.execute(
                "SELECT player_id, first_name, last_name, full_name, first_seen_game_id, last_seen_game_id "
                "FROM players WHERE player_id = ?",
                [int(player_id)],
            ).df()
            if not df.empty:
                player_data = df.iloc[0].to_dict()
        
        # Fallback for name cleanup
        if "full_name" not in player_data or not player_data.get("full_name"):
            full = _get_player_name(con, player_id)
            if full:
                player_data["full_name"] = full
        
        if "full_name" not in player_data:
            raise HTTPException(status_code=404, detail="player_id not found")
        
        # Calculate seasons_count from apm_results (reliable)
        # Note: games_count in database is unreliable, so we show seasons instead
        try:
            seasons_df = con.execute(
                """
                SELECT COUNT(DISTINCT season) as seasons_count
                FROM apm_results
                WHERE player_id = ? AND metric_name = 'corsi_rapm_5v5'
                """,
                [int(player_id)],
            ).df()
            if not seasons_df.empty and seasons_df.iloc[0]["seasons_count"] > 0:
                player_data["seasons_count"] = int(seasons_df.iloc[0]["seasons_count"])
        except Exception:
            pass

        
        return {"player": player_data}
    finally:
        con.close()



@app.get("/api/leaderboards/corsi-rapm")
def corsi_rapm_leaderboard(
    season: Optional[str] = Query(default=None, description="Season like 20242025"),
    top: int = Query(default=20, ge=1, le=200),
    metric: str = Query(default="corsi_rapm_5v5"),
) -> dict[str, Any]:
    """
    Return top-N players by Corsi RAPM for a season (or all seasons if omitted).
    Using SQL JOIN for performance.
    """
    con = _connect()
    try:
        sql = f"""
            SELECT a.season, a.player_id, a.value, p.full_name
            FROM apm_results a
            LEFT JOIN players p ON a.player_id = p.player_id
            WHERE a.metric_name = ?
            {"AND a.season = ?" if season else ""}
            ORDER BY a.value DESC
        """
        params = [metric, season] if season else [metric]
        df = con.execute(sql, params).df()

        if df.empty:
            return {"season": season, "metric": metric, "rows": []}

        if season is None:
            out_rows: list[dict[str, Any]] = []
            for s in sorted(df["season"].unique(), reverse=True):
                sub = df[df["season"] == s].head(top)
                out_rows.extend(sub.to_dict(orient="records"))
            rows = out_rows
        else:
            rows = df.head(top).to_dict(orient="records")

        return {"season": season, "metric": metric, "top": top, "rows": rows}
    finally:
        con.close()


@app.get("/api/leaderboards")
def leaderboard(
    metric: str = Query(..., min_length=1, description="Metric name in apm_results (e.g. corsi_rapm_5v5)"),
    season: Optional[str] = Query(default=None, description="Optional season like 20242025"),
    top: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    """
    Generic leaderboard over `apm_results` using SQL JOIN for performance.
    """
    con = _connect()
    try:
        sql = f"""
            SELECT a.season, a.player_id, a.value, a.games_count, a.events_count, p.full_name
            FROM apm_results a
            LEFT JOIN players p ON a.player_id = p.player_id
            WHERE a.metric_name = ?
            {"AND a.season = ?" if season else ""}
            ORDER BY a.value DESC
        """
        params = [metric, season] if season else [metric]
        df = con.execute(sql, params).df()

        if df.empty:
            return {"season": season, "metric": metric, "top": int(top), "rows": []}

        if season is None:
            out_rows = []
            for s in sorted(df["season"].unique(), reverse=True):
                sub = df[df["season"] == s].head(top)
                out_rows.extend(sub.to_dict(orient="records"))
            rows = out_rows
        else:
            rows = df.head(top).to_dict(orient="records")

        return {"season": season, "metric": metric, "top": int(top), "rows": rows}
    finally:
        con.close()


@app.get("/api/player/{player_id}/rapm")
def player_rapm(
    player_id: int,
    metric: str = Query(default="corsi_rapm_5v5"),
) -> dict[str, Any]:
    """
    Return a player's RAPM values by season for the given metric.
    """
    con = _connect()
    try:
        df = con.execute(
            "SELECT season, player_id, value FROM apm_results WHERE metric_name = ? AND player_id = ? ORDER BY season",
            [metric, int(player_id)],
        ).df()

        full_name = _get_player_name(con, player_id)

        return {
            "player_id": int(player_id),
            "full_name": full_name,
            "metric": metric,
            "rows": df.to_dict(orient="records"),
        }
    finally:
        con.close()


def _generate_player_explanation(player_data: dict) -> str:
    """Generate natural language explanation from player analytics."""
    rows = player_data.get("rows", [])

    if not rows:
        return "Insufficient data to generate player explanation."

    # Analyze stable vs emerging skills
    stable_skills = []
    emerging_skills = []

    for row in rows:
        skill_name = row.get("label", f"Dimension {row['dim_idx']}")
        mean = row["forecast_mean"]
        is_stable = row.get("is_stable", False)
        seasons = row.get("stable_seasons", 0)

        skill_info = {
            "name": skill_name,
            "strength": mean,
            "seasons": seasons
        }

        if is_stable:
            stable_skills.append(skill_info)
        else:
            emerging_skills.append(skill_info)

    # Sort by strength
    stable_skills.sort(key=lambda x: x["strength"], reverse=True)
    emerging_skills.sort(key=lambda x: abs(x["strength"]), reverse=True)

    # Build explanation
    explanation_parts = []

    # Overall assessment
    total_skills = len(stable_skills) + len(emerging_skills)
    stable_count = len(stable_skills)
    emerging_count = len(emerging_skills)

    explanation_parts.append(f"This player demonstrates {stable_count} stable skills (consistent across seasons) and {emerging_count} emerging skills (developing or inconsistent).")

    # Top strengths
    if stable_skills:
        top_stable = [s for s in stable_skills if s["strength"] > 0.1][:3]
        if top_stable:
            strength_names = [s["name"] for s in top_stable]
            explanation_parts.append(f"Their most consistent strengths are in {', '.join(strength_names[:2])}{' and ' + strength_names[2] if len(strength_names) > 2 else ''}.")

    # Emerging skills analysis
    if emerging_skills:
        strong_emerging = [s for s in emerging_skills if s["strength"] > 0.2][:2]
        weak_emerging = [s for s in emerging_skills if s["strength"] < -0.2][:2]

        if strong_emerging:
            emerging_names = [s["name"] for s in strong_emerging]
            explanation_parts.append(f"Emerging skills show significant potential in {', '.join(emerging_names)}.")

        if weak_emerging:
            weak_names = [s["name"] for s in weak_emerging]
            explanation_parts.append(f"Emerging challenges appear in {', '.join(weak_names)}.")

    # Weaknesses
    weak_stable = [s for s in stable_skills if s["strength"] < -0.1][:2]
    if weak_stable:
        weak_names = [s["name"] for s in weak_stable]
        explanation_parts.append(f"Consistent weaknesses appear in {', '.join(weak_names)}.")

    # Development trajectory
    high_emerging = len([s for s in emerging_skills if s["strength"] > 0.2])
    low_emerging = len([s for s in emerging_skills if s["strength"] < -0.2])

    if high_emerging > low_emerging:
        explanation_parts.append("The player's development trajectory suggests improving capabilities.")
    elif low_emerging > high_emerging:
        explanation_parts.append("Development trends indicate ongoing challenges in certain areas.")
    else:
        explanation_parts.append("The player's skills show balanced development patterns.")

    return " ".join(explanation_parts)


@app.get("/api/explanations/player/{player_id}")
def explain_player(
    player_id: int,
    model: str = Query(default="sae_apm_v1_k12_a1", description="Latent model name"),
    season: str = Query(default="20242025", description="Season"),
    window: int = Query(default=10, description="Rolling window size"),
    horizon: int = Query(default=3, description="Forecast horizon")
) -> dict[str, Any]:
    """
    Generate natural language explanation of a player's skills and development.

    Analyzes DLM forecasts and latent skill dimensions to provide insights about:
    - Current strengths and weaknesses
    - Stable vs emerging skills
    - Development trajectory
    - Performance consistency
    """
    # Get player data first (duplicate logic from player_dlm_forecast)
    con = _connect()
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "dlm_forecasts" not in tables:
            raise HTTPException(status_code=404, detail="dlm_forecasts table not found; run compute_dlm_forecasts.py first")

        # latest window_end_game_id for this player/season/window
        latest = con.execute(
            """
            SELECT window_end_game_id
            FROM dlm_forecasts
            WHERE model_name = ? AND season = ? AND window_size = ? AND player_id = ? AND horizon_games = ?
            ORDER BY window_end_time_utc DESC NULLS LAST, window_end_game_id DESC
            LIMIT 1
            """,
            [model, season, int(window), int(player_id), int(horizon)],
        ).fetchone()
        if not latest:
            player_data = {"player_id": int(player_id), "model": model, "season": season, "window": int(window), "horizon": int(horizon), "rows": []}
        else:
            window_end_game_id = str(latest[0])

            df = con.execute(
                """
                SELECT dim_idx, forecast_mean, forecast_var, filtered_mean, filtered_var, n_obs, q, r, window_end_game_id, window_end_time_utc
                FROM dlm_forecasts
                WHERE model_name = ? AND season = ? AND window_size = ? AND player_id = ? AND horizon_games = ? AND window_end_game_id = ?
                ORDER BY dim_idx
                """,
                [model, season, int(window), int(player_id), int(horizon), window_end_game_id],
            ).df()

            rows = df.to_dict(orient="records")

            # Attach dim labels/stability if available
            stable_threshold = 3  # default
            if "latent_dim_meta" in tables and rows:
                meta = con.execute(
                    """
                    SELECT dim_idx, label, stable_seasons
                    FROM latent_dim_meta
                    WHERE model_name = ?
                    """,
                    [model],
                ).df()
                meta_map = {int(r["dim_idx"]): {"label": r["label"], "stable_seasons": int(r["stable_seasons"])} for _, r in meta.iterrows()}
                for r in rows:
                    k = int(r["dim_idx"])
                    m = meta_map.get(k)
                    if m:
                        r["label"] = m["label"]
                        r["stable_seasons"] = m["stable_seasons"]
                        r["is_stable"] = int(m["stable_seasons"]) >= int(stable_threshold)

            player_data = {
                "player_id": int(player_id),
                "model": model,
                "season": season,
                "window": int(window),
                "horizon": int(horizon),
                "window_end_game_id": window_end_game_id,
                "stable_threshold": int(stable_threshold),
                "rows": rows,
            }
    finally:
        con.close()

    if not player_data.get("rows"):
        return {
            "player_id": player_id,
            "explanation": "No forecast data available for this player.",
            "data_quality": "insufficient"
        }

    # Generate explanation
    explanation = _generate_player_explanation(player_data)

    player_name = _get_player_name(con, player_id) or f"Player {player_id}"

    return {
        "player_id": int(player_id),
        "player_name": player_name,
        "model": model,
        "season": season,
        "explanation": explanation,
        "data_quality": "good" if len(player_data["rows"]) >= 8 else "limited",
        "stable_skills": len([r for r in player_data["rows"] if r.get("is_stable", False)]),
        "emerging_skills": len([r for r in player_data["rows"] if not r.get("is_stable", False)]),
        "analysis_timestamp": player_data.get("window_end_game_id")
    }


@app.get("/api/player/{player_id}/profile")
def player_profile(
    player_id: int,
    season: Optional[str] = Query(default=None, description="Season (if omitted, uses latest)"),
) -> dict[str, Any]:
    """
    Full multi-metric profile for a player in a given season.
    Returns all RAPM metrics, percentile ranks, position, and TOI.
    """
    con = _connect()
    try:
        # Resolve season
        if season is None:
            row = con.execute(
                "SELECT MAX(season) FROM apm_results WHERE player_id = ?",
                [int(player_id)],
            ).fetchone()
            if not row or not row[0]:
                raise HTTPException(status_code=404, detail="No data for this player")
            season = str(row[0])

        # Get player info
        player_data: dict[str, Any] = {"player_id": int(player_id)}
        if _players_table_exists(con):
            df = con.execute(
                "SELECT player_id, first_name, last_name, full_name, position "
                "FROM players WHERE player_id = ?",
                [int(player_id)],
            ).df()
            if not df.empty:
                player_data = df.iloc[0].to_dict()

        if "full_name" not in player_data or not player_data.get("full_name"):
            full = _get_player_name(con, player_id)
            if full:
                player_data["full_name"] = full

        # Get all RAPM metrics for this player+season
        metrics_df = con.execute(
            """
            SELECT metric_name, value, games_count, toi_seconds, events_count
            FROM apm_results
            WHERE player_id = ? AND season = ?
            ORDER BY metric_name
            """,
            [int(player_id), season],
        ).df()

        metrics_list = metrics_df.to_dict(orient="records") if not metrics_df.empty else []

        # Get percentile ranks: for each metric, what percentile is this player in?
        percentiles: dict[str, Any] = {}
        for m in metrics_list:
            mn = m["metric_name"]
            pct_row = con.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE value <= ?) * 100.0 / COUNT(*) AS percentile,
                    COUNT(*) as total_players
                FROM apm_results
                WHERE metric_name = ? AND season = ?
                """,
                [m["value"], mn, season],
            ).fetchone()
            if pct_row:
                percentiles[mn] = {
                    "percentile": round(float(pct_row[0]), 1),
                    "total_players": int(pct_row[1]),
                }

        # Get career history (all seasons for corsi_rapm_5v5)
        career_df = con.execute(
            """
            SELECT season, metric_name, value
            FROM apm_results
            WHERE player_id = ? AND metric_name IN (
                'corsi_rapm_5v5', 'xg_rapm_5v5', 'goals_rapm_5v5',
                'corsi_off_rapm_5v5', 'corsi_def_rapm_5v5',
                'xg_off_rapm_5v5', 'xg_def_rapm_5v5'
            )
            ORDER BY season, metric_name
            """,
            [int(player_id)],
        ).df()
        career = career_df.to_dict(orient="records") if not career_df.empty else []

        return {
            "player": player_data,
            "season": season,
            "metrics": metrics_list,
            "percentiles": percentiles,
            "career": career,
        }
    finally:
        con.close()


@app.get("/api/stats/overview")
def stats_overview() -> dict[str, Any]:
    """
    Dashboard overview stats for the homepage.
    """
    con = _connect()
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

        seasons = []
        total_players = 0
        total_metrics = 0
        total_games = 0
        latest_season = None

        if "apm_results" in tables:
            row = con.execute("SELECT COUNT(DISTINCT season) FROM apm_results").fetchone()
            seasons_count = int(row[0]) if row else 0

            row = con.execute("SELECT COUNT(DISTINCT player_id) FROM apm_results").fetchone()
            total_players = int(row[0]) if row else 0

            row = con.execute("SELECT COUNT(DISTINCT metric_name) FROM apm_results").fetchone()
            total_metrics = int(row[0]) if row else 0

            rows = con.execute("SELECT DISTINCT season FROM apm_results ORDER BY season").fetchall()
            seasons = [r[0] for r in rows]
            latest_season = seasons[-1] if seasons else None

        if "games" in tables:
            row = con.execute("SELECT COUNT(*) FROM games").fetchone()
            total_games = int(row[0]) if row else 0

        # Top 5 players by latest season corsi_rapm_5v5
        top_players = []
        if latest_season and "apm_results" in tables:
            df = con.execute(
                """
                SELECT a.player_id, a.value, p.full_name
                FROM apm_results a
                LEFT JOIN players p ON a.player_id = p.player_id
                WHERE a.metric_name = 'corsi_rapm_5v5' AND a.season = ?
                ORDER BY a.value DESC
                LIMIT 5
                """,
                [latest_season],
            ).df()
            top_players = df.to_dict(orient="records") if not df.empty else []

        return {
            "seasons": seasons,
            "seasons_count": len(seasons),
            "total_players": total_players,
            "total_metrics": total_metrics,
            "total_games": total_games,
            "latest_season": latest_season,
            "top_players": top_players,
        }
    finally:
        con.close()


@app.get("/api/explanations/team/{team_id}")
def explain_team(team_id: int) -> dict[str, Any]:
    """
    Generate team-level explanations (placeholder for future implementation).

    Would analyze team composition, skill distribution, and strategic strengths.
    """
    return {
        "team_id": team_id,
        "status": "not_implemented",
        "message": "Team-level explanations require additional team analytics aggregation. Currently available: individual player explanations.",
        "suggestion": "Use /api/explanations/player/{player_id} for individual player insights."
    }

