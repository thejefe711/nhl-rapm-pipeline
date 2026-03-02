"""
RAPM Hyperparameter Search — MLflow + Optuna
=============================================

Searches over ridge alpha and min_toi, logging every trial to MLflow
so you can spot outlier configs in the UI.

Usage:
    python tune_rapm_hyperparams.py

Then open the MLflow dashboard:
    mlflow ui        (from the project root)
    → http://localhost:5000
"""

import subprocess
import sys
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import optuna

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent          # nhl_pipeline/
SCRIPT = ROOT / "scripts" / "core" / "compute_corsi_apm.py"
DB_PATH = ROOT / "nhl_canonical.duckdb"

# ── Search config ─────────────────────────────────────────────────────────────
EVAL_SEASON = "20252026"        # season to tune on
GAME_LIMIT = 200                # increased for better stability
N_TRIALS = 10                   # reduced for speed
TARGET_METRIC = "corsi_rapm_5v5"

# Good RAPM distributions: std ~3-6, almost no players with |value| > 15
OUTLIER_THRESHOLD = 15.0


# ── Fitness function ──────────────────────────────────────────────────────────

def _run_pipeline(alpha: float, min_toi: int) -> bool:
    """Run compute_corsi_apm.py with the given config. Returns True on success."""
    cmd = [
        sys.executable, str(SCRIPT),
        "--season", EVAL_SEASON,
        "--alphas", str(alpha),
        "--min-toi", str(min_toi),
        "--metrics", "corsi",
        "--limit", str(GAME_LIMIT),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT.parent))
    if result.returncode != 0:
        print(f"  ✗ Pipeline failed:\n{result.stderr[-800:]}")
        return False
    return True


def _read_results() -> np.ndarray:
    """Read the current RAPM values from DuckDB for the eval season."""
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        df = con.execute(
            "SELECT value FROM apm_results WHERE metric_name = ? AND season = ?",
            [TARGET_METRIC, EVAL_SEASON],
        ).fetchdf()
        con.close()
        return df["value"].dropna().values
    except Exception as e:
        print(f"  ✗ DB read failed: {e}")
        return np.array([])


def _fitness(values: np.ndarray) -> float:
    """
    Lower = better.
    Combines the spread (std) with the fraction of extreme outliers.
    Good RAPM: std ~3-6, outlier_frac near 0.
    """
    if len(values) < 10:
        return float("inf")
    std = float(np.std(values))
    outlier_frac = float(np.mean(np.abs(values) > OUTLIER_THRESHOLD))
    return std + 10.0 * outlier_frac


# ── Optuna objective ──────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    alpha = trial.suggest_float("alpha", 1e2, 1e6, log=True)
    min_toi = trial.suggest_int("min_toi", 300, 1800, step=100)

    print(f"\n── Trial {trial.number}: alpha={alpha:.1f}, min_toi={min_toi}s ──")

    with mlflow.start_run():
        mlflow.log_params({
            "alpha": alpha,
            "min_toi": min_toi,
            "season": EVAL_SEASON,
            "game_limit": GAME_LIMIT,
        })

        success = _run_pipeline(alpha, min_toi)
        if not success:
            mlflow.log_metric("fitness_score", 9999.0)
            mlflow.log_param("status", "pipeline_failed")
            return 9999.0

        values = _read_results()
        score = _fitness(values)

        # Log detailed metrics so you can explore in MLflow UI
        mlflow.log_metrics({
            "fitness_score":    score,
            "rapm_std":         float(np.std(values))         if len(values) else float("nan"),
            "rapm_mean":        float(np.mean(values))        if len(values) else float("nan"),
            "rapm_p95":         float(np.percentile(values, 95)) if len(values) else float("nan"),
            "rapm_p5":          float(np.percentile(values,  5)) if len(values) else float("nan"),
            "outlier_frac":     float(np.mean(np.abs(values) > OUTLIER_THRESHOLD)) if len(values) else float("nan"),
            "n_players":        float(len(values)),
        })
        mlflow.log_param("status", "ok")

        print(f"   score={score:.4f}  std={np.std(values):.2f}  n={len(values)}")

    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mlflow.set_experiment("rapm_hyperparam_search")

    # Suppress Optuna's per-trial INFO logs for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print("\n" + "=" * 60)
    print("BEST CONFIG FOUND")
    print("=" * 60)
    print(f"  alpha:        {best['alpha']:.2f}")
    print(f"  min_toi:      {best['min_toi']}s")
    print(f"  fitness_score: {study.best_value:.4f}")
    print()
    print("To use this config, run:")
    print(
        f"  python compute_corsi_apm.py "
        f"--alphas {best['alpha']:.1f} "
        f"--min-toi {best['min_toi']}"
    )
    print()
    print("View all trials:")
    print("  mlflow ui    → http://localhost:5000")


if __name__ == "__main__":
    main()
