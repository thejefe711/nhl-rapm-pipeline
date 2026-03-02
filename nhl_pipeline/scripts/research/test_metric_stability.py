"""
test_metric_stability.py
------------------------
Year-to-year correlation stability analysis for ALL metrics stored in apm_results.

Rather than maintaining a hardcoded list, this script introspects the database at
runtime and tests every distinct metric_name found in apm_results.

Usage:
    python test_metric_stability.py
    python test_metric_stability.py --min-toi 1800 --output stability_report.json
    python test_metric_stability.py --plot                    # requires matplotlib
    python test_metric_stability.py --filter rapm             # metrics containing 'rapm'
    python test_metric_stability.py --exclude-seasons 2020    # skip COVID year
    python test_metric_stability.py --strict                  # CI: fail on INSUFFICIENT_DATA too

Output:
    - Console table: metric | r | r^2 | n | lam | signal_grade
    - Optional JSON report
    - Optional scatter plots per metric
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Resolves to: nhl_pipeline/nhl_canonical.duckdb  (same DB used by compute_corsi_apm.py)
# This file lives at: nhl_pipeline/scripts/research/test_metric_stability.py
# parents[2] = nhl_pipeline/
DB_PATH = Path(__file__).resolve().parents[2] / "nhl_canonical.duckdb"

# Signal grade thresholds — MUST remain sorted descending by threshold value.
# The grading loop relies on this ordering; an assertion enforces it at import time.
SIGNAL_GRADES = [
    (0.60, "STRONG",   "Ship with confidence"),
    (0.45, "MODERATE", "Ship with reliability warning"),
    (0.30, "WEAK",     "Exploratory only -- do not feature on leaderboards"),
    (0.00, "NOISE",    "Do not ship -- likely overfitting or role artifact"),
]
assert all(
    SIGNAL_GRADES[i][0] >= SIGNAL_GRADES[i + 1][0]
    for i in range(len(SIGNAL_GRADES) - 1)
), "SIGNAL_GRADES must be sorted descending by threshold"

MIN_PLAYERS_FOR_VALID_TEST = 30  # below this, correlation estimate is too noisy
OUTLIER_QUANTILE = 0.01          # flag players outside [1st, 99th] percentile per pair


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

class PairBreakdown(TypedDict):
    season_n:  str
    season_n1: str
    r:         float
    n:         int


class StabilityResult(TypedDict):
    metric:           str
    r:                float | None
    r2:               float | None
    p_value:          float | None
    n_player_pairs:   int
    n_season_pairs:   int
    n_outliers:       int
    signal_grade:     str
    note:             str
    shrinkage_lambda: float | None
    pair_breakdown:   list[PairBreakdown]


def _empty_result(metric: str, note: str, n: int = 0) -> StabilityResult:
    return StabilityResult(
        metric=metric,
        r=None,
        r2=None,
        p_value=None,
        n_player_pairs=n,
        n_season_pairs=0,
        n_outliers=0,
        signal_grade="INSUFFICIENT_DATA",
        note=note,
        shrinkage_lambda=None,
        pair_breakdown=[],
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_metrics(
    con: duckdb.DuckDBPyConnection,
    name_filter: str | None = None,
    exclude_seasons: list[str] | None = None,
) -> list[str]:
    """
    Return all distinct metric_names in apm_results, optionally filtered.

    NOTE: metric_name should be indexed for performance on large tables.
    If this becomes slow, add: CREATE INDEX idx_apm_metric ON apm_results(metric_name)
    """
    where = ""
    if exclude_seasons:
        placeholders = ", ".join("?" * len(exclude_seasons))
        where = f"WHERE season NOT IN ({placeholders})"

    query = f"SELECT DISTINCT metric_name FROM apm_results {where} ORDER BY metric_name"
    params = exclude_seasons or []
    rows = con.execute(query, params).fetchall()
    metrics = [r[0] for r in rows]

    if name_filter:
        metrics = [m for m in metrics if name_filter.lower() in m.lower()]

    return metrics


def load_metric(
    con: duckdb.DuckDBPyConnection,
    metric: str,
    min_toi: int,
    exclude_seasons: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load all player-season rows for a single metric from apm_results.

    Returns: DataFrame with columns [player_id, season, value, toi_seconds]
    """
    season_filter = ""
    params: list = [metric, min_toi]

    if exclude_seasons:
        placeholders = ", ".join("?" * len(exclude_seasons))
        season_filter = f"AND season NOT IN ({placeholders})"
        params.extend(exclude_seasons)

    query = f"""
        SELECT
            player_id,
            season,
            value,
            toi_seconds
        FROM apm_results
        WHERE metric_name = ?
          AND toi_seconds >= ?
          AND value IS NOT NULL
          {season_filter}
        ORDER BY player_id, season
    """
    try:
        return con.execute(query, params).df()
    except Exception as e:
        print(f"[WARN] Could not load metric '{metric}': {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Core stability computation
# ---------------------------------------------------------------------------

def compute_yoy_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all consecutive year-over-year player pairs from a long-form DataFrame.

    Input columns:  [player_id, season, value, toi_seconds]
    Output columns: [player_id, season_n, season_n1, val_n, val_n1, toi_n, toi_n1, min_toi]
    """
    if df.empty or "value" not in df.columns:
        return pd.DataFrame()

    # Explicit copy -- caller's DataFrame is not modified
    df = df[["player_id", "season", "value", "toi_seconds"]].dropna(subset=["value"]).copy()
    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        return pd.DataFrame()

    consecutive_pairs = [(s, seasons[i + 1]) for i, s in enumerate(seasons[:-1])]

    pairs = []
    for s_n, s_n1 in consecutive_pairs:
        left = (
            df[df["season"] == s_n][["player_id", "value", "toi_seconds"]]
            .rename(columns={"value": "val_n", "toi_seconds": "toi_n"})
        )
        right = (
            df[df["season"] == s_n1][["player_id", "value", "toi_seconds"]]
            .rename(columns={"value": "val_n1", "toi_seconds": "toi_n1"})
        )
        merged = left.merge(right, on="player_id")
        merged["season_n"]  = s_n
        merged["season_n1"] = s_n1
        pairs.append(merged)

    if not pairs:
        return pd.DataFrame()

    result = pd.concat(pairs, ignore_index=True)
    # min_toi: the weaker season is the reliability constraint
    result["min_toi"] = result[["toi_n", "toi_n1"]].min(axis=1)
    return result


def _grade_r(r: float) -> tuple[str, str]:
    """Map a Pearson r value to (signal_grade, note)."""
    for threshold, grade, label in SIGNAL_GRADES:
        if r >= threshold:
            return grade, label
    return "NOISE", "Do not ship"


def _flag_outliers(pairs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Log a count of extreme outlier pairs (outside 1st/99th percentile on either axis).
    Does NOT remove them -- caller decides whether to clip.
    """
    mask = pd.Series(False, index=pairs.index)
    for col in ["val_n", "val_n1"]:
        q_low  = pairs[col].quantile(OUTLIER_QUANTILE)
        q_high = pairs[col].quantile(1 - OUTLIER_QUANTILE)
        mask |= (pairs[col] < q_low) | (pairs[col] > q_high)
    return pairs, int(mask.sum())


def shrinkage_lambda(r: float, median_min_toi: int) -> float:
    """
    Empirical Bayes shrinkage factor based on YoY stability and sample size.

    lambda = r * sqrt(min(median_min_toi, n_ref) / n_ref), capped at [0, 1].

    Uses median_min_toi (the median of the minimum TOI across both seasons in each
    player pair) rather than overall median TOI. This avoids overestimating reliability
    for players who qualified in one season but not the other.

    n_ref ~= 2000s (~33 min), the approximate single-season reliability threshold.
    """
    n_ref = 2000
    lam = r * np.sqrt(min(median_min_toi, n_ref) / n_ref)
    return float(np.clip(lam, 0.0, 1.0))


def stability_for_metric(df: pd.DataFrame, metric: str) -> StabilityResult:
    """Compute full stability profile for one metric across all YoY pairs."""
    pairs = compute_yoy_pairs(df)

    if pairs.empty or len(pairs) < MIN_PLAYERS_FOR_VALID_TEST:
        return _empty_result(
            metric,
            note=f"Need >= {MIN_PLAYERS_FOR_VALID_TEST} player-season pairs",
            n=len(pairs),
        )

    pairs, n_outliers = _flag_outliers(pairs)
    if n_outliers > 0:
        print(f"[WARN] {metric}: {n_outliers} outlier pair(s) detected (not removed)")

    r, p = stats.pearsonr(pairs["val_n"], pairs["val_n1"])
    r2   = r ** 2
    # scipy returns NaN for p when |r| is so close to 1.0 that the t-stat
    # overflows float64.  Treat as the smallest representable positive float.
    if np.isnan(p):
        p = 5e-324
    grade, label = _grade_r(r)

    # Per-season-pair breakdown (useful for spotting outlier seasons, e.g. COVID)
    pair_breakdown: list[PairBreakdown] = []
    for (s_n, s_n1), grp in pairs.groupby(["season_n", "season_n1"]):
        if len(grp) >= 10:
            r_pair, _ = stats.pearsonr(grp["val_n"], grp["val_n1"])
            pair_breakdown.append(
                PairBreakdown(season_n=s_n, season_n1=s_n1, r=round(r_pair, 3), n=len(grp))
            )

    # Shrinkage uses median of minimum per-pair TOI (weakest season is the constraint)
    median_min_toi = int(pairs["min_toi"].median())
    lam = round(shrinkage_lambda(r, median_min_toi), 3)

    return StabilityResult(
        metric=metric,
        r=round(r, 3),
        r2=round(r2, 3),
        p_value=round(p, 4),
        n_player_pairs=len(pairs),
        n_season_pairs=len(pair_breakdown),
        n_outliers=n_outliers,
        signal_grade=grade,
        note=label,
        shrinkage_lambda=lam,
        pair_breakdown=pair_breakdown,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

GRADE_COLOR = {
    "STRONG":            "\033[92m",   # green
    "MODERATE":          "\033[93m",   # yellow
    "WEAK":              "\033[33m",   # dark yellow
    "NOISE":             "\033[91m",   # red
    "INSUFFICIENT_DATA": "\033[90m",   # grey
}
RESET = "\033[0m"


def print_report(results: list[StabilityResult]) -> None:
    # r^2 uses ASCII caret -- avoids Unicode issues on Windows
    header = f"{'METRIC':<40} {'r':>6} {'r^2':>6} {'n':>6} {'lam':>6}  {'out':>4}  GRADE"
    print()
    print("=" * 86)
    print("  METRIC STABILITY REPORT -- Year-over-Year Correlation Analysis")
    print("=" * 86)
    print(header)
    print("-" * 86)

    for res in sorted(results, key=lambda x: (x["r"] or -99), reverse=True):
        r_str  = f"{res['r']:.3f}"  if res["r"]  is not None else "   N/A"
        r2_str = f"{res['r2']:.3f}" if res["r2"] is not None else "   N/A"
        n_str  = str(res["n_player_pairs"])
        lam    = f"{res['shrinkage_lambda']:.2f}" if res["shrinkage_lambda"] is not None else "  N/A"
        out    = str(res["n_outliers"]) if res["n_outliers"] > 0 else "-"
        grade  = res["signal_grade"]
        color  = GRADE_COLOR.get(grade, "")
        print(f"  {res['metric']:<38} {r_str:>6} {r2_str:>6} {n_str:>6} {lam:>6}  {out:>4}  {color}{grade}{RESET}")

    print("-" * 86)
    print()
    print("  Signal grades:")
    for _, g, l in SIGNAL_GRADES:
        color = GRADE_COLOR.get(g, "")
        print(f"    {color}{g:<12}{RESET}  {l}")
    print()
    print("  lam = Bayesian shrinkage factor (0=full regression to mean, 1=trust observed)")
    print("  out = outlier player-pairs flagged outside [1st, 99th] percentile")
    print()

    # Per-pair breakdown for metrics with season-level variance
    for res in results:
        if res.get("pair_breakdown") and len(res["pair_breakdown"]) > 1:
            print(f"  {res['metric']} -- per-season-pair breakdown:")
            for pb in res["pair_breakdown"]:
                print(f"    {pb['season_n']} -> {pb['season_n1']}:  r={pb['r']:.3f}  n={pb['n']}")
            print()

    # Launch summary
    shippable   = [r for r in results if r["signal_grade"] in ("STRONG", "MODERATE")]
    exploratory = [r for r in results if r["signal_grade"] == "WEAK"]
    blocked     = [r for r in results if r["signal_grade"] == "NOISE"]
    no_data     = [r for r in results if r["signal_grade"] == "INSUFFICIENT_DATA"]
    print("  LAUNCH SUMMARY")
    print(f"    READY:        {len(shippable):>3} metrics")
    print(f"    EXPLORATORY:  {len(exploratory):>3} metrics  (show with reliability warning)")
    print(f"    BLOCKED:      {len(blocked):>3} metrics  (do not ship)")
    print(f"    NO DATA:      {len(no_data):>3} metrics  (check schema / TOI filter)")
    print()


def save_report(results: list[StabilityResult], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"stability_results": list(results)}, f, indent=2)
    print(f"[INFO] Report saved to {out_path}")


# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------

def plot_scatters(metric_data: dict[str, pd.DataFrame], results: list[StabilityResult]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plots")
        return

    good_metrics = [r for r in results if r["r"] is not None and r["r"] >= 0.30]
    if not good_metrics:
        print("[INFO] No metrics with r >= 0.30 to plot")
        return

    ncols = 3
    nrows = (len(good_metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    grade_color = {
        "STRONG":   "#16a34a",
        "MODERATE": "#ca8a04",
        "WEAK":     "#ea580c",
        "NOISE":    "#dc2626",
    }

    for ax, res in zip(axes, good_metrics):
        metric = res["metric"]
        df = metric_data.get(metric, pd.DataFrame())
        # compute_yoy_pairs is called again here; pairs are not cached in results
        # to avoid bloating the JSON report with raw data
        pairs = compute_yoy_pairs(df)
        if pairs.empty:
            ax.set_visible(False)
            continue

        ax.scatter(pairs["val_n"], pairs["val_n1"], alpha=0.35, s=18, color="#2563eb")
        slope, intercept, *_ = stats.linregress(pairs["val_n"], pairs["val_n1"])
        x_range = np.linspace(pairs["val_n"].min(), pairs["val_n"].max(), 100)
        ax.plot(x_range, slope * x_range + intercept, color="#dc2626", linewidth=1.5)

        color = grade_color.get(res["signal_grade"], "#6b7280")
        ax.set_title(
            f"{metric}\nr={res['r']:.3f}  [{res['signal_grade']}]",
            color=color,
            fontsize=9,
        )
        ax.set_xlabel("Year N", fontsize=8)
        ax.set_ylabel("Year N+1", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(good_metrics):]:
        ax.set_visible(False)

    fig.suptitle("Metric Stability: Year-over-Year Correlations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("stability_plots.png", dpi=150, bbox_inches="tight")
    print("[INFO] Scatter plots saved to stability_plots.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Test year-over-year metric stability (all metrics)")
    parser.add_argument("--db",              default=str(DB_PATH), help="Path to DuckDB file")
    parser.add_argument("--min-toi",         default=1800, type=int, help="Minimum TOI seconds per season")
    parser.add_argument("--output",          default=None, help="Save JSON report to this path")
    parser.add_argument("--plot",            action="store_true", help="Generate scatter plots (requires matplotlib)")
    parser.add_argument("--filter",          default=None, help="Only test metrics whose name contains this string (case-insensitive)")
    parser.add_argument("--exclude-seasons", default=None, nargs="+", help="Seasons to exclude, e.g. --exclude-seasons 2020 2021")
    parser.add_argument("--strict",          action="store_true", help="Exit non-zero if any metric is INSUFFICIENT_DATA (for CI)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return 1

    exclude_seasons: list[str] = args.exclude_seasons or []

    print(f"[INFO] Connecting to {db_path}")

    # Context manager ensures connection is closed even on exception
    with duckdb.connect(str(db_path), read_only=True) as con:

        all_metrics = discover_metrics(con, name_filter=args.filter, exclude_seasons=exclude_seasons)
        if not all_metrics:
            filter_note = f" matching '{args.filter}'" if args.filter else ""
            print(f"[ERROR] No metrics found in apm_results{filter_note}.")
            return 1

        print(
            f"[INFO] Found {len(all_metrics)} metric(s) to test"
            + (f" (filter: {args.filter!r})" if args.filter else "")
            + (f" (excluding seasons: {exclude_seasons})" if exclude_seasons else "")
        )

        results: list[StabilityResult] = []
        metric_data: dict[str, pd.DataFrame] = {}  # preserved for optional plotting

        for i, metric in enumerate(all_metrics, 1):
            print(f"[{i:>3}/{len(all_metrics)}] {metric}", end=" ... ", flush=True)
            df = load_metric(con, metric, args.min_toi, exclude_seasons=exclude_seasons)

            if df.empty:
                print("no data")
                results.append(_empty_result(metric, note="No rows after TOI filter"))
                continue

            metric_data[metric] = df
            res = stability_for_metric(df, metric)
            results.append(res)
            print(f"r={res['r']}" if res["r"] is not None else "INSUFFICIENT_DATA")

    # con is closed here

    print_report(results)

    if args.output:
        save_report(results, args.output)

    if args.plot:
        plot_scatters(metric_data, results)

    # CI exit codes
    noise_metrics = [r["metric"] for r in results if r["signal_grade"] == "NOISE"]
    if noise_metrics:
        print(f"[WARN] Noise-grade metrics: {', '.join(noise_metrics)}")

    no_data_metrics = [r["metric"] for r in results if r["signal_grade"] == "INSUFFICIENT_DATA"]
    if no_data_metrics:
        print(f"[WARN] Insufficient-data metrics: {', '.join(no_data_metrics)}")

    if noise_metrics:
        return 1
    if args.strict and no_data_metrics:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
