import duckdb

from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
con = duckdb.connect(str(ROOT / "nhl_canonical.duckdb"), read_only=True)
seasons = ['20252026', '20242025', '20232024', '20222023', '20212022', '20202021']

with open("status_output.txt", "w") as f:
    for season in seasons:
        res = con.execute(
            "SELECT COUNT(DISTINCT metric_name), MIN(games_count), MAX(games_count) FROM apm_results WHERE season = ?",
            [season]
        ).fetchone()
        metrics, min_g, max_g = res
        turnover = con.execute(
            "SELECT COUNT(*) FROM apm_results WHERE season = ? AND metric_name LIKE '%turnover%'",
            [season]
        ).fetchone()[0]
        status = "OK" if turnover > 0 and min_g == max_g and min_g > 53 else "NEEDS FIX"
        f.write(f"{season} | {metrics} metrics | games {min_g}-{max_g} | turnover:{turnover} | {status}\n")

con.close()
print("Done - see status_output.txt")
