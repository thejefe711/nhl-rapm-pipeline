import duckdb

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
df = con.execute("SELECT * FROM apm_results").fetchdf()
con.close()

print("=== SEASON SUMMARY ===")
summary = df.groupby('season').agg(
    n_metrics=('metric_name', 'nunique'),
    total_rows=('player_id', 'count'),
    n_players=('player_id', 'nunique')
).reset_index()
print(summary.to_string(index=False))
print()

for season in sorted(df['season'].unique()):
    sdf = df[df['season'] == season]
    metrics = sorted(sdf['metric_name'].unique())
    print(f"{season}: {len(metrics)} metrics, {sdf['player_id'].nunique()} players")
    for m in metrics:
        count = len(sdf[sdf['metric_name'] == m])
        print(f"  - {m} ({count} players)")
    print()
