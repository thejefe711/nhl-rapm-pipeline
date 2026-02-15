import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
# Fetch all penalty metrics (old and new names just in case, though old should be gone)
df = con.execute("SELECT * FROM apm_results WHERE metric_name LIKE '%penalties%'").fetchdf()
con.close()

print(f"Metrics found: {df['metric_name'].unique()}")

committed = df[df['metric_name'] == 'penalties_committed_rapm_5v5'][['season', 'player_id', 'value']].rename(columns={'value': 'committed'})
drawn = df[df['metric_name'] == 'penalties_drawn_rapm_5v5'][['season', 'player_id', 'value']].rename(columns={'value': 'drawn'})

if committed.empty or drawn.empty:
    print("Error: No data found for new metrics!")
    exit(1)

merged = committed.merge(drawn, on=['season', 'player_id'])
merged['sum'] = merged['committed'] + merged['drawn']
merged['corr'] = merged['committed'].corr(merged['drawn'])

print(f"\nTotal player-seasons: {len(merged)}")
print(f"Correlation (committed vs drawn): {merged['corr'].iloc[0]:.4f}")
print(f"Max |committed + drawn|: {merged['sum'].abs().max():.4f}")

# Check if they are still exact negatives
is_exact_neg = merged['sum'].abs().max() < 1e-9
print(f"Are they exact negatives? {is_exact_neg}")

if not is_exact_neg:
    print("\nSUCCESS: Metrics are now independent!")
    
    print("\nTop 5 Agitators (High Drawn RAPM):")
    top_drawn = merged.nlargest(5, 'drawn')
    for _, r in top_drawn.iterrows():
        print(f"  {r['season']} ID {int(r['player_id'])}: Drawn={r['drawn']:+.3f}  Committed={r['committed']:+.3f}")

    print("\nTop 5 Disciplined (Low Committed RAPM):")
    # Low committed means negative value (since +ve = takes more penalties)
    # Wait, ridge regression centers on 0. 
    # Positive coef = takes MORE penalties than average
    # Negative coef = takes FEWER penalties than average
    best_disc = merged.nsmallest(5, 'committed')
    for _, r in best_disc.iterrows():
        print(f"  {r['season']} ID {int(r['player_id'])}: Committed={r['committed']:+.3f}  Drawn={r['drawn']:+.3f}")
else:
    print("\nFAILURE: Metrics are still exact negatives of each other.")
