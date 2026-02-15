import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
df = con.execute('SELECT * FROM apm_results').fetchdf()
con.close()

drawn = df[df['metric_name'] == 'penalties_drawn_rapm_5v5'][['season', 'player_id', 'value']].rename(columns={'value': 'drawn'})
taken = df[df['metric_name'] == 'penalties_taken_rapm_5v5'][['season', 'player_id', 'value']].rename(columns={'value': 'taken'})
merged = drawn.merge(taken, on=['season', 'player_id'])
merged['sum'] = merged['drawn'] + merged['taken']
merged['abs_sum'] = np.abs(merged['sum'])

print(f"Total player-seasons checked: {len(merged)}")
print(f"Max |drawn + taken|: {merged['abs_sum'].max():.10f}")
print(f"Mean |drawn + taken|: {merged['abs_sum'].mean():.10f}")
print(f"Exactly zero: {(merged['abs_sum'] == 0).sum()} / {len(merged)}")
print(f"All within 1e-10: {(merged['abs_sum'] < 1e-10).all()}")
print()

worst = merged.nlargest(5, 'abs_sum')
print("Top 5 largest |drawn + taken|:")
for _, r in worst.iterrows():
    print(f"  {r['season']} player {int(r['player_id'])}: drawn={r['drawn']:+.8f} taken={r['taken']:+.8f} sum={r['sum']:+.2e}")
