import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
df = con.execute('SELECT * FROM apm_results').fetchdf()
con.close()

seasons = sorted(df['season'].unique())

for pid, name in [(8478402, 'Connor McDavid'), (8482093, 'Seth Jarvis')]:
    pdf = df[df['player_id'] == pid]
    metrics = sorted(pdf['metric_name'].unique())
    player_seasons = sorted(pdf['season'].unique())
    print(f'\n{"="*100}')
    print(f'  {name} — {len(metrics)} metrics, {len(player_seasons)} seasons')
    print(f'{"="*100}')
    header = f'{"Metric":<45s}'
    for s in player_seasons:
        header += f' | {s[-4:-2]}-{s[-2:]}'
    print(header)
    print('-' * (45 + len(player_seasons) * 8))
    for m in metrics:
        mdf = pdf[pdf['metric_name'] == m]
        line = f'{m:<45s}'
        for s in player_seasons:
            row = mdf[mdf['season'] == s]
            if row.empty:
                line += ' |      '
            else:
                v = row.iloc[0]['value']
                line += f' | {v:+5.2f}'
        print(line)
    print()
