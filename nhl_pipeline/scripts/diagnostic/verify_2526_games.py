import duckdb
con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
res = con.execute("SELECT COUNT(DISTINCT game_id) FROM events WHERE LEFT(CAST(game_id AS VARCHAR), 4) = '2025'").fetchone()[0]
print(f"2025-2026 Games: {res}")
con.close()
