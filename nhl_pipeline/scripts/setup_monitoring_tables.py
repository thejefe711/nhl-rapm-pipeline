import duckdb

def main():
    conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
    
    # Create lineage_log table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS lineage_log (
        stage VARCHAR,
        timestamp TIMESTAMP,
        duration_seconds DOUBLE,
        status VARCHAR,
        details VARCHAR
    )
    """)
    
    # Create validation_results table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS validation_results (
        check_name VARCHAR,
        check_time TIMESTAMP,
        passed BOOLEAN,
        details VARCHAR,
        severity VARCHAR
    )
    """)
    
    print("Monitoring tables created/verified.")

if __name__ == "__main__":
    main()
