from nhl_pipeline.src.monitoring.alerting import AlertManager

def main():
    manager = AlertManager("nhl_pipeline/nhl_canonical.duckdb")
    print("Checking pipeline health...")
    try:
        manager.check_pipeline_health()
        print("Pipeline health check ran successfully.")
    except Exception as e:
        print(f"Pipeline health check failed: {e}")

    print("Checking anomalies...")
    try:
        manager.check_anomalies()
        print("Anomaly check ran successfully.")
    except Exception as e:
        print(f"Anomaly check failed: {e}")

if __name__ == "__main__":
    main()
