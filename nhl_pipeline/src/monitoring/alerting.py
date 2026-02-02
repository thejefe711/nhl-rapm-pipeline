import duckdb
import requests
import os
from typing import List, Dict

class AlertManager:
    def __init__(self, db_path: str, webhook_url: str = None):
        self.db = duckdb.connect(db_path)
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")

    def check_pipeline_health(self):
        # Check for recent failures
        failures = self.db.execute("""
            SELECT stage, failures_7d 
            FROM v_pipeline_health 
            WHERE failures_7d > 0
        """).fetchall()
        
        for stage, count in failures:
            self.send_alert(f"Pipeline Alert: Stage '{stage}' has failed {count} times in the last 7 days.")

    def check_anomalies(self):
        # Check for statistical anomalies
        anomalies = self.db.execute("SELECT player_id, metric_name, z_score FROM v_anomalies").fetchall()
        
        if anomalies:
            msg = f"Data Quality Alert: Found {len(anomalies)} statistical anomalies (z-score > 3)."
            # List top 5
            for pid, metric, z in anomalies[:5]:
                msg += f"\n- Player {pid}, {metric}: z={z:.2f}"
            self.send_alert(msg)

    def send_alert(self, message: str):
        print(f"ALERT: {message}")
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json={"text": message})
            except Exception as e:
                print(f"Failed to send webhook: {e}")

if __name__ == "__main__":
    # Example usage
    manager = AlertManager("nhl_canonical.duckdb")
    # manager.check_pipeline_health() # Requires view to exist
    # manager.check_anomalies() # Requires view to exist
