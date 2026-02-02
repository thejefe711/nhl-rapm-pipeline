from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time

@dataclass
class MonitorConfig:
    history_db_path: str
    alert_webhook: Optional[str] = None

@dataclass
class HealthReport:
    overall_status: str
    stage_metrics: Dict[str, Any]
    alerts: List[str]
    recommendations: List[str]

class PipelineHealthMonitor:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.alerts: List[str] = []
        self.metrics: Dict[str, Any] = {}
        # In a real implementation, we'd load historical stats here
        self.historical_stats = {} 

    def check_execution_time(self, stage: str, duration: float):
        # Placeholder for historical lookup
        historical_avg = self.historical_stats.get(stage, {}).get('avg_duration', 1.0)
        
        if duration > historical_avg * 2:
            self.alert(f"Stage {stage} took {duration:.2f}s, 2x historical avg {historical_avg:.2f}s")
        
        self.metrics.setdefault(stage, {})['duration'] = duration

    def check_row_counts(self, stage: str, current: int, previous: int):
        delta_pct = (current - previous) / previous if previous > 0 else 0.0
        
        if delta_pct < -0.05:
            self.alert(f"Stage {stage} row count dropped {delta_pct:.1%}: {previous} → {current}")
            
        self.metrics.setdefault(stage, {})['row_count'] = current
        self.metrics.setdefault(stage, {})['row_delta_pct'] = delta_pct

    def alert(self, message: str):
        self.alerts.append(message)
        # Here we would send to webhook if configured

    def generate_health_report(self) -> HealthReport:
        status = "HEALTHY"
        if self.alerts:
            status = "DEGRADED"
        
        recommendations = []
        if status == "DEGRADED":
            recommendations.append("Check logs for stages with alerts.")
            
        return HealthReport(
            overall_status=status,
            stage_metrics=self.metrics,
            alerts=self.alerts,
            recommendations=recommendations
        )
