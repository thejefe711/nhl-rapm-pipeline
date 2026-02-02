import numpy as np
from typing import Dict, Any, List
from .common import ValidationResult

EXPECTED_RANGES = {
    'goals_rapm_off': {'min': -1.0, 'max': 1.0, 'mean_range': (-0.1, 0.1)},
    'goals_rapm_def': {'min': -1.0, 'max': 1.0, 'mean_range': (-0.1, 0.1)},
    'xg_rapm_off': {'min': -2.0, 'max': 2.0, 'mean_range': (-0.1, 0.1)},
    'xg_rapm_def': {'min': -2.0, 'max': 2.0, 'mean_range': (-0.1, 0.1)},
    'corsi_rapm_off': {'min': -15.0, 'max': 15.0, 'mean_range': (-0.5, 0.5)},
    'corsi_rapm_def': {'min': -15.0, 'max': 15.0, 'mean_range': (-0.5, 0.5)},
    'xg_per_shot': {'min': 0.01, 'max': 0.95, 'mean_range': (0.06, 0.10)},
    'toi_minutes': {'min': 0, 'max': 30, 'mean_range': (12, 18)},
}

class OutputValidator:
    def validate_metric(self, metric_name: str, values: np.ndarray) -> ValidationResult:
        expected = EXPECTED_RANGES.get(metric_name)
        if not expected:
            return ValidationResult(
                check=f"range_check_{metric_name}",
                passed=True, 
                details="No range defined",
                severity="INFO"
            )
        
        issues = []
        
        if values.size == 0:
             return ValidationResult(
                check=f"range_check_{metric_name}",
                passed=False, 
                details="No values to validate",
                severity="WARNING"
            )

        if values.min() < expected['min']:
            issues.append(f"Min {values.min():.3f} below expected {expected['min']}")
        if values.max() > expected['max']:
            issues.append(f"Max {values.max():.3f} above expected {expected['max']}")
        
        mean_val = values.mean()
        if not (expected['mean_range'][0] <= mean_val <= expected['mean_range'][1]):
            issues.append(f"Mean {mean_val:.3f} outside expected range {expected['mean_range']}")
        
        return ValidationResult(
            check=f"range_check_{metric_name}",
            passed=len(issues) == 0, 
            details="; ".join(issues) if issues else "All checks passed",
            severity="ERROR" if issues else "PASS"
        )
