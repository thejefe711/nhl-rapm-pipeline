import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any
from .common import ValidationReport, ValidationResult

class HistoricalConsistencyValidator:
    def validate_year_over_year(self, current: pd.DataFrame, previous: pd.DataFrame) -> ValidationReport:
        # Merge on player_id
        merged = current.merge(previous, on='player_id', suffixes=('_curr', '_prev'))
        
        results = []
        
        # Year-over-year correlation for stable metrics
        for metric in ['xg_rapm_off', 'corsi_rapm_off']:
            if f'{metric}_curr' in merged.columns and f'{metric}_prev' in merged.columns:
                r = merged[f'{metric}_curr'].corr(merged[f'{metric}_prev'])
                results.append(ValidationResult(
                    check=f"{metric}_yoy_correlation",
                    passed=r >= 0.4,
                    details=f"YoY correlation for {metric}: {r:.3f} (expected >= 0.4)",
                    severity="WARNING"
                ))
        
        # Distribution shift detection (KS test)
        for metric in ['xg_rapm_off', 'goals_rapm_off']:
            if metric in current.columns and metric in previous.columns:
                ks_stat, p_value = ks_2samp(
                    current[metric].dropna(),
                    previous[metric].dropna()
                )
                results.append(ValidationResult(
                    check=f"{metric}_distribution_shift",
                    passed=p_value > 0.01,
                    details=f"KS test p-value for {metric}: {p_value:.4f} (expected > 0.01)",
                    severity="WARNING"
                ))
        
        return ValidationReport(results)
