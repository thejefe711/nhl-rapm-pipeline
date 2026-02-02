import pandas as pd
from typing import List
from .common import ValidationResult

class CrossMetricValidator:
    def validate_consistency(self, player_stats: pd.DataFrame) -> List[ValidationResult]:
        results = []
        
        # xG RAPM should correlate with Goals RAPM (r > 0.3)
        if 'xg_rapm_off' in player_stats.columns and 'goals_rapm_off' in player_stats.columns:
            r = player_stats['xg_rapm_off'].corr(player_stats['goals_rapm_off'])
            results.append(ValidationResult(
                check="correlation_xg_goals_rapm_off",
                passed=r > 0.3,
                details=f"xG vs Goals RAPM correlation: {r:.3f} (expected > 0.3)",
                severity="WARNING"
            ))
        
        # Corsi RAPM should correlate with xG RAPM (r > 0.5)
        if 'corsi_rapm_off' in player_stats.columns and 'xg_rapm_off' in player_stats.columns:
            r = player_stats['corsi_rapm_off'].corr(player_stats['xg_rapm_off'])
            results.append(ValidationResult(
                check="correlation_corsi_xg_rapm_off",
                passed=r > 0.5,
                details=f"Corsi vs xG RAPM correlation: {r:.3f} (expected > 0.5)",
                severity="WARNING"
            ))
        
        # Offensive + Defensive impact should not be perfectly correlated
        if 'xg_rapm_off' in player_stats.columns and 'xg_rapm_def' in player_stats.columns:
            r = player_stats['xg_rapm_off'].corr(player_stats['xg_rapm_def'])
            results.append(ValidationResult(
                check="correlation_off_def_rapm",
                passed=abs(r) < 0.9,
                details=f"Off/Def correlation: {r:.3f} (expected abs < 0.9)",
                severity="WARNING"
            ))
        
        # Sum of all impacts should be near zero (zero-sum game)
        if 'xg_rapm_off' in player_stats.columns and 'xg_rapm_def' in player_stats.columns and 'toi' in player_stats.columns:
            total_off = (player_stats['xg_rapm_off'] * player_stats['toi']).sum()
            total_def = (player_stats['xg_rapm_def'] * player_stats['toi']).sum()
            net = total_off - total_def
            # This check is tricky because RAPM is zero-sum per stint, but weighted by TOI it should be close
            # Threshold of 100 is arbitrary from user request
            results.append(ValidationResult(
                check="zero_sum_impact",
                passed=abs(net) < 100,
                details=f"Net league impact: {net:.1f} (expected abs < 100)",
                severity="WARNING"
            ))
        
        return results
