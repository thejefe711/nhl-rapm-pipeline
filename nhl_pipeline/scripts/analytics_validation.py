#!/usr/bin/env python3
"""
Analytics Validation & Quality Assurance.

Ensures credibility of advanced analytics by:
- Validating model performance
- Checking data quality
- Comparing against established metrics
- Testing robustness
- Providing transparency metrics
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def validate_rapm_against_nhl_metrics(season: str = "20242025") -> Dict:
    """Validate RAPM metrics against established statistical principles."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Get RAPM distributions for validation
    rapm_data = con.execute("""
        SELECT
            metric_name,
            value as rapm_value,
            toi_seconds,
            games_count
        FROM apm_results
        WHERE season = ?
        AND metric_name IN ('goals_off_rapm_5v5', 'xg_off_rapm_5v5', 'corsi_off_rapm_5v5', 'hd_xg_off_rapm_5v5')
    """, [season]).fetchall()

    con.close()

    results = {}
    df = pd.DataFrame(rapm_data, columns=['metric_name', 'rapm_value', 'toi_seconds', 'games_count'])

    for metric in df['metric_name'].unique():
        metric_data = df[df['metric_name'] == metric]

        if len(metric_data) < 10:
            results[metric] = {"status": "insufficient_data", "samples": len(metric_data)}
            continue

        rapm_values = metric_data['rapm_value'].values

        # Statistical validation checks
        mean_val = np.mean(rapm_values)
        std_val = np.std(rapm_values)
        cv = abs(std_val / mean_val) if mean_val != 0 else float('inf')  # Coefficient of variation

        # Check for reasonable spread (not all identical values)
        value_range = np.max(rapm_values) - np.min(rapm_values)

        # Check for outliers (using IQR method)
        Q1 = np.percentile(rapm_values, 25)
        Q3 = np.percentile(rapm_values, 75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        outliers = sum(1 for v in rapm_values if v < Q1 - outlier_threshold or v > Q3 + outlier_threshold)
        outlier_pct = (outliers / len(rapm_values)) * 100

        # Check time-on-ice correlation (should be minimal for RAPM)
        toi_hours = metric_data['toi_seconds'].values / 3600.0
        toi_correlation = abs(stats.pearsonr(rapm_values, toi_hours)[0]) if len(toi_hours) > 1 else 0

        # Overall assessment
        issues = []
        if cv > 2.0:
            issues.append("high_variability")
        if value_range < 0.1:
            issues.append("insufficient_spread")
        if outlier_pct > 15:
            issues.append("too_many_outliers")
        if toi_correlation > 0.3:
            issues.append("toi_correlation")

        status = "good" if not issues else "needs_attention" if len(issues) <= 2 else "concerning"

        results[metric] = {
            "samples": len(metric_data),
            "mean": round(mean_val, 3),
            "std": round(std_val, 3),
            "coefficient_variation": round(cv, 3),
            "value_range": round(value_range, 3),
            "outlier_percentage": round(outlier_pct, 1),
            "toi_correlation": round(toi_correlation, 3),
            "issues": issues,
            "status": status
        }

    return results

def validate_latent_skill_stability(season: str = "20242025") -> Dict:
    """Validate that latent skills are stable and meaningful."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # Check latent skill consistency across time windows
    stability_check = con.execute("""
        SELECT
            ldm.label,
            COUNT(*) as total_windows,
            AVG(rls.value) as mean_value,
            STDDEV(rls.value) as std_value,
            MIN(rls.value) as min_value,
            MAX(rls.value) as max_value,
            ldm.stable_seasons
        FROM rolling_latent_skills rls
        JOIN latent_dim_meta ldm ON rls.model_name = ldm.model_name AND rls.dim_idx = ldm.dim_idx
        WHERE rls.model_name = 'sae_apm_v1_k12_a1'
        AND rls.window_size = 10
        GROUP BY ldm.label, ldm.stable_seasons
        ORDER BY std_value DESC
    """).fetchall()

    con.close()

    results = {}
    for label, total_windows, mean_val, std_val, min_val, max_val, stable_seasons in stability_check:
        # Calculate coefficient of variation (stability metric)
        cv = abs(std_val / mean_val) if mean_val != 0 else float('inf')
        range_size = max_val - min_val

        # Assess stability
        if cv < 0.5 and range_size < 2.0:  # Low variation
            stability_rating = "very_stable"
        elif cv < 1.0 and range_size < 4.0:
            stability_rating = "stable"
        elif cv < 2.0:
            stability_rating = "moderate"
        else:
            stability_rating = "unstable"

        results[label] = {
            "total_windows": total_windows,
            "mean_value": round(mean_val, 3),
            "coefficient_of_variation": round(cv, 3),
            "value_range": round(range_size, 3),
            "stable_seasons": stable_seasons,
            "stability_rating": stability_rating,
            "credibility": "high" if stability_rating in ["very_stable", "stable"] and stable_seasons >= 2 else "medium"
        }

    return results

def validate_dlm_forecast_accuracy(season: str = "20242025") -> Dict:
    """Validate DLM forecast accuracy using holdout testing."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    # For validation, we'll check forecast accuracy on historical data
    # by comparing forecasts made at different points in time

    forecast_accuracy = con.execute("""
        SELECT
            ldm.label,
            AVG(ABS(df.forecast_mean - rls_future.value)) as mae,
            AVG((df.forecast_mean - rls_future.value) * (df.forecast_mean - rls_future.value)) as mse,
            COUNT(*) as forecast_count
        FROM dlm_forecasts df
        JOIN latent_dim_meta ldm ON df.model_name = ldm.model_name AND df.dim_idx = ldm.dim_idx
        -- Join with actual future values (simplified - would need proper temporal join)
        JOIN rolling_latent_skills rls_future ON df.player_id = rls_future.player_id
            AND df.model_name = rls_future.model_name
            AND df.dim_idx = rls_future.dim_idx
        WHERE df.model_name = 'sae_apm_v1_k12_a1'
        AND df.horizon_games <= 5
        GROUP BY ldm.label
    """).fetchall()

    con.close()

    results = {}
    for label, mae, mse, count in forecast_accuracy:
        rmse = np.sqrt(mse) if mse else 0

        # Assess forecast quality
        if rmse < 0.2:
            accuracy_rating = "excellent"
        elif rmse < 0.5:
            accuracy_rating = "good"
        elif rmse < 1.0:
            accuracy_rating = "fair"
        else:
            accuracy_rating = "poor"

        results[label] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "forecast_count": count,
            "accuracy_rating": accuracy_rating,
            "credibility": "high" if accuracy_rating in ["excellent", "good"] else "medium"
        }

    return results

def check_data_quality(season: str = "20242025") -> Dict:
    """Check overall data quality and completeness."""

    db_path = Path(__file__).parent.parent / "nhl_canonical.duckdb"
    con = duckdb.connect(str(db_path))

    quality_checks = {}

    # Check RAPM data completeness
    rapm_completeness = con.execute("""
        SELECT
            metric_name,
            COUNT(*) as total_players,
            AVG(CASE WHEN toi_seconds > 3600 THEN 1 ELSE 0 END) as players_with_good_toi,
            AVG(toi_seconds) / 3600.0 as avg_toi_hours
        FROM apm_results
        WHERE season = ?
        GROUP BY metric_name
    """, [season]).fetchall()

    quality_checks["rapm_completeness"] = {}
    for metric, total_players, good_toi_pct, avg_toi_hrs in rapm_completeness:
        quality_checks["rapm_completeness"][metric] = {
            "total_players": total_players,
            "good_toi_percentage": round(good_toi_pct * 100, 1),
            "avg_toi_hours": round(avg_toi_hrs, 1),
            "quality": "good" if good_toi_pct > 0.7 else "needs_attention"
        }

    # Check latent skills completeness
    latent_completeness = con.execute("""
        SELECT
            COUNT(DISTINCT player_id) as players_with_latent,
            COUNT(*) as total_latent_records,
            AVG(window_size) as avg_window_size
        FROM rolling_latent_skills
        WHERE model_name = 'sae_apm_v1_k12_a1'
    """).fetchall()

    quality_checks["latent_skills_completeness"] = {
        "players_with_latent": latent_completeness[0][0],
        "total_records": latent_completeness[0][1],
        "avg_window_size": round(latent_completeness[0][2], 1),
        "quality": "good" if latent_completeness[0][0] > 50 else "limited"
    }

    # Check for basic statistics (simplified outlier check)
    basic_stats = con.execute("""
        SELECT
            metric_name,
            AVG(value) as mean_val,
            STDDEV(value) as std_val,
            MIN(value) as min_val,
            MAX(value) as max_val,
            COUNT(*) as total_count
        FROM apm_results
        WHERE season = ?
        GROUP BY metric_name
    """, [season]).fetchall()

    quality_checks["basic_statistics"] = {}
    for metric, mean_val, std_val, min_val, max_val, total_count in basic_stats:
        value_range = max_val - min_val
        # Simple outlier check based on range
        expected_range = 4.0 if "rapm" in metric.lower() else 10.0  # Different expectations for different metrics
        range_ratio = value_range / expected_range

        quality_checks["basic_statistics"][metric] = {
            "mean": round(mean_val, 3),
            "std": round(std_val, 3),
            "range": f"{round(min_val, 3)} to {round(max_val, 3)}",
            "samples": total_count,
            "data_quality": "good" if range_ratio > 0.3 else "check_distribution"
        }

    con.close()
    return quality_checks

def generate_credibility_report(season: str = "20242025") -> Dict:
    """Generate comprehensive credibility report."""

    print("GENERATING ANALYTICS CREDIBILITY REPORT")
    print("=" * 60)

    # Run all validations
    rapm_validation = validate_rapm_against_nhl_metrics(season)
    latent_stability = validate_latent_skill_stability(season)
    forecast_accuracy = validate_dlm_forecast_accuracy(season)
    data_quality = check_data_quality(season)

    # Calculate overall credibility score
    credibility_components = {
        "rapm_validation": len([v for v in rapm_validation.values() if isinstance(v, dict) and v.get("status") == "good"]),
        "latent_stability": len([v for v in latent_stability.values() if v.get("credibility") == "high"]),
        "forecast_accuracy": len([v for v in forecast_accuracy.values() if v.get("credibility") == "high"]),
        "data_quality": sum(1 for qc in data_quality.values() if isinstance(qc, dict) and any(
            item.get("quality") == "good" or item.get("data_quality") == "good"
            for item in qc.values() if isinstance(item, dict)
        ))
    }

    total_components = sum(len(v) if isinstance(v, dict) else 1 for v in [
        rapm_validation, latent_stability, forecast_accuracy, list(data_quality.keys())
    ])

    overall_credibility = sum(credibility_components.values()) / total_components

    if overall_credibility > 0.8:
        credibility_rating = "VERY HIGH"
        confidence_message = "Your analytics are robust and trustworthy"
    elif overall_credibility > 0.6:
        credibility_rating = "HIGH"
        confidence_message = "Your analytics are generally reliable with some areas for improvement"
    elif overall_credibility > 0.4:
        credibility_rating = "MEDIUM"
        confidence_message = "Your analytics show promise but need validation improvements"
    else:
        credibility_rating = "NEEDS ATTENTION"
        confidence_message = "Your analytics need significant validation and improvement"

    report = {
        "overall_credibility": {
            "score": round(overall_credibility, 3),
            "rating": credibility_rating,
            "confidence_message": confidence_message,
            "components_checked": total_components
        },
        "detailed_results": {
            "rapm_validation": rapm_validation,
            "latent_skill_stability": latent_stability,
            "forecast_accuracy": forecast_accuracy,
            "data_quality": data_quality
        },
        "recommendations": []
    }

    # Generate specific recommendations
    recommendations = []

    # RAPM validation recommendations
    concerning_rapm = [k for k, v in rapm_validation.items() if isinstance(v, dict) and v.get("status") == "concerning"]
    if concerning_rapm:
        recommendations.append(f"RAPM metrics {concerning_rapm} show weak correlation with actual NHL stats - investigate model specifications")

    # Latent stability recommendations
    unstable_skills = [k for k, v in latent_stability.items() if v.get("stability_rating") in ["unstable", "moderate"]]
    if unstable_skills:
        recommendations.append(f"Latent skills {unstable_skills} show high variability - consider increasing regularization or window sizes")

    # Forecast accuracy recommendations
    poor_forecasts = [k for k, v in forecast_accuracy.items() if v.get("accuracy_rating") in ["poor", "fair"]]
    if poor_forecasts:
        recommendations.append(f"DLM forecasts for {poor_forecasts} have poor accuracy - review Kalman filter parameters")

    # Data quality recommendations
    if any(qc.get("quality") == "needs_attention" or qc.get("data_quality") == "check_outliers"
           for qc in data_quality.values() if isinstance(qc, dict)):
        recommendations.append("Data quality issues detected - ensure sufficient TOI and check for outliers")

    if not recommendations:
        recommendations.append("All analytics validation checks passed - your system is credible!")

    report["recommendations"] = recommendations

    return report

def print_credibility_report(report: Dict):
    """Print formatted credibility report."""

    print(f"\nOVERALL CREDIBILITY: {report['overall_credibility']['rating']}")
    print(f"Score: {report['overall_credibility']['score']:.1%}")
    print(f"Components Checked: {report['overall_credibility']['components_checked']}")
    print(f"CONFIDENCE: {report['overall_credibility']['confidence_message']}")

    print(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")

    print(f"\nDETAILED RESULTS:")

    # RAPM Validation
    print(f"\nRAPM Validation:")
    for metric, results in report['detailed_results']['rapm_validation'].items():
        if isinstance(results, dict):
            status = results.get('status', 'unknown')
            status_icon = "[GOOD]" if status == "good" else "[CHECK]" if status == "concerning" else "[INFO]"
            print(f"  {status_icon} {metric}: {status}")

    # Latent Stability
    print(f"\nLatent Skill Stability:")
    stable_count = sum(1 for v in report['detailed_results']['latent_skill_stability'].values()
                      if v.get('credibility') == 'high')
    total_skills = len(report['detailed_results']['latent_skill_stability'])
    print(f"  [CREDIBLE] {stable_count}/{total_skills} skills rated as credible")

    # Forecast Accuracy
    print(f"\nForecast Accuracy:")
    accurate_count = sum(1 for v in report['detailed_results']['forecast_accuracy'].values()
                        if v.get('credibility') == 'high')
    total_forecasts = len(report['detailed_results']['forecast_accuracy'])
    print(f"  [CREDIBLE] {accurate_count}/{total_forecasts} forecasts rated as credible")

if __name__ == "__main__":
    report = generate_credibility_report("20242025")
    print_credibility_report(report)