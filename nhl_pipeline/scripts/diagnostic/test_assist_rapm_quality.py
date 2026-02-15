"""
Test-Driven Validation of RAPM Assist Metrics

This script validates the quality and reliability of assist-related RAPM metrics.
Run with: python -m pytest test_assist_rapm_quality.py -v
"""

import duckdb
import pytest
from pathlib import Path


DB_PATH = Path(__file__).parent / "nhl_canonical.duckdb"


@pytest.fixture
def con():
    """Database connection fixture."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    yield conn
    conn.close()


class TestAssistRAPMDataQuality:
    """Tests to validate assist RAPM metric quality."""
    
    def test_elite_playmakers_rank_highly_in_primary_assists(self, con):
        """
        TEST: Known elite playmakers should rank in top 20 for primary assist RAPM.
        
        Elite playmakers (by historical assist totals):
        - Connor McDavid (8478402)
        - Nathan MacKinnon (8477492)
        - Nikita Kucherov (8476453)
        - Leon Draisaitl (8477934)
        """
        elite_playmakers = [8478402, 8477492, 8476453, 8477934]
        
        # Get top 50 primary assist RAPM for latest season
        df = con.execute("""
            SELECT player_id, value,
                   ROW_NUMBER() OVER (ORDER BY value DESC) as rank
            FROM apm_results
            WHERE metric_name = 'primary_assist_rapm_5v5'
              AND season = (SELECT MAX(season) FROM apm_results)
            ORDER BY value DESC
            LIMIT 100
        """).df()
        
        top_50_ids = set(df[df['rank'] <= 50]['player_id'].tolist())
        
        # At least 2 of 4 elite playmakers should be in top 50
        elite_in_top_50 = len(set(elite_playmakers) & top_50_ids)
        
        assert elite_in_top_50 >= 2, (
            f"Only {elite_in_top_50}/4 elite playmakers in top 50 primary assist RAPM. "
            f"This suggests the metric may not be capturing playmaking skill correctly."
        )
    
    def test_top_primary_assist_players_have_sufficient_toi(self, con):
        """
        TEST: Top 10 primary assist RAPM leaders should have meaningful TOI (>300 minutes).
        
        If players with very low TOI are topping the leaderboard, 
        the metric is too noisy to be useful.
        """
        df = con.execute("""
            SELECT player_id, value, toi_seconds
            FROM apm_results
            WHERE metric_name = 'primary_assist_rapm_5v5'
              AND season = (SELECT MAX(season) FROM apm_results)
            ORDER BY value DESC
            LIMIT 10
        """).df()
        
        min_toi_minutes = 300  # ~5 games of 60 min TOI
        low_toi_count = (df['toi_seconds'] < min_toi_minutes * 60).sum()
        
        assert low_toi_count <= 2, (
            f"{low_toi_count}/10 top players have < {min_toi_minutes} min TOI. "
            f"Top RAPM leaders should be regulars, not small-sample outliers."
        )
    
    def test_primary_assist_rapm_has_reasonable_spread(self, con):
        """
        TEST: Primary assist RAPM should have meaningful spread (std > 0.05).
        
        If all values are clustered near zero, the metric isn't differentiating players.
        """
        df = con.execute("""
            SELECT STDDEV(value) as std_val, 
                   MAX(value) - MIN(value) as range_val
            FROM apm_results
            WHERE metric_name = 'primary_assist_rapm_5v5'
              AND season = (SELECT MAX(season) FROM apm_results)
        """).df()
        
        std_val = df['std_val'].iloc[0]
        range_val = df['range_val'].iloc[0]
        
        assert std_val > 0.05, (
            f"Primary assist RAPM std={std_val:.4f} is too small. "
            f"Metric may be over-regularized."
        )
        
        assert range_val > 0.3, (
            f"Primary assist RAPM range={range_val:.4f} is too small. "
            f"Top and bottom players should differ by at least 0.3."
        )
    
    def test_correlation_with_xg_variant(self, con):
        """
        TEST: Primary assist RAPM should correlate with xG primary assist RAPM.
        
        Both metrics should identify similar players as top performers.
        """
        df = con.execute("""
            SELECT a.player_id, 
                   a.value as primary_assist_rapm,
                   b.value as xg_primary_assist_rapm
            FROM apm_results a
            JOIN apm_results b ON a.player_id = b.player_id AND a.season = b.season
            WHERE a.metric_name = 'primary_assist_rapm_5v5'
              AND b.metric_name = 'xg_primary_assist_on_goals_rapm_5v5'
              AND a.season = (SELECT MAX(season) FROM apm_results)
        """).df()
        
        if len(df) < 10:
            pytest.skip("Not enough data to test correlation")
        
        correlation = df['primary_assist_rapm'].corr(df['xg_primary_assist_rapm'])
        
        assert correlation > 0.3, (
            f"Correlation between primary assist variants is {correlation:.3f}. "
            f"Expected > 0.3 if both metrics measure similar skill."
        )
    
    def test_top_leader_is_not_an_obvious_outlier(self, con):
        """
        TEST: The #1 player should not have a value > 3 standard deviations from mean.
        
        Extreme outliers at the top often indicate data issues.
        """
        df = con.execute("""
            SELECT value, 
                   AVG(value) OVER () as mean_val,
                   STDDEV(value) OVER () as std_val
            FROM apm_results
            WHERE metric_name = 'primary_assist_rapm_5v5'
              AND season = (SELECT MAX(season) FROM apm_results)
            ORDER BY value DESC
            LIMIT 1
        """).df()
        
        top_value = df['value'].iloc[0]
        mean_val = df['mean_val'].iloc[0]
        std_val = df['std_val'].iloc[0]
        
        z_score = (top_value - mean_val) / std_val if std_val > 0 else 0
        
        assert z_score < 4, (
            f"Top primary assist RAPM has z-score={z_score:.2f}. "
            f"Values > 4 std from mean are suspicious outliers."
        )


class TestCompareWithCoreMetrics:
    """Compare assist metrics against well-validated core metrics."""
    
    def test_top_corsi_players_not_all_negative_in_assists(self, con):
        """
        TEST: Top 10 Corsi RAPM players shouldn't all have negative assist RAPM.
        
        Good overall players should generally contribute to assists too.
        """
        df = con.execute("""
            WITH top_corsi AS (
                SELECT player_id
                FROM apm_results
                WHERE metric_name = 'corsi_rapm_5v5'
                  AND season = (SELECT MAX(season) FROM apm_results)
                ORDER BY value DESC
                LIMIT 10
            )
            SELECT a.player_id, a.value as assist_rapm
            FROM apm_results a
            JOIN top_corsi t ON a.player_id = t.player_id
            WHERE a.metric_name = 'primary_assist_rapm_5v5'
              AND a.season = (SELECT MAX(season) FROM apm_results)
        """).df()
        
        negative_count = (df['assist_rapm'] < 0).sum()
        
        assert negative_count < 8, (
            f"{negative_count}/10 top Corsi players have negative assist RAPM. "
            f"This suggests the assist metric may have issues."
        )


if __name__ == "__main__":
    # Quick manual run
    con = duckdb.connect(str(DB_PATH), read_only=True)
    
    print("=" * 60)
    print("RAPM ASSIST METRICS VALIDATION")
    print("=" * 60)
    
    # Check metric stats
    print("\n1. METRIC STATISTICS:")
    for metric in ['primary_assist_rapm_5v5', 'xg_primary_assist_on_goals_rapm_5v5']:
        stats = con.execute(f"""
            SELECT COUNT(*) as n, 
                   AVG(value) as mean, 
                   STDDEV(value) as std,
                   MIN(value) as min_val,
                   MAX(value) as max_val
            FROM apm_results
            WHERE metric_name = '{metric}'
              AND season = (SELECT MAX(season) FROM apm_results)
        """).fetchone()
        print(f"  {metric}:")
        print(f"    n={stats[0]}, mean={stats[1]:.4f}, std={stats[2]:.4f}")
        print(f"    range=[{stats[3]:.4f}, {stats[4]:.4f}]")
    
    # Check elite playmakers
    print("\n2. ELITE PLAYMAKER RANKINGS:")
    elite = {8478402: "McDavid", 8477492: "MacKinnon", 8476453: "Kucherov", 8477934: "Draisaitl"}
    for pid, name in elite.items():
        result = con.execute(f"""
            WITH ranked AS (
                SELECT player_id, value,
                       ROW_NUMBER() OVER (ORDER BY value DESC) as rank
                FROM apm_results
                WHERE metric_name = 'primary_assist_rapm_5v5'
                  AND season = (SELECT MAX(season) FROM apm_results)
            )
            SELECT rank, value FROM ranked WHERE player_id = {pid}
        """).fetchone()
        if result:
            print(f"  {name}: rank #{result[0]}, value={result[1]:.3f}")
        else:
            print(f"  {name}: NOT FOUND")
    
    # Top 10 leaders
    print("\n3. TOP 10 PRIMARY ASSIST RAPM LEADERS:")
    leaders = con.execute("""
        SELECT a.player_id, p.full_name, a.value, a.toi_seconds
        FROM apm_results a
        LEFT JOIN players p ON a.player_id = p.player_id
        WHERE a.metric_name = 'primary_assist_rapm_5v5'
          AND a.season = (SELECT MAX(season) FROM apm_results)
        ORDER BY a.value DESC
        LIMIT 10
    """).fetchall()
    for i, (pid, name, val, toi) in enumerate(leaders):
        toi_min = (toi or 0) / 60
        print(f"  {i+1}. {name or pid}: {val:.3f} ({toi_min:.0f} min TOI)")
    
    con.close()
    print("\n" + "=" * 60)
    print("Run 'pytest test_assist_rapm_quality.py -v' for full test suite")
