'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { getLeaderboard, getSeasons, LeaderboardRow } from '@/lib/api';
import styles from './page.module.css';

const METRICS = [
    // Core Metrics
    { value: 'corsi_rapm_5v5', label: 'Corsi RAPM (5v5)', description: 'Overall shot differential impact', category: 'Core' },
    { value: 'xg_rapm_5v5', label: 'Expected Goals (5v5)', description: 'Expected goals differential impact', category: 'Core' },
    { value: 'goals_rapm_5v5', label: 'Goals RAPM (5v5)', description: 'Actual goals differential impact', category: 'Core' },
    // Offensive
    { value: 'corsi_off_rapm_5v5', label: 'Offensive Corsi', description: 'Offensive shot generation', category: 'Offensive' },
    { value: 'xg_off_rapm_5v5', label: 'Offensive xG', description: 'Offensive expected goals generation', category: 'Offensive' },
    { value: 'finishing_residual_rapm_5v5', label: 'Finishing (Above xG)', description: 'Scoring above expected goals', category: 'Offensive' },
    // Defensive
    { value: 'corsi_def_rapm_5v5', label: 'Defensive Corsi', description: 'Defensive shot suppression', category: 'Defensive' },
    { value: 'xg_def_rapm_5v5', label: 'Defensive xG', description: 'Defensive expected goals suppression', category: 'Defensive' },
    // High Danger
    { value: 'hd_xg_rapm_5v5_ge020', label: 'High Danger xG', description: 'High danger chance impact', category: 'High Danger' },
    { value: 'hd_xg_off_rapm_5v5_ge020', label: 'HD Offensive xG', description: 'High danger offensive generation', category: 'High Danger' },
    { value: 'hd_xg_def_rapm_5v5_ge020', label: 'HD Defensive xG', description: 'High danger defensive suppression', category: 'High Danger' },
    // Turnovers & Transitions
    { value: 'turnover_to_xg_swing_rapm_5v5_w10', label: 'Turnover xG Swing', description: 'xG impact from turnovers', category: 'Transitions' },
    { value: 'takeaway_to_xg_swing_rapm_5v5_w10', label: 'Takeaway xG Swing', description: 'xG gained from takeaways', category: 'Transitions' },
    { value: 'giveaway_to_xg_swing_rapm_5v5_w10', label: 'Giveaway xG Swing', description: 'xG lost from giveaways', category: 'Transitions' },
    { value: 'faceoff_loss_to_xg_swing_rapm_5v5_w10', label: 'Faceoff Loss xG', description: 'xG impact from faceoff losses', category: 'Transitions' },
    { value: 'blocked_shot_to_xg_swing_rapm_5v5_w10', label: 'Blocked Shot xG', description: 'xG saved via blocked shots', category: 'Transitions' },
    // Special Teams
    { value: 'corsi_pp_off_rapm', label: 'Power Play Corsi', description: 'Power play shot generation', category: 'Special Teams' },
    { value: 'xg_pp_off_rapm', label: 'Power Play xG', description: 'Power play expected goals', category: 'Special Teams' },
    { value: 'corsi_pk_def_rapm', label: 'Penalty Kill Corsi', description: 'Penalty kill shot suppression', category: 'Special Teams' },
    { value: 'xg_pk_def_rapm', label: 'Penalty Kill xG', description: 'Penalty kill expected goals suppression', category: 'Special Teams' },
    // Penalties
    { value: 'penalties_drawn_rapm_5v5', label: 'Penalties Drawn', description: 'Ability to draw penalties', category: 'Discipline' },
    { value: 'penalties_taken_rapm_5v5', label: 'Penalties Taken', description: 'Penalty avoidance (lower is better)', category: 'Discipline' },
    // Playmaking
    { value: 'primary_assist_rapm_5v5', label: 'Primary Assists', description: 'Primary assist generation', category: 'Playmaking' },
    { value: 'secondary_assist_rapm_5v5', label: 'Secondary Assists', description: 'Secondary assist generation', category: 'Playmaking' },
    { value: 'xg_primary_assist_on_goals_rapm_5v5', label: 'Primary Assist xG', description: 'xG on primary assists', category: 'Playmaking' },
    { value: 'xg_secondary_assist_on_goals_rapm_5v5', label: 'Secondary Assist xG', description: 'xG on secondary assists', category: 'Playmaking' },
];

export default function LeaderboardsPage() {
    const [seasons, setSeasons] = useState<string[]>([]);
    const [selectedSeason, setSelectedSeason] = useState<string>('');
    const [selectedMetric, setSelectedMetric] = useState('corsi_rapm_5v5');
    const [leaderboard, setLeaderboard] = useState<LeaderboardRow[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        async function loadSeasons() {
            try {
                const data = await getSeasons();
                setSeasons(data.seasons);
                if (data.seasons.length > 0) {
                    setSelectedSeason(data.seasons[data.seasons.length - 1]);
                }
            } catch (err) {
                console.error('Failed to load seasons:', err);
            }
        }
        loadSeasons();
    }, []);

    useEffect(() => {
        async function loadLeaderboard() {
            if (!selectedSeason) return;

            setIsLoading(true);
            try {
                const data = await getLeaderboard(selectedMetric, selectedSeason, 50);
                setLeaderboard(data.rows);
            } catch (err) {
                console.error('Failed to load leaderboard:', err);
                setLeaderboard([]);
            } finally {
                setIsLoading(false);
            }
        }
        loadLeaderboard();
    }, [selectedSeason, selectedMetric]);

    const currentMetricInfo = METRICS.find(m => m.value === selectedMetric);

    return (
        <div className={styles.page}>
            <div className="container">
                <header className={styles.header}>
                    <div className={styles.headerBadge}>
                        <span>Rankings</span>
                    </div>
                    <h1>Leaderboards</h1>
                    <p className={styles.subtitle}>
                        Top performers by advanced analytics metrics
                    </p>
                </header>

                {/* Filters */}
                <section className={styles.filters}>
                    <div className={styles.filterCard}>
                        <div className={styles.filterGroup}>
                            <label htmlFor="season">Season</label>
                            <select
                                id="season"
                                className="input select"
                                value={selectedSeason}
                                onChange={(e) => setSelectedSeason(e.target.value)}
                            >
                                {seasons.map((season) => (
                                    <option key={season} value={season}>
                                        {formatSeason(season)}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className={styles.filterGroup}>
                            <label htmlFor="metric">Metric</label>
                            <select
                                id="metric"
                                className="input select"
                                value={selectedMetric}
                                onChange={(e) => setSelectedMetric(e.target.value)}
                            >
                                {METRICS.map((m) => (
                                    <option key={m.value} value={m.value}>
                                        {m.label}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {currentMetricInfo && (
                        <p className={styles.metricDescription}>
                            {currentMetricInfo.description}
                        </p>
                    )}
                </section>

                {/* Leaderboard */}
                <section className={styles.leaderboard}>
                    {isLoading ? (
                        <div className={styles.loading}>
                            <div className="spinner" />
                        </div>
                    ) : leaderboard.length === 0 ? (
                        <div className={styles.noData}>
                            <p>No data available for this selection</p>
                        </div>
                    ) : (
                        <>
                            {/* Data Quality Warning */}
                            {leaderboard[0]?.games_count && leaderboard[0].games_count < 100 && (
                                <div className={styles.dataWarning}>
                                    <span className={styles.warningIcon}>⚠️</span>
                                    <div>
                                        <strong>Limited Sample Size</strong>
                                        <p>This metric was computed using only {leaderboard[0].games_count} games. Results may be less reliable than metrics with full season data.</p>
                                    </div>
                                </div>
                            )}
                            {/* Top 3 Podium */}
                            <div className={styles.podium}>
                                {leaderboard.slice(0, 3).map((row, index) => (
                                    <Link
                                        key={row.player_id}
                                        href={`/players/${row.player_id}`}
                                        className={`${styles.podiumCard} ${styles[`rank${index + 1}`]}`}
                                    >
                                        <div className={styles.podiumMedal}>
                                            {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                                        </div>
                                        <div className={styles.podiumAvatar}>
                                            {(row.full_name || 'P').split(' ').map(n => n[0]).join('')}
                                        </div>
                                        <h3 className={styles.podiumName}>
                                            {row.full_name || `Player #${row.player_id}`}
                                        </h3>
                                        <div className={`${styles.podiumValue} ${row.value >= 0 ? styles.positive : styles.negative}`}>
                                            {row.value >= 0 ? '+' : ''}{row.value.toFixed(2)}
                                        </div>
                                    </Link>
                                ))}
                            </div>

                            {/* Rest of Leaderboard */}
                            <div className={styles.tableContainer}>
                                <table className="table">
                                    <thead>
                                        <tr>
                                            <th style={{ width: '60px' }}>Rank</th>
                                            <th>Player</th>
                                            <th style={{ width: '120px', textAlign: 'right' }}>{currentMetricInfo?.label || 'Value'}</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {leaderboard.slice(3).map((row, index) => (
                                            <tr key={row.player_id}>
                                                <td className={styles.rank}>{index + 4}</td>
                                                <td>
                                                    <Link href={`/players/${row.player_id}`} className={styles.playerLink}>
                                                        <div className={styles.playerAvatar}>
                                                            {(row.full_name || 'P')[0]}
                                                        </div>
                                                        <span>{row.full_name || `Player #${row.player_id}`}</span>
                                                    </Link>
                                                </td>
                                                <td className={`${styles.value} ${row.value >= 0 ? styles.positive : styles.negative}`}>
                                                    {row.value >= 0 ? '+' : ''}{row.value.toFixed(2)}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </>
                    )}
                </section>
            </div>
        </div>
    );
}

function formatSeason(season: string): string {
    if (season.length === 8) {
        return `${season.slice(0, 4)}-${season.slice(6, 8)}`;
    }
    return season;
}
