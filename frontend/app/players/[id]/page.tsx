'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { getPlayer, getPlayerRAPM, getPlayerExplanation, PlayerDetail, RAPMRow, PlayerExplanation } from '@/lib/api';
import styles from './page.module.css';

// Available RAPM metrics organized by category
const RAPM_METRICS = [
    // Core Metrics
    { value: 'corsi_rapm_5v5', label: 'Corsi RAPM (5v5)', category: 'Core' },
    { value: 'xg_rapm_5v5', label: 'Expected Goals (5v5)', category: 'Core' },
    { value: 'goals_rapm_5v5', label: 'Goals RAPM (5v5)', category: 'Core' },
    // Offensive
    { value: 'corsi_off_rapm_5v5', label: 'Offensive Corsi', category: 'Offensive' },
    { value: 'xg_off_rapm_5v5', label: 'Offensive xG', category: 'Offensive' },
    // Defensive
    { value: 'corsi_def_rapm_5v5', label: 'Defensive Corsi', category: 'Defensive' },
    { value: 'xg_def_rapm_5v5', label: 'Defensive xG', category: 'Defensive' },
    // High Danger
    { value: 'hd_xg_rapm_5v5_ge020', label: 'High Danger xG', category: 'High Danger' },
    { value: 'hd_xg_off_rapm_5v5_ge020', label: 'HD Offensive xG', category: 'High Danger' },
    { value: 'hd_xg_def_rapm_5v5_ge020', label: 'HD Defensive xG', category: 'High Danger' },
    // Turnovers & Transitions
    { value: 'turnover_to_xg_swing_rapm_5v5_w10', label: 'Turnover xG Swing', category: 'Transitions' },
    { value: 'takeaway_to_xg_swing_rapm_5v5_w10', label: 'Takeaway xG Swing', category: 'Transitions' },
    { value: 'giveaway_to_xg_swing_rapm_5v5_w10', label: 'Giveaway xG Swing', category: 'Transitions' },
    // Penalties
    { value: 'penalties_drawn_rapm_5v5', label: 'Penalties Drawn', category: 'Discipline' },
    { value: 'penalties_committed_rapm_5v5', label: 'Penalties Taken', category: 'Discipline' },
    // Playmaking
    { value: 'primary_assist_rapm_5v5', label: 'Primary Assists', category: 'Playmaking' },
    { value: 'secondary_assist_rapm_5v5', label: 'Secondary Assists', category: 'Playmaking' },
    { value: 'xg_primary_assist_on_goals_rapm_5v5', label: 'Primary Assist xG', category: 'Playmaking' },
    { value: 'xg_secondary_assist_on_goals_rapm_5v5', label: 'Secondary Assist xG', category: 'Playmaking' },
];

export default function PlayerDetailPage() {
    const params = useParams();
    const playerId = Number(params.id);

    const [player, setPlayer] = useState<PlayerDetail | null>(null);
    const [rapmData, setRapmData] = useState<RAPMRow[]>([]);
    const [explanation, setExplanation] = useState<PlayerExplanation | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isRapmLoading, setIsRapmLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedMetric, setSelectedMetric] = useState('corsi_rapm_5v5');

    // Load player info and explanation (once)
    useEffect(() => {
        async function loadPlayerData() {
            setIsLoading(true);
            setError(null);

            try {
                const playerRes = await getPlayer(playerId);
                setPlayer(playerRes.player);

                try {
                    const explanationRes = await getPlayerExplanation(playerId);
                    setExplanation(explanationRes);
                } catch {
                    // Explanation not available
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load player data');
            } finally {
                setIsLoading(false);
            }
        }

        if (playerId) {
            loadPlayerData();
        }
    }, [playerId]);

    // Load RAPM data (when metric changes)
    useEffect(() => {
        async function loadRapmData() {
            if (!playerId) return;

            setIsRapmLoading(true);
            try {
                const rapmRes = await getPlayerRAPM(playerId, selectedMetric);
                setRapmData(rapmRes.rows);
            } catch (err) {
                console.error('Failed to load RAPM data:', err);
                setRapmData([]);
            } finally {
                setIsRapmLoading(false);
            }
        }

        loadRapmData();
    }, [playerId, selectedMetric]);

    const handleMetricChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        setSelectedMetric(e.target.value);
    };

    const currentMetricLabel = RAPM_METRICS.find(m => m.value === selectedMetric)?.label || selectedMetric;

    if (isLoading) {
        return (
            <div className={styles.page}>
                <div className="container">
                    <div className={styles.loading}>
                        <div className="spinner" />
                        <p>Loading player data...</p>
                    </div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className={styles.page}>
                <div className="container">
                    <div className={styles.error}>
                        <div className={styles.errorIcon}>⚠️</div>
                        <h2>Error Loading Player</h2>
                        <p>{error}</p>
                        <Link href="/players" className="btn btn-secondary">
                            ← Back to Search
                        </Link>
                    </div>
                </div>
            </div>
        );
    }

    const latestRAPM = rapmData.length > 0 ? rapmData[rapmData.length - 1].value : null;
    const avgRAPM = rapmData.length > 0 ? rapmData.reduce((sum, r) => sum + r.value, 0) / rapmData.length : null;

    return (
        <div className={styles.page}>
            <div className="container">
                <Link href="/players" className={styles.backLink}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M19 12H5M12 19l-7-7 7-7" />
                    </svg>
                    <span>Back to Search</span>
                </Link>

                {/* Player Header */}
                <header className={styles.playerHeader}>
                    <div className={styles.playerAvatar}>
                        {player?.full_name?.split(' ').map(n => n[0]).join('') || '?'}
                    </div>
                    <div className={styles.playerInfo}>
                        <h1>{player?.full_name || `Player #${playerId}`}</h1>
                        <div className={styles.playerMeta}>
                            <span className={styles.metaItem}>
                                <span className={styles.metaLabel}>ID</span>
                                <span className={styles.metaValue}>{playerId}</span>
                            </span>
                            {player?.seasons_count && (
                                <span className={styles.metaItem}>
                                    <span className={styles.metaLabel}>Seasons</span>
                                    <span className={styles.metaValue}>{player.seasons_count}</span>
                                </span>
                            )}
                        </div>
                    </div>

                    {latestRAPM !== null && (
                        <div className={styles.mainStats}>
                            <div className={styles.mainStatCard}>
                                <div className={`${styles.mainStatValue} ${latestRAPM >= 0 ? styles.positive : styles.negative}`}>
                                    {latestRAPM >= 0 ? '+' : ''}{latestRAPM.toFixed(2)}
                                </div>
                                <div className={styles.mainStatLabel}>Latest RAPM</div>
                            </div>
                            {avgRAPM !== null && (
                                <div className={styles.mainStatCard}>
                                    <div className={`${styles.mainStatValue} ${avgRAPM >= 0 ? styles.positive : styles.negative}`}>
                                        {avgRAPM >= 0 ? '+' : ''}{avgRAPM.toFixed(2)}
                                    </div>
                                    <div className={styles.mainStatLabel}>Career Avg</div>
                                </div>
                            )}
                        </div>
                    )}
                </header>

                <div className={styles.contentGrid}>
                    {/* Left Column - Stats */}
                    <div className={styles.leftColumn}>
                        {/* Season Stats */}
                        <section className={styles.section}>
                            <div className={styles.sectionHeader}>
                                <h2>RAPM by Season</h2>
                                <select
                                    className={styles.metricSelector}
                                    value={selectedMetric}
                                    onChange={handleMetricChange}
                                    disabled={isRapmLoading}
                                >
                                    {RAPM_METRICS.map(m => (
                                        <option key={m.value} value={m.value}>{m.label}</option>
                                    ))}
                                </select>
                            </div>

                            {isRapmLoading ? (
                                <div className={styles.loading}>
                                    <div className="spinner" />
                                    <p>Loading {currentMetricLabel}...</p>
                                </div>
                            ) : rapmData.length > 0 ? (

                                <div className={styles.seasonList}>
                                    {rapmData.slice().reverse().map((row) => (
                                        <div key={row.season} className={styles.seasonRow}>
                                            <div className={styles.seasonInfo}>
                                                <span className={styles.seasonName}>{formatSeason(row.season)}</span>
                                            </div>
                                            <div className={styles.seasonBar}>
                                                <div
                                                    className={`${styles.seasonBarFill} ${row.value >= 0 ? styles.positive : styles.negative}`}
                                                    style={{ width: `${Math.min(Math.abs(row.value) * 20, 100)}%` }}
                                                />
                                            </div>
                                            <div className={`${styles.seasonValue} ${row.value >= 0 ? styles.positive : styles.negative}`}>
                                                {row.value >= 0 ? '+' : ''}{row.value.toFixed(2)}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className={styles.noData}>
                                    <p>No RAPM data available</p>
                                </div>
                            )}
                        </section>
                    </div>

                    {/* Right Column - AI Insights */}
                    <div className={styles.rightColumn}>
                        {explanation && (
                            <section className={styles.section}>
                                <div className={styles.sectionHeader}>
                                    <h2>🤖 AI Analysis</h2>
                                    <span className={`badge ${explanation.data_quality === 'good' ? 'badge-success' : 'badge-warning'}`}>
                                        {explanation.data_quality}
                                    </span>
                                </div>

                                <div className={styles.aiCard}>
                                    <p className={styles.aiText}>{explanation.explanation}</p>

                                    <div className={styles.skillsGrid}>
                                        <div className={styles.skillBox}>
                                            <div className={styles.skillValue}>{explanation.stable_skills}</div>
                                            <div className={styles.skillLabel}>Stable Skills</div>
                                        </div>
                                        <div className={styles.skillBox}>
                                            <div className={styles.skillValue}>{explanation.emerging_skills}</div>
                                            <div className={styles.skillLabel}>Emerging</div>
                                        </div>
                                    </div>
                                </div>
                            </section>
                        )}

                        {!explanation && (
                            <section className={styles.section}>
                                <div className={styles.sectionHeader}>
                                    <h2>🤖 AI Analysis</h2>
                                </div>
                                <div className={styles.noData}>
                                    <p>AI analysis not available for this player</p>
                                </div>
                            </section>
                        )}
                    </div>
                </div>
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
