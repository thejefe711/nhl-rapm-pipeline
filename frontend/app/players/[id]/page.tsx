'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import RadarChart from '@/components/RadarChart';
import {
    getPlayer, getPlayerRAPM, getPlayerExplanation, getPlayerProfile,
    PlayerDetail, RAPMRow, PlayerExplanation, PlayerProfile, PlayerMetric
} from '@/lib/api';
import styles from './page.module.css';

const RAPM_METRICS = [
    { value: 'corsi_rapm_5v5', label: 'Corsi RAPM (5v5)', category: 'Core' },
    { value: 'xg_rapm_5v5', label: 'Expected Goals (5v5)', category: 'Core' },
    { value: 'goals_rapm_5v5', label: 'Goals RAPM (5v5)', category: 'Core' },
    { value: 'corsi_off_rapm_5v5', label: 'Offensive Corsi', category: 'Offensive' },
    { value: 'xg_off_rapm_5v5', label: 'Offensive xG', category: 'Offensive' },
    { value: 'corsi_def_rapm_5v5', label: 'Defensive Corsi', category: 'Defensive' },
    { value: 'xg_def_rapm_5v5', label: 'Defensive xG', category: 'Defensive' },
    { value: 'hd_xg_rapm_5v5_ge020', label: 'High Danger xG', category: 'High Danger' },
    { value: 'hd_xg_off_rapm_5v5_ge020', label: 'HD Offensive xG', category: 'High Danger' },
    { value: 'hd_xg_def_rapm_5v5_ge020', label: 'HD Defensive xG', category: 'High Danger' },
    { value: 'turnover_to_xg_swing_rapm_5v5_w10', label: 'Turnover xG Swing', category: 'Transitions' },
    { value: 'takeaway_to_xg_swing_rapm_5v5_w10', label: 'Takeaway xG Swing', category: 'Transitions' },
    { value: 'giveaway_to_xg_swing_rapm_5v5_w10', label: 'Giveaway xG Swing', category: 'Transitions' },
    { value: 'penalties_drawn_rapm_5v5', label: 'Penalties Drawn', category: 'Discipline' },
    { value: 'penalties_committed_rapm_5v5', label: 'Penalties Taken', category: 'Discipline' },
    { value: 'primary_assist_rapm_5v5', label: 'Primary Assists', category: 'Playmaking' },
    { value: 'secondary_assist_rapm_5v5', label: 'Secondary Assists', category: 'Playmaking' },
    { value: 'xg_primary_assist_on_goals_rapm_5v5', label: 'Primary Assist xG', category: 'Playmaking' },
    { value: 'xg_secondary_assist_on_goals_rapm_5v5', label: 'Secondary Assist xG', category: 'Playmaking' },
];

const RADAR_METRICS = [
    { key: 'corsi_off_rapm_5v5', label: 'Off Corsi' },
    { key: 'corsi_def_rapm_5v5', label: 'Def Corsi' },
    { key: 'xg_off_rapm_5v5', label: 'Off xG' },
    { key: 'xg_def_rapm_5v5', label: 'Def xG' },
    { key: 'hd_xg_rapm_5v5_ge020', label: 'HD xG' },
    { key: 'primary_assist_rapm_5v5', label: 'Playmaking' },
    { key: 'penalties_drawn_rapm_5v5', label: 'Drawing' },
    { key: 'takeaway_to_xg_swing_rapm_5v5_w10', label: 'Transitions' },
];

const METRIC_CATEGORIES = [
    {
        id: 'core',
        title: 'Core Impact',
        keys: ['corsi_rapm_5v5', 'xg_rapm_5v5', 'goals_rapm_5v5'],
    },
    {
        id: 'offense',
        title: 'Offensive',
        keys: ['corsi_off_rapm_5v5', 'xg_off_rapm_5v5', 'hd_xg_off_rapm_5v5_ge020'],
    },
    {
        id: 'defense',
        title: 'Defensive',
        keys: ['corsi_def_rapm_5v5', 'xg_def_rapm_5v5', 'hd_xg_def_rapm_5v5_ge020'],
    },
    {
        id: 'playmaking',
        title: 'Playmaking',
        keys: ['primary_assist_rapm_5v5', 'secondary_assist_rapm_5v5', 'xg_primary_assist_on_goals_rapm_5v5'],
    },
    {
        id: 'transitions',
        title: 'Transitions',
        keys: ['takeaway_to_xg_swing_rapm_5v5_w10', 'giveaway_to_xg_swing_rapm_5v5_w10', 'turnover_to_xg_swing_rapm_5v5_w10'],
    },
    {
        id: 'discipline',
        title: 'Discipline',
        keys: ['penalties_drawn_rapm_5v5', 'penalties_committed_rapm_5v5'],
    },
];

function getMetricLabel(key: string): string {
    return RAPM_METRICS.find(m => m.value === key)?.label || key;
}

function getPositionLabel(pos: string | undefined): string {
    if (!pos) return 'Unknown';
    const map: Record<string, string> = { F: 'Forward', D: 'Defenseman', G: 'Goalie', L: 'Left Wing', R: 'Right Wing', C: 'Center' };
    return map[pos] || pos;
}

export default function PlayerDetailPage() {
    const params = useParams();
    const playerId = Number(params.id);

    const [profile, setProfile] = useState<PlayerProfile | null>(null);
    const [rapmData, setRapmData] = useState<RAPMRow[]>([]);
    const [explanation, setExplanation] = useState<PlayerExplanation | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isRapmLoading, setIsRapmLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedMetric, setSelectedMetric] = useState('corsi_rapm_5v5');
    const [activeCategory, setActiveCategory] = useState('core');

    useEffect(() => {
        async function loadPlayerData() {
            setIsLoading(true);
            setError(null);

            try {
                const profileRes = await getPlayerProfile(playerId);
                setProfile(profileRes);

                try {
                    const explanationRes = await getPlayerExplanation(playerId);
                    setExplanation(explanationRes);
                } catch { /* not available */ }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load player data');
            } finally {
                setIsLoading(false);
            }
        }

        if (playerId) loadPlayerData();
    }, [playerId]);

    useEffect(() => {
        async function loadRapmData() {
            if (!playerId) return;
            setIsRapmLoading(true);
            try {
                const rapmRes = await getPlayerRAPM(playerId, selectedMetric);
                setRapmData(rapmRes.rows);
            } catch {
                setRapmData([]);
            } finally {
                setIsRapmLoading(false);
            }
        }
        loadRapmData();
    }, [playerId, selectedMetric]);

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

    if (error || !profile) {
        return (
            <div className={styles.page}>
                <div className="container">
                    <div className={styles.error}>
                        <div className={styles.errorIcon}>!</div>
                        <h2>Error Loading Player</h2>
                        <p>{error || 'Player not found'}</p>
                        <Link href="/players" className="btn btn-secondary">Back to Search</Link>
                    </div>
                </div>
            </div>
        );
    }

    const player = profile.player;
    const metricsMap = new Map(profile.metrics.map(m => [m.metric_name, m]));
    const percentilesMap = profile.percentiles;

    const latestRAPM = rapmData.length > 0 ? rapmData[rapmData.length - 1].value : null;
    const avgRAPM = rapmData.length > 0 ? rapmData.reduce((sum, r) => sum + r.value, 0) / rapmData.length : null;
    const currentMetricLabel = RAPM_METRICS.find(m => m.value === selectedMetric)?.label || selectedMetric;

    // Build radar data
    const radarData = RADAR_METRICS.map(rm => {
        const pct = percentilesMap[rm.key];
        const metric = metricsMap.get(rm.key);
        return {
            label: rm.label,
            value: metric?.value ?? 0,
            percentile: pct?.percentile ?? 50,
        };
    });

    // Core stat cards
    const corsiMetric = metricsMap.get('corsi_rapm_5v5');
    const xgMetric = metricsMap.get('xg_rapm_5v5');
    const goalsMetric = metricsMap.get('goals_rapm_5v5');

    return (
        <div className={styles.page}>
            <div className="container">
                <Link href="/players" className={styles.backLink}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M19 12H5M12 19l-7-7 7-7" />
                    </svg>
                    <span>Back to Players</span>
                </Link>

                {/* Player Header */}
                <header className={styles.playerHeader}>
                    <div className={styles.headerLeft}>
                        <div className={styles.playerAvatar}>
                            {player.full_name?.split(' ').map(n => n[0]).join('') || '?'}
                        </div>
                        <div className={styles.playerInfo}>
                            <div className={styles.playerNameRow}>
                                <h1>{player.full_name || `Player #${playerId}`}</h1>
                                {player.position && (
                                    <span className={styles.positionBadge}>
                                        {getPositionLabel(player.position)}
                                    </span>
                                )}
                            </div>
                            <div className={styles.playerMeta}>
                                <span className={styles.metaItem}>
                                    <span className={styles.metaLabel}>Season</span>
                                    <span className={styles.metaValue}>{formatSeason(profile.season)}</span>
                                </span>
                                {player.seasons_count && (
                                    <span className={styles.metaItem}>
                                        <span className={styles.metaLabel}>Career Seasons</span>
                                        <span className={styles.metaValue}>{player.seasons_count}</span>
                                    </span>
                                )}
                                {corsiMetric?.games_count && (
                                    <span className={styles.metaItem}>
                                        <span className={styles.metaLabel}>Games</span>
                                        <span className={styles.metaValue}>{corsiMetric.games_count}</span>
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className={styles.headerStats}>
                        {corsiMetric && (
                            <div className={styles.headerStatCard}>
                                <div className={`${styles.headerStatValue} ${corsiMetric.value >= 0 ? styles.positive : styles.negative}`}>
                                    {corsiMetric.value >= 0 ? '+' : ''}{corsiMetric.value.toFixed(2)}
                                </div>
                                <div className={styles.headerStatLabel}>Corsi RAPM</div>
                                {percentilesMap['corsi_rapm_5v5'] && (
                                    <div className={styles.percentileBadge}>
                                        {Math.round(percentilesMap['corsi_rapm_5v5'].percentile)}th pct
                                    </div>
                                )}
                            </div>
                        )}
                        {xgMetric && (
                            <div className={styles.headerStatCard}>
                                <div className={`${styles.headerStatValue} ${xgMetric.value >= 0 ? styles.positive : styles.negative}`}>
                                    {xgMetric.value >= 0 ? '+' : ''}{xgMetric.value.toFixed(3)}
                                </div>
                                <div className={styles.headerStatLabel}>xG RAPM</div>
                                {percentilesMap['xg_rapm_5v5'] && (
                                    <div className={styles.percentileBadge}>
                                        {Math.round(percentilesMap['xg_rapm_5v5'].percentile)}th pct
                                    </div>
                                )}
                            </div>
                        )}
                        {goalsMetric && (
                            <div className={styles.headerStatCard}>
                                <div className={`${styles.headerStatValue} ${goalsMetric.value >= 0 ? styles.positive : styles.negative}`}>
                                    {goalsMetric.value >= 0 ? '+' : ''}{goalsMetric.value.toFixed(3)}
                                </div>
                                <div className={styles.headerStatLabel}>Goals RAPM</div>
                                {percentilesMap['goals_rapm_5v5'] && (
                                    <div className={styles.percentileBadge}>
                                        {Math.round(percentilesMap['goals_rapm_5v5'].percentile)}th pct
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </header>

                <div className={styles.contentGrid}>
                    {/* Left Column */}
                    <div className={styles.leftColumn}>
                        {/* Radar Chart */}
                        <section className={styles.section}>
                            <div className={styles.sectionHeader}>
                                <h2>Player Profile</h2>
                                <span className={styles.sectionBadge}>Percentile Radar</span>
                            </div>
                            <RadarChart data={radarData} size={320} />
                            <div className={styles.radarLegend}>
                                <span className={styles.radarNote}>
                                    50th percentile (league average) shown as dashed ring
                                </span>
                            </div>
                        </section>

                        {/* Categorized Metrics */}
                        <section className={styles.section}>
                            <div className={styles.sectionHeader}>
                                <h2>All Metrics</h2>
                                <span className={styles.sectionBadge}>{formatSeason(profile.season)}</span>
                            </div>

                            <div className={styles.categoryTabs}>
                                {METRIC_CATEGORIES.map(cat => (
                                    <button
                                        key={cat.id}
                                        className={`${styles.categoryTab} ${activeCategory === cat.id ? styles.activeTab : ''}`}
                                        onClick={() => setActiveCategory(cat.id)}
                                    >
                                        {cat.title}
                                    </button>
                                ))}
                            </div>

                            <div className={styles.metricsList}>
                                {METRIC_CATEGORIES.find(c => c.id === activeCategory)?.keys.map(key => {
                                    const m = metricsMap.get(key);
                                    const pct = percentilesMap[key];
                                    if (!m) return null;

                                    return (
                                        <div key={key} className={styles.metricRow}>
                                            <div className={styles.metricInfo}>
                                                <span className={styles.metricName}>{getMetricLabel(key)}</span>
                                                <div className={`${styles.metricValue} ${m.value >= 0 ? styles.positive : styles.negative}`}>
                                                    {m.value >= 0 ? '+' : ''}{m.value.toFixed(3)}
                                                </div>
                                            </div>
                                            {pct && (
                                                <div className={styles.percentileBar}>
                                                    <div className={styles.percentileTrack}>
                                                        <div
                                                            className={styles.percentileFill}
                                                            style={{
                                                                width: `${pct.percentile}%`,
                                                                background: pct.percentile >= 75
                                                                    ? 'var(--positive)'
                                                                    : pct.percentile >= 50
                                                                        ? 'var(--accent)'
                                                                        : pct.percentile >= 25
                                                                            ? 'var(--warning)'
                                                                            : 'var(--negative)',
                                                            }}
                                                        />
                                                        <div className={styles.percentileMedian} />
                                                    </div>
                                                    <span className={styles.percentileLabel}>
                                                        {Math.round(pct.percentile)}th
                                                    </span>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </section>
                    </div>

                    {/* Right Column */}
                    <div className={styles.rightColumn}>
                        {/* Season Trend */}
                        <section className={styles.section}>
                            <div className={styles.sectionHeader}>
                                <h2>Season Trend</h2>
                                <select
                                    className={styles.metricSelector}
                                    value={selectedMetric}
                                    onChange={(e) => setSelectedMetric(e.target.value)}
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

                                    {/* Career summary */}
                                    {avgRAPM !== null && (
                                        <div className={styles.careerSummary}>
                                            <span className={styles.careerLabel}>Career Average</span>
                                            <span className={`${styles.careerValue} ${avgRAPM >= 0 ? styles.positive : styles.negative}`}>
                                                {avgRAPM >= 0 ? '+' : ''}{avgRAPM.toFixed(2)}
                                            </span>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className={styles.noData}>
                                    <p>No data available for {currentMetricLabel}</p>
                                </div>
                            )}
                        </section>

                        {/* AI Analysis */}
                        <section className={styles.section}>
                            <div className={styles.sectionHeader}>
                                <h2>AI Analysis</h2>
                                {explanation && (
                                    <span className={`badge ${explanation.data_quality === 'good' ? 'badge-success' : 'badge-warning'}`}>
                                        {explanation.data_quality}
                                    </span>
                                )}
                            </div>

                            {explanation ? (
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
                            ) : (
                                <div className={styles.noData}>
                                    <p>AI analysis requires DLM forecast data.</p>
                                    <Link href="/glossary#methodology" className={styles.learnMore}>
                                        Learn about our methodology &rarr;
                                    </Link>
                                </div>
                            )}
                        </section>

                        {/* Quick Actions */}
                        <section className={styles.section}>
                            <div className={styles.sectionHeader}>
                                <h2>Quick Actions</h2>
                            </div>
                            <div className={styles.quickActions}>
                                <Link href={`/compare`} className={styles.actionBtn}>
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                                    </svg>
                                    Compare with another player
                                </Link>
                                <Link href={`/leaderboards?metric=corsi_rapm_5v5`} className={styles.actionBtn}>
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10" />
                                    </svg>
                                    View leaderboards
                                </Link>
                                <Link href="/glossary" className={styles.actionBtn}>
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253" />
                                    </svg>
                                    Metric glossary
                                </Link>
                            </div>
                        </section>
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
