'use client';

import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { getLeaderboard, getSeasons, getConditionalLeaderboard, LeaderboardRow } from '@/lib/api';
import ReliabilityBadge from '@/components/ReliabilityBadge';
import styles from './page.module.css';

const METRIC_CATEGORIES = [
    {
        id: 'core',
        label: 'Core',
        metrics: [
            { value: 'corsi_rapm_5v5', label: 'Corsi RAPM', description: 'Overall shot differential impact' },
            { value: 'xg_rapm_5v5', label: 'xG RAPM', description: 'Expected goals differential impact' },
            { value: 'goals_rapm_5v5', label: 'Goals RAPM', description: 'Actual goals differential impact' },
        ],
    },
    {
        id: 'offense',
        label: 'Offense',
        metrics: [
            { value: 'corsi_off_rapm_5v5', label: 'Off Corsi', description: 'Offensive shot generation' },
            { value: 'xg_off_rapm_5v5', label: 'Off xG', description: 'Offensive expected goals generation' },
            { value: 'hd_xg_off_rapm_5v5_ge020', label: 'HD Off xG', description: 'High danger offensive generation' },
        ],
    },
    {
        id: 'defense',
        label: 'Defense',
        metrics: [
            { value: 'corsi_def_rapm_5v5', label: 'Def Corsi', description: 'Defensive shot suppression' },
            { value: 'xg_def_rapm_5v5', label: 'Def xG', description: 'Defensive expected goals suppression' },
            { value: 'hd_xg_def_rapm_5v5_ge020', label: 'HD Def xG', description: 'High danger defensive suppression' },
        ],
    },
    {
        id: 'danger',
        label: 'High Danger',
        metrics: [
            { value: 'hd_xg_rapm_5v5_ge020', label: 'HD xG Net', description: 'High danger chance impact' },
            { value: 'hd_xg_off_rapm_5v5_ge020', label: 'HD Off xG', description: 'High danger offensive generation' },
            { value: 'hd_xg_def_rapm_5v5_ge020', label: 'HD Def xG', description: 'High danger defensive suppression' },
        ],
    },
    {
        id: 'transitions',
        label: 'Transitions',
        metrics: [
            { value: 'takeaway_to_xg_swing_rapm_5v5_w10', label: 'Takeaway Swing', description: 'xG gained from takeaways' },
            { value: 'giveaway_to_xg_swing_rapm_5v5_w10', label: 'Giveaway Swing', description: 'xG lost from giveaways' },
            { value: 'turnover_to_xg_swing_rapm_5v5_w10', label: 'Turnover Net', description: 'Net xG from turnovers' },
        ],
    },
    {
        id: 'playmaking',
        label: 'Playmaking',
        metrics: [
            { value: 'primary_assist_rapm_5v5', label: 'Primary Assists', description: 'Primary assist generation' },
            { value: 'secondary_assist_rapm_5v5', label: 'Secondary Assists', description: 'Secondary assist generation' },
            { value: 'xg_primary_assist_on_goals_rapm_5v5', label: 'Assist xG', description: 'xG on primary assists' },
        ],
    },
    {
        id: 'discipline',
        label: 'Discipline',
        metrics: [
            { value: 'penalties_drawn_rapm_5v5', label: 'Drawn', description: 'Ability to draw penalties' },
            { value: 'penalties_committed_rapm_5v5', label: 'Taken', description: 'Penalty avoidance (lower better)' },
        ],
    },
    {
        id: 'situational',
        label: 'Situational',
        metrics: [
            { value: 'shutdown_score_z', label: 'Shutdown Rank', description: 'Defensive suppression against elite competition' },
            { value: 'breaker_score_z', label: 'Breaker Rank', description: 'Offensive generation against elite competition' },
            { value: 'clutch_shutdown_z', label: 'Clutch Shutdown', description: 'Defensive suppression in close games (score diff <= 1)' },
            { value: 'clutch_breaker_z', label: 'Clutch Breaker', description: 'Offensive generation in close games (score diff <= 1)' },
            { value: 'psi_z', label: 'Independence (low PSI)', description: 'Performance independent of linemates (lower PSI is better)' },
            { value: 'psi_twoway_z', label: 'Two-way PSI', description: 'Two-way performance dependence on linemates' },
            { value: 'elasticity_z', label: 'Elasticity', description: 'Growth potential with more ice time' },
        ],
    },
];

export default function LeaderboardsPage() {
    return (
        <Suspense fallback={<div className={styles.page}><div className="container"><div className={styles.loading}><div className="spinner" /></div></div></div>}>
            <LeaderboardsContent />
        </Suspense>
    );
}

function LeaderboardsContent() {
    const searchParams = useSearchParams();
    const initialMetric = searchParams.get('metric') || 'corsi_rapm_5v5';

    const [seasons, setSeasons] = useState<string[]>([]);
    const [selectedSeason, setSelectedSeason] = useState<string>('');
    const [selectedPosition, setSelectedPosition] = useState<string>('All');
    const [selectedMetric, setSelectedMetric] = useState(initialMetric);
    const [activeCategory, setActiveCategory] = useState(() => {
        const cat = METRIC_CATEGORIES.find(c => c.metrics.some(m => m.value === initialMetric));
        return cat?.id || 'core';
    });
    const [leaderboard, setLeaderboard] = useState<LeaderboardRow[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [minToi, setMinToi] = useState(3600); // default: ~10 GP

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
                let data;
                if (activeCategory === 'situational') {
                    data = await getConditionalLeaderboard(selectedMetric, selectedSeason, selectedPosition, 50, minToi);
                } else {
                    data = await getLeaderboard(selectedMetric, selectedSeason, 50, minToi);
                }
                setLeaderboard(data.rows);
            } catch (err) {
                console.error('Failed to load leaderboard:', err);
                setLeaderboard([]);
            } finally {
                setIsLoading(false);
            }
        }
        loadLeaderboard();
    }, [selectedSeason, selectedMetric, activeCategory, selectedPosition, minToi]);

    const handleCategoryChange = (catId: string) => {
        setActiveCategory(catId);
        const cat = METRIC_CATEGORIES.find(c => c.id === catId);
        if (cat && cat.metrics.length > 0) {
            setSelectedMetric(cat.metrics[0].value);
        }
    };

    const currentCategory = METRIC_CATEGORIES.find(c => c.id === activeCategory);
    const currentMetricInfo = currentCategory?.metrics.find(m => m.value === selectedMetric);

    // Find max value for bar scaling
    const maxAbsValue = leaderboard.length > 0
        ? Math.max(...leaderboard.map(r => Math.abs(r.value)))
        : 1;

    return (
        <div className={styles.page}>
            <div className="container">
                <header className={styles.header}>
                    <div className={styles.headerBadge}>
                        <span>Rankings</span>
                    </div>
                    <h1>Leaderboards</h1>
                    <p className={styles.subtitle}>
                        Top performers across 20+ advanced analytics metrics
                    </p>
                </header>

                {/* Category Tabs */}
                <div className={styles.categoryTabs}>
                    {METRIC_CATEGORIES.map(cat => (
                        <button
                            key={cat.id}
                            className={`${styles.categoryTab} ${activeCategory === cat.id ? styles.activeTab : ''}`}
                            onClick={() => handleCategoryChange(cat.id)}
                        >
                            {cat.label}
                        </button>
                    ))}
                </div>

                {/* Filters */}
                <section className={styles.filters}>
                    <div className={styles.filterRow}>
                        <div className={styles.metricTabs}>
                            {currentCategory?.metrics.map(m => (
                                <button
                                    key={m.value}
                                    className={`${styles.metricTab} ${selectedMetric === m.value ? styles.activeMetricTab : ''}`}
                                    onClick={() => setSelectedMetric(m.value)}
                                >
                                    {m.label}
                                </button>
                            ))}
                        </div>
                        <select
                            className={styles.seasonSelect}
                            value={selectedSeason}
                            onChange={(e) => setSelectedSeason(e.target.value)}
                        >
                            {seasons.map((season) => (
                                <option key={season} value={season}>
                                    {formatSeason(season)}
                                </option>
                            ))}
                        </select>
                        {activeCategory === 'situational' && (
                            <select
                                className={styles.seasonSelect}
                                value={selectedPosition}
                                onChange={(e) => setSelectedPosition(e.target.value)}
                            >
                                <option value="All">All Positions</option>
                                <option value="F">Forward</option>
                                <option value="D">Defenseman</option>
                                <option value="L">Left Wing</option>
                                <option value="R">Right Wing</option>
                                <option value="C">Center</option>
                            </select>
                        )}
                    </div>
                    <div className={styles.qualifierRow}>
                        <span className={styles.qualifierLabel}>Min. Qualifier:</span>
                        {[
                            { label: 'All', toi: 0 },
                            { label: '10+ GP', toi: 3600 },
                            { label: '20+ GP', toi: 7200 },
                            { label: '30+ GP', toi: 10800 },
                        ].map(opt => (
                            <button
                                key={opt.label}
                                className={`${styles.qualifierBtn} ${minToi === opt.toi ? styles.activeQualifier : ''}`}
                                onClick={() => setMinToi(opt.toi)}
                            >
                                {opt.label}
                            </button>
                        ))}
                    </div>

                    {currentMetricInfo && (
                        <p className={styles.metricDescription}>
                            {currentMetricInfo.description}
                            <Link href={`/glossary#${activeCategory}`} className={styles.glossaryLink}>
                                Learn more
                            </Link>
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
                            {/* Top 3 Podium */}
                            <div className={styles.podium}>
                                {leaderboard.slice(0, 3).map((row, index) => (
                                    <Link
                                        key={row.player_id}
                                        href={`/players/${row.player_id}`}
                                        className={`${styles.podiumCard} ${styles[`rank${index + 1}`]}`}
                                    >
                                        <div className={styles.podiumRank}>
                                            {index === 0 ? '#1' : index === 1 ? '#2' : '#3'}
                                        </div>
                                        <div className={styles.podiumAvatar}>
                                            {(row.full_name || 'P').split(' ').map(n => n[0]).join('')}
                                        </div>
                                        <h3 className={styles.podiumName}>
                                            {row.full_name || `Player #${row.player_id}`}
                                        </h3>
                                        {activeCategory === 'situational' && (
                                            <div style={{ marginTop: '4px', marginBottom: '4px', display: 'flex', justifyContent: 'center' }}>
                                                <ReliabilityBadge isReliable={row.is_reliable ?? true} toiSeconds={row.total_toi_seconds} />
                                            </div>
                                        )}
                                        <div className={`${styles.podiumValue} ${row.value >= 0 ? styles.positive : styles.negative}`}>
                                            {row.value >= 0 ? '+' : ''}{row.value.toFixed(3)}
                                        </div>
                                    </Link>
                                ))}
                            </div>

                            {/* Full Table */}
                            <div className={styles.tableContainer}>
                                {leaderboard.slice(3).map((row, index) => (
                                    <Link
                                        key={row.player_id}
                                        href={`/players/${row.player_id}`}
                                        className={styles.tableRow}
                                    >
                                        <span className={styles.rank}>{index + 4}</span>
                                        <div className={styles.playerCell}>
                                            <div className={styles.playerAvatar}>
                                                {(row.full_name || 'P')[0]}
                                            </div>
                                            <span className={styles.playerName}>
                                                {row.full_name || `Player #${row.player_id}`}
                                            </span>
                                            {activeCategory === 'situational' && (
                                                <div style={{ marginLeft: '8px' }}>
                                                    <ReliabilityBadge isReliable={row.is_reliable ?? true} toiSeconds={row.total_toi_seconds} />
                                                </div>
                                            )}
                                        </div>
                                        <div className={styles.valueCell}>
                                            <div className={styles.valueBar}>
                                                <div
                                                    className={`${styles.valueBarFill} ${row.value >= 0 ? styles.barPositive : styles.barNegative}`}
                                                    style={{ width: `${(Math.abs(row.value) / maxAbsValue) * 100}%` }}
                                                />
                                            </div>
                                            <span className={`${styles.value} ${row.value >= 0 ? styles.positive : styles.negative}`}>
                                                {row.value >= 0 ? '+' : ''}{row.value.toFixed(3)}
                                            </span>
                                        </div>
                                    </Link>
                                ))}
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
