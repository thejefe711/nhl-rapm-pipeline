'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getStatsOverview, StatsOverview } from '@/lib/api';
import SearchBox from '@/components/SearchBox';
import styles from './page.module.css';

const FEATURES = [
    {
        title: 'RAPM Metrics',
        desc: 'Ridge regression isolates individual impact from linemates, competition, and zone starts.',
        href: '/leaderboards',
        cta: 'Explore Leaderboards',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 3v18h18"/><path d="M7 16l4-8 4 4 5-9"/></svg>
        ),
    },
    {
        title: 'Player Profiles',
        desc: 'Radar charts, percentile rankings, career trends, and 20+ metrics per player.',
        href: '/players',
        cta: 'Search Players',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="8" r="4"/><path d="M5 20v-1a7 7 0 0114 0v1"/></svg>
        ),
    },
    {
        title: 'Head-to-Head Compare',
        desc: 'Side-by-side comparison across multiple metrics. Find the edge between similar players.',
        href: '/compare',
        cta: 'Compare Players',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"/></svg>
        ),
    },
    {
        title: 'Glossary & Methodology',
        desc: 'Every metric explained. xG models, transition metrics, and our full analytical pipeline.',
        href: '/glossary',
        cta: 'Read Glossary',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
        ),
    },
    {
        title: 'AI Player Insights',
        desc: 'Latent skill embeddings and Kalman filter forecasts generate development trajectory insights.',
        href: '/players',
        cta: 'View Insights',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>
        ),
    },
    {
        title: 'Multi-Season Tracking',
        desc: 'Track player development, identify breakouts, and spot declining veterans across seasons.',
        href: '/leaderboards',
        cta: 'View Trends',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>
        ),
    },
];

const METRICS = [
    { label: 'Corsi RAPM', desc: 'Shot attempt differential per 60 minutes', href: '/leaderboards?metric=corsi_rapm_5v5' },
    { label: 'Expected Goals', desc: 'xG-weighted shot quality model', href: '/leaderboards?metric=xg_rapm_5v5' },
    { label: 'High Danger xG', desc: 'Highest quality scoring chances', href: '/leaderboards?metric=hd_xg_rapm_5v5_ge020' },
    { label: 'Transitions', desc: 'Impact after turnovers and takeaways', href: '/leaderboards?metric=takeaway_to_xg_swing_rapm_5v5_w10' },
    { label: 'Playmaking', desc: 'Primary assist generation & xG', href: '/leaderboards?metric=primary_assist_rapm_5v5' },
    { label: '20+ More', desc: 'Discipline, finishing, defense & more', href: '/glossary' },
];

export default function Home() {
    const router = useRouter();
    const [stats, setStats] = useState<StatsOverview | null>(null);

    useEffect(() => {
        getStatsOverview().then(setStats).catch(() => {});
    }, []);

    return (
        <div className={styles.page}>
            {/* ── Hero: Full-bleed with hockey photo ── */}
            <section className={styles.hero}>
                <div className={styles.heroOverlay} />
                <div className={`container ${styles.heroContainer}`}>
                    <div className={styles.heroContent}>
                        <div className={styles.heroBadge}>
                            <span className={styles.heroBadgeDot} />
                            Advanced Hockey Analytics
                        </div>

                        <h1 className={styles.heroTitle}>
                            The Deepest Player<br />
                            Analytics in Hockey.
                        </h1>

                        <p className={styles.heroDesc}>
                            RAPM, expected goals, high-danger metrics, and AI-powered
                            insights for 800+ NHL players across 6 seasons.
                        </p>

                        <div className={styles.heroSearch}>
                            <SearchBox
                                onSelect={(p) => router.push(`/players/${p.player_id}`)}
                                placeholder="Search any NHL player..."
                            />
                            <p className={styles.searchHint}>
                                Try McDavid, Matthews, Makar, or MacKinnon
                            </p>
                        </div>

                        <div className={styles.heroStats}>
                            <div className={styles.heroStat}>
                                <span className={styles.heroStatNum}>{stats?.seasons_count || '6'}</span>
                                <span className={styles.heroStatLbl}>Seasons</span>
                            </div>
                            <div className={styles.heroStatDiv} />
                            <div className={styles.heroStat}>
                                <span className={styles.heroStatNum}>{stats?.total_players ? `${stats.total_players}+` : '800+'}</span>
                                <span className={styles.heroStatLbl}>Players</span>
                            </div>
                            <div className={styles.heroStatDiv} />
                            <div className={styles.heroStat}>
                                <span className={styles.heroStatNum}>{stats?.total_metrics || '20+'}</span>
                                <span className={styles.heroStatLbl}>Metrics</span>
                            </div>
                            <div className={styles.heroStatDiv} />
                            <div className={styles.heroStat}>
                                <span className={styles.heroStatNum}>{stats?.total_games || '300+'}</span>
                                <span className={styles.heroStatLbl}>Games</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* ── Top Players: Light gray bg ── */}
            {stats?.top_players && stats.top_players.length > 0 && (
                <section className={styles.topPlayers}>
                    <div className="container">
                        <div className={styles.sectionHead}>
                            <div>
                                <p className={styles.eyebrow}>Leaderboard Preview</p>
                                <h2>
                                    Top Corsi RAPM {stats.latest_season
                                        ? <span className={styles.seasonTag}>{formatSeason(stats.latest_season)}</span>
                                        : null
                                    }
                                </h2>
                            </div>
                            <Link href="/leaderboards" className="btn btn-secondary">
                                All Rankings
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                            </Link>
                        </div>

                        <div className={styles.playerList}>
                            {stats.top_players.map((player, i) => (
                                <Link
                                    key={player.player_id}
                                    href={`/players/${player.player_id}`}
                                    className={styles.playerRow}
                                >
                                    <span className={styles.playerRank}>{String(i + 1).padStart(2, '0')}</span>
                                    <span className={styles.playerAvatar}>
                                        {(player.full_name || 'P').split(' ').map(n => n[0]).join('')}
                                    </span>
                                    <span className={styles.playerName}>
                                        {player.full_name || `Player #${player.player_id}`}
                                    </span>
                                    <span className={styles.playerDots} />
                                    <span className={`${styles.playerValue} ${player.value >= 0 ? styles.positive : styles.negative}`}>
                                        {player.value >= 0 ? '+' : ''}{player.value.toFixed(2)}
                                    </span>
                                    <svg className={styles.playerArrow} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                                </Link>
                            ))}
                        </div>
                    </div>
                </section>
            )}

            {/* ── Features: Green tinted bg ── */}
            <section className={styles.featuresTinted}>
                <div className="container">
                    <div className={styles.sectionHeadCenter}>
                        <p className={styles.eyebrow}>Capabilities</p>
                        <h2>Built for Serious Hockey Analysis.</h2>
                        <p className={styles.sectionSubtitle}>
                            Everything you need to evaluate NHL players at the highest level of analytical rigor.
                        </p>
                    </div>

                    <div className={styles.featureGrid}>
                        {FEATURES.map((f, i) => (
                            <Link key={i} href={f.href} className={styles.featureCard}>
                                <div className={styles.featureIconWrap}>{f.icon}</div>
                                <h3>{f.title}</h3>
                                <p>{f.desc}</p>
                                <span className={styles.featureCta}>{f.cta} &rarr;</span>
                            </Link>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Metrics: White ── */}
            <section className={styles.metrics}>
                <div className="container">
                    <div className={styles.sectionHead}>
                        <div>
                            <p className={styles.eyebrow}>What We Measure</p>
                            <h2>Beyond Traditional Stats.</h2>
                        </div>
                        <Link href="/glossary" className="btn btn-secondary">
                            Full Glossary
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                        </Link>
                    </div>

                    <div className={styles.metricGrid}>
                        {METRICS.map((m, i) => (
                            <Link key={i} href={m.href} className={styles.metricCard}>
                                <h4>{m.label}</h4>
                                <p>{m.desc}</p>
                            </Link>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── CTA: Dark ── */}
            <section className={styles.ctaDark}>
                <div className="container">
                    <div className={styles.ctaInner}>
                        <h2 className={styles.ctaTitle}>Ready to find undervalued players?</h2>
                        <p className={styles.ctaDesc}>
                            Search any player, explore the leaderboards, or dive into the methodology.
                        </p>
                        <div className={styles.ctaActions}>
                            <Link href="/players" className="btn btn-primary">
                                Search Players
                            </Link>
                            <Link href="/glossary" className={styles.ctaSecondary}>
                                Read Methodology &rarr;
                            </Link>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
}

function formatSeason(season: string): string {
    if (season.length === 8) {
        return `${season.slice(0, 4)}\u2013${season.slice(6, 8)}`;
    }
    return season;
}
