import Link from 'next/link';
import styles from './page.module.css';

export default function Home() {
    return (
        <div className={styles.page}>
            {/* Hero Section */}
            <section className={styles.hero}>
                <div className={styles.heroGlow} />
                <div className="container">
                    <div className={styles.heroContent}>
                        <div className={styles.heroBadge}>
                            <span className={styles.heroBadgeDot} />
                            <span>Powered by Advanced Analytics</span>
                        </div>

                        <h1 className={styles.heroTitle}>
                            Unlock <span className="text-gradient">Player Value</span>
                            <br />
                            with RAPM Analytics
                        </h1>

                        <p className={styles.heroSubtitle}>
                            Go beyond traditional stats. Our Regularized Adjusted Plus-Minus
                            model isolates individual player impact from linemates, competition,
                            and situational factors.
                        </p>

                        <div className={styles.heroActions}>
                            <Link href="/players" className="btn btn-primary">
                                <span>Search Players</span>
                                <span className={styles.btnArrow}>→</span>
                            </Link>
                            <Link href="/leaderboards" className="btn btn-secondary">
                                View Leaderboards
                            </Link>
                        </div>

                        {/* Hero Stats */}
                        <div className={styles.heroStats}>
                            <div className={styles.heroStat}>
                                <div className={styles.heroStatValue}>6</div>
                                <div className={styles.heroStatLabel}>Seasons</div>
                            </div>
                            <div className={styles.heroStatDivider} />
                            <div className={styles.heroStat}>
                                <div className={styles.heroStatValue}>800+</div>
                                <div className={styles.heroStatLabel}>Players</div>
                            </div>
                            <div className={styles.heroStatDivider} />
                            <div className={styles.heroStat}>
                                <div className={styles.heroStatValue}>15+</div>
                                <div className={styles.heroStatLabel}>Metrics</div>
                            </div>
                            <div className={styles.heroStatDivider} />
                            <div className={styles.heroStat}>
                                <div className={styles.heroStatValue}>5v5</div>
                                <div className={styles.heroStatLabel}>Focus</div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className={styles.features}>
                <div className="container">
                    <div className={styles.sectionHeader}>
                        <span className={styles.sectionTag}>Features</span>
                        <h2>Everything You Need for<br />Deep Hockey Analysis</h2>
                    </div>

                    <div className={styles.featureGrid}>
                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>
                                <span>📊</span>
                            </div>
                            <h3>RAPM Metrics</h3>
                            <p>
                                Regularized Adjusted Plus-Minus isolates individual impact
                                from linemate effects using ridge regression.
                            </p>
                            <div className={styles.featureHighlight}>
                                <span className={styles.featureTag}>Core Feature</span>
                            </div>
                        </div>

                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>
                                <span>🔍</span>
                            </div>
                            <h3>Player Search</h3>
                            <p>
                                Instant search across 800+ players with autocomplete.
                                View detailed profiles and multi-season trends.
                            </p>
                        </div>

                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>
                                <span>🏆</span>
                            </div>
                            <h3>Leaderboards</h3>
                            <p>
                                Filter by season and metric type. See who dominates
                                in Corsi, xG, offense, defense, and more.
                            </p>
                        </div>

                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>
                                <span>⚔️</span>
                            </div>
                            <h3>Compare</h3>
                            <p>
                                Side-by-side player comparisons. Overlay stats to find
                                the edge between two similar players.
                            </p>
                        </div>

                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>
                                <span>🤖</span>
                            </div>
                            <h3>AI Insights</h3>
                            <p>
                                Natural language explanations of player skills,
                                development trajectory, and forecasted performance.
                            </p>
                            <div className={styles.featureHighlight}>
                                <span className={styles.featureTag}>AI Powered</span>
                            </div>
                        </div>

                        <div className={styles.featureCard}>
                            <div className={styles.featureIcon}>
                                <span>📈</span>
                            </div>
                            <h3>Multi-Season</h3>
                            <p>
                                Track player development across 6+ seasons. Identify
                                rising stars and declining veterans.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className={styles.cta}>
                <div className="container">
                    <div className={styles.ctaCard}>
                        <div className={styles.ctaContent}>
                            <h2>Ready to discover undervalued players?</h2>
                            <p>Start exploring the most comprehensive RAPM database in hockey analytics.</p>
                        </div>
                        <Link href="/players" className="btn btn-primary">
                            Get Started
                            <span className={styles.btnArrow}>→</span>
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    );
}
