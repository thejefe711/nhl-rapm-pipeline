import Link from 'next/link';
import styles from './Footer.module.css';

export default function Footer() {
    return (
        <footer className={styles.footer}>
            <div className={styles.container}>
                <div className={styles.grid}>
                    {/* Brand Column */}
                    <div className={styles.brand}>
                        <Link href="/" className={styles.logo}>
                            <svg className={styles.logoIcon} viewBox="0 0 16 16" fill="currentColor" width="16" height="16">
                                <ellipse cx="8" cy="8" rx="7" ry="5.5" />
                            </svg>
                            <div>
                                <span className={styles.logoPrimary}>Data-Puck</span>
                                <span className={styles.logoSecondary}>.com</span>
                            </div>
                        </Link>
                        <p className={styles.brandDesc}>
                            Advanced hockey analytics powered by RAPM, expected goals models,
                            latent skill embeddings, and dynamic forecasting.
                        </p>
                        <div className={styles.techStack}>
                            <span className={styles.techBadge}>RAPM</span>
                            <span className={styles.techBadge}>xG Model</span>
                            <span className={styles.techBadge}>DLM Forecast</span>
                            <span className={styles.techBadge}>SAE Embeddings</span>
                        </div>
                    </div>

                    {/* Navigate */}
                    <div className={styles.column}>
                        <h4 className={styles.columnTitle}>Explore</h4>
                        <ul className={styles.links}>
                            <li><Link href="/players">Player Search</Link></li>
                            <li><Link href="/leaderboards">Leaderboards</Link></li>
                            <li><Link href="/compare">Compare Players</Link></li>
                            <li><Link href="/glossary">Glossary &amp; Methodology</Link></li>
                        </ul>
                    </div>

                    {/* Metrics */}
                    <div className={styles.column}>
                        <h4 className={styles.columnTitle}>Key Metrics</h4>
                        <ul className={styles.links}>
                            <li><Link href="/leaderboards?metric=corsi_rapm_5v5">Corsi RAPM</Link></li>
                            <li><Link href="/leaderboards?metric=xg_rapm_5v5">Expected Goals RAPM</Link></li>
                            <li><Link href="/leaderboards?metric=hd_xg_rapm_5v5_ge020">High Danger xG</Link></li>
                            <li><Link href="/leaderboards?metric=primary_assist_rapm_5v5">Playmaking RAPM</Link></li>
                        </ul>
                    </div>

                    {/* About */}
                    <div className={styles.column}>
                        <h4 className={styles.columnTitle}>About</h4>
                        <ul className={styles.links}>
                            <li><Link href="/glossary#methodology">Methodology</Link></li>
                            <li><Link href="/glossary#faq">FAQ</Link></li>
                            <li><Link href="/glossary#data-sources">Data Sources</Link></li>
                        </ul>
                        <div className={styles.dataInfo}>
                            <div className={styles.dataInfoDot} />
                            <span>Data refreshed regularly from NHL API</span>
                        </div>
                    </div>
                </div>

                <div className={styles.bottom}>
                    <p className={styles.copyright}>
                        Data-Puck.com &mdash; Independent analytics project. Not affiliated with the NHL.
                    </p>
                    <p className={styles.disclaimer}>
                        All data sourced from publicly available NHL API endpoints. Player names and team information are property of the NHL.
                    </p>
                </div>
            </div>
        </footer>
    );
}
