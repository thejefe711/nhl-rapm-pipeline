import Link from 'next/link';
import styles from './page.module.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Glossary & Methodology | Data-Puck.com',
    description: 'Understand every metric: RAPM, Corsi, Expected Goals (xG), High Danger scoring chances, playmaking metrics, and our analytical methodology.',
};

const METRIC_CATEGORIES = [
    {
        id: 'core',
        title: 'Core Metrics',
        description: 'The foundation of player evaluation. These metrics capture overall impact on shot and goal differentials.',
        metrics: [
            {
                name: 'Corsi RAPM (5v5)',
                key: 'corsi_rapm_5v5',
                description: 'Regularized Adjusted Plus-Minus based on Corsi (all shot attempts). Measures a player\'s isolated impact on their team\'s shot attempt differential per 60 minutes at 5-on-5.',
                interpretation: 'Positive = team generates more shot attempts and allows fewer when this player is on ice. A value of +0.50 means ~0.5 extra net shot attempts per 60 min.',
                goodRange: '> +0.30',
                eliteRange: '> +1.00',
            },
            {
                name: 'Expected Goals RAPM (5v5)',
                key: 'xg_rapm_5v5',
                description: 'RAPM based on Expected Goals (xG). Uses shot location, type, and game state to weight each shot attempt by its probability of becoming a goal.',
                interpretation: 'More predictive than raw Corsi because it accounts for shot quality. Positive values mean the player improves their team\'s expected goal differential.',
                goodRange: '> +0.05',
                eliteRange: '> +0.15',
            },
            {
                name: 'Goals RAPM (5v5)',
                key: 'goals_rapm_5v5',
                description: 'RAPM based on actual goals scored. Subject to more variance than xG-based metrics due to shooting percentage fluctuations.',
                interpretation: 'Captures finishing talent and goaltending effects. Compare to xG RAPM to identify overperformers (good finishers) vs. underperformers (bad luck or poor finishing).',
                goodRange: '> +0.03',
                eliteRange: '> +0.10',
            },
        ],
    },
    {
        id: 'offensive',
        title: 'Offensive Metrics',
        description: 'Isolate a player\'s impact on generating offense for their team.',
        metrics: [
            {
                name: 'Offensive Corsi',
                key: 'corsi_off_rapm_5v5',
                description: 'The offensive component of Corsi RAPM. Measures how many additional shot attempts a player generates for their team.',
                interpretation: 'Purely measures shot generation. High values indicate players who drive possession and create shooting opportunities.',
                goodRange: '> +0.20',
                eliteRange: '> +0.60',
            },
            {
                name: 'Offensive xG',
                key: 'xg_off_rapm_5v5',
                description: 'Quality-adjusted offensive impact. Weights shot attempts by their probability of becoming goals based on location and context.',
                interpretation: 'Better than raw Corsi for offense because it rewards players who generate high-quality chances, not just volume.',
                goodRange: '> +0.03',
                eliteRange: '> +0.08',
            },
        ],
    },
    {
        id: 'defensive',
        title: 'Defensive Metrics',
        description: 'Measure a player\'s ability to suppress opposition offense.',
        metrics: [
            {
                name: 'Defensive Corsi',
                key: 'corsi_def_rapm_5v5',
                description: 'The defensive component of Corsi RAPM. Measures how well a player suppresses opponent shot attempts.',
                interpretation: 'Positive values mean fewer opposition shots when on ice. Important for evaluating defensemen and defensive forwards.',
                goodRange: '> +0.15',
                eliteRange: '> +0.40',
            },
            {
                name: 'Defensive xG',
                key: 'xg_def_rapm_5v5',
                description: 'Quality-adjusted defensive impact. Positive values mean the player reduces the quality of opposition scoring chances.',
                interpretation: 'Captures not just shot suppression but whether a player pushes opponents to lower-danger areas.',
                goodRange: '> +0.02',
                eliteRange: '> +0.06',
            },
        ],
    },
    {
        id: 'high-danger',
        title: 'High Danger Metrics',
        description: 'Focus on high-quality scoring chances (xG >= 0.20 per shot). These are the most impactful events in hockey.',
        metrics: [
            {
                name: 'High Danger xG',
                key: 'hd_xg_rapm_5v5_ge020',
                description: 'Net impact on high-danger expected goals (shots with xG >= 0.20). Combines both offensive generation and defensive suppression of dangerous chances.',
                interpretation: 'The most "real hockey" metric. Captures a player\'s involvement in the chances that actually decide games.',
                goodRange: '> +0.02',
                eliteRange: '> +0.06',
            },
            {
                name: 'HD Offensive xG',
                key: 'hd_xg_off_rapm_5v5_ge020',
                description: 'Offensive generation of high-danger chances only. Identifies players who create the most dangerous scoring opportunities.',
                interpretation: 'Net-front presence, slot access, and rush generation all contribute. High values indicate players who manufacture the best chances.',
                goodRange: '> +0.01',
                eliteRange: '> +0.04',
            },
            {
                name: 'HD Defensive xG',
                key: 'hd_xg_def_rapm_5v5_ge020',
                description: 'Suppression of high-danger chances against. Measures a player\'s ability to keep opponents out of dangerous areas.',
                interpretation: 'Critical for defensemen evaluation. Positive means fewer high-danger chances against when on ice.',
                goodRange: '> +0.01',
                eliteRange: '> +0.03',
            },
        ],
    },
    {
        id: 'transitions',
        title: 'Transition Metrics',
        description: 'Capture the xG impact in the 10 seconds following key transition events. Pioneering metrics unique to this platform.',
        metrics: [
            {
                name: 'Takeaway xG Swing',
                key: 'takeaway_to_xg_swing_rapm_5v5_w10',
                description: 'Measures the xG differential in the 10 seconds after a takeaway. Positive values mean the player\'s takeaways lead to better scoring chances.',
                interpretation: 'Identifies players whose takeaways are truly dangerous - they not only steal the puck but convert it into offense.',
                goodRange: '> +0.01',
                eliteRange: '> +0.03',
            },
            {
                name: 'Giveaway xG Swing',
                key: 'giveaway_to_xg_swing_rapm_5v5_w10',
                description: 'Measures the xG differential after giveaways. More negative = more costly turnovers.',
                interpretation: 'A negative value means the player\'s giveaways lead to dangerous chances against. Critical for evaluating puck management.',
                goodRange: '> -0.01',
                eliteRange: '> 0.00',
            },
            {
                name: 'Turnover xG Swing',
                key: 'turnover_to_xg_swing_rapm_5v5_w10',
                description: 'Combined metric capturing the net xG impact of all turnovers (takeaways + giveaways) in a 10-second window.',
                interpretation: 'The definitive transition metric. Positive values mean the player wins the turnover battle in terms of dangerous chances created vs. allowed.',
                goodRange: '> +0.01',
                eliteRange: '> +0.03',
            },
        ],
    },
    {
        id: 'playmaking',
        title: 'Playmaking Metrics',
        description: 'Measure a player\'s ability to set up goals through assists.',
        metrics: [
            {
                name: 'Primary Assists RAPM',
                key: 'primary_assist_rapm_5v5',
                description: 'RAPM-adjusted primary assist generation rate. Primary assists (the last pass before a goal) are the most repeatable assist type.',
                interpretation: 'High values identify true playmakers who consistently set up goals. More predictive than total assists.',
                goodRange: '> +0.01',
                eliteRange: '> +0.03',
            },
            {
                name: 'Secondary Assists RAPM',
                key: 'secondary_assist_rapm_5v5',
                description: 'RAPM-adjusted secondary assist generation. Secondary assists (two passes before a goal) are less repeatable and more subject to luck.',
                interpretation: 'Less stable than primary assists but can identify players who start plays that lead to goals.',
                goodRange: '> +0.005',
                eliteRange: '> +0.015',
            },
            {
                name: 'Primary Assist xG',
                key: 'xg_primary_assist_on_goals_rapm_5v5',
                description: 'Expected goals credit for the shots that primary assists lead to. Measures the quality of chances created, not just whether they score.',
                interpretation: 'More stable than counting goals. Identifies players who create high-quality chances for their teammates regardless of finishing.',
                goodRange: '> +0.01',
                eliteRange: '> +0.03',
            },
        ],
    },
    {
        id: 'discipline',
        title: 'Discipline Metrics',
        description: 'Special teams impact through penalty drawing and avoidance.',
        metrics: [
            {
                name: 'Penalties Drawn',
                key: 'penalties_drawn_rapm_5v5',
                description: 'Rate of drawing penalties at 5v5. Higher values mean the player frequently puts their team on the power play.',
                interpretation: 'Players who draw penalties provide hidden value. A power play is worth ~0.08 expected goals, so high values here add significant team value.',
                goodRange: '> +0.01',
                eliteRange: '> +0.03',
            },
            {
                name: 'Penalties Taken',
                key: 'penalties_committed_rapm_5v5',
                description: 'Rate of taking penalties at 5v5. Lower (more negative) values indicate a player who costs their team power plays.',
                interpretation: 'Negative values are bad - the player sends their team to the penalty kill frequently. This directly costs expected goals.',
                goodRange: '> -0.01',
                eliteRange: '> 0.00',
            },
        ],
    },
];

const FAQ_ITEMS = [
    {
        q: 'What is RAPM and why is it better than +/-?',
        a: 'RAPM (Regularized Adjusted Plus-Minus) uses ridge regression to separate individual player contributions from their linemates and opponents. Traditional +/- gives equal credit to all 5 skaters, making it heavily influenced by linemate quality. RAPM solves this by building a model that accounts for every player on the ice simultaneously.',
    },
    {
        q: 'What does "5v5" mean in the metric names?',
        a: '5v5 (five-on-five) refers to even-strength play - when both teams have 5 skaters on the ice. We focus on 5v5 because it represents the majority of game time and removes the confounding effects of power play/penalty kill deployments. Special teams metrics are available separately.',
    },
    {
        q: 'How is the xG (Expected Goals) model built?',
        a: 'Our xG model assigns a probability (0 to 1) to each shot based on: shot location (distance and angle), shot type (wrist, slap, backhand, etc.), whether it was a rebound or rush, game state, and whether the net was empty. It is trained on historical NHL play-by-play data.',
    },
    {
        q: 'What does the "per 60" rate mean?',
        a: 'All RAPM values are expressed per 60 minutes of ice time. This normalizes for different deployment levels. A player who plays 22 minutes/game is compared fairly to one who plays 14 minutes/game.',
    },
    {
        q: 'Why do some metrics have small values (like 0.05)?',
        a: 'The regularization in RAPM intentionally shrinks estimates toward zero to reduce noise. This means the values are conservative - a player with +0.10 xG RAPM truly does have significant positive impact. Small differences between players are meaningful.',
    },
    {
        q: 'How many games are needed for reliable RAPM estimates?',
        a: 'We require a minimum of 600 seconds of 5v5 ice time for inclusion. Generally, 30+ games provides reasonable estimates, and a full 82-game season provides the most reliable numbers. We display data quality warnings when sample sizes are limited.',
    },
    {
        q: 'What are the latent skill dimensions and DLM forecasts?',
        a: 'We use a Sparse Autoencoder (SAE) to decompose RAPM metrics into latent skill dimensions - hidden factors that explain player performance patterns. A Dynamic Linear Model (Kalman filter) then forecasts how these skills evolve over time, enabling player development tracking and projections.',
    },
    {
        q: 'How often is the data updated?',
        a: 'Data is refreshed from the NHL API regularly. The pipeline processes raw play-by-play and shift data, validates on-ice reconstructions against official boxscores, and recomputes all RAPM metrics. Check the "Live" indicator in the navigation for current status.',
    },
];

export default function GlossaryPage() {
    return (
        <div className={styles.page}>
            <div className="container">
                {/* Hero */}
                <header className={styles.header}>
                    <div className={styles.headerBadge}>
                        <span>Reference</span>
                    </div>
                    <h1>Glossary &amp; <span className="text-gradient">Methodology</span></h1>
                    <p className={styles.subtitle}>
                        Everything you need to understand our analytics. Comprehensive definitions
                        for every metric, our modeling approach, and frequently asked questions.
                    </p>
                </header>

                {/* Table of Contents */}
                <nav className={styles.toc}>
                    <h3 className={styles.tocTitle}>Categories</h3>
                    <div className={styles.tocGrid}>
                        {METRIC_CATEGORIES.map(cat => (
                            <a key={cat.id} href={`#${cat.id}`} className={styles.tocItem}>
                                <span className={styles.tocLabel}>{cat.title}</span>
                                <span className={styles.tocCount}>{cat.metrics.length} metrics</span>
                            </a>
                        ))}
                        <a href="#methodology" className={styles.tocItem}>
                            <span className={styles.tocLabel}>Methodology</span>
                            <span className={styles.tocCount}>Deep dive</span>
                        </a>
                        <a href="#faq" className={styles.tocItem}>
                            <span className={styles.tocLabel}>FAQ</span>
                            <span className={styles.tocCount}>{FAQ_ITEMS.length} questions</span>
                        </a>
                    </div>
                </nav>

                {/* Metric Categories */}
                {METRIC_CATEGORIES.map(cat => (
                    <section key={cat.id} id={cat.id} className={styles.category}>
                        <div className={styles.categoryHeader}>
                            <h2>{cat.title}</h2>
                            <p>{cat.description}</p>
                        </div>

                        <div className={styles.metricList}>
                            {cat.metrics.map(metric => (
                                <div key={metric.key} className={styles.metricCard}>
                                    <div className={styles.metricHeader}>
                                        <h3>{metric.name}</h3>
                                        <code className={styles.metricKey}>{metric.key}</code>
                                    </div>
                                    <p className={styles.metricDesc}>{metric.description}</p>

                                    <div className={styles.metricDetails}>
                                        <div className={styles.detailRow}>
                                            <span className={styles.detailLabel}>Interpretation</span>
                                            <p className={styles.detailValue}>{metric.interpretation}</p>
                                        </div>
                                        <div className={styles.thresholds}>
                                            <div className={styles.threshold}>
                                                <span className={styles.thresholdLabel}>Good</span>
                                                <span className={styles.thresholdValue}>{metric.goodRange}</span>
                                            </div>
                                            <div className={styles.threshold}>
                                                <span className={styles.thresholdLabel}>Elite</span>
                                                <span className={styles.thresholdElite}>{metric.eliteRange}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <Link
                                        href={`/leaderboards?metric=${metric.key}`}
                                        className={styles.metricLink}
                                    >
                                        View Leaderboard &rarr;
                                    </Link>
                                </div>
                            ))}
                        </div>
                    </section>
                ))}

                {/* Methodology */}
                <section id="methodology" className={styles.methodology}>
                    <div className={styles.categoryHeader}>
                        <h2>Methodology</h2>
                        <p>How we build our analytics from raw NHL data to final player evaluations.</p>
                    </div>

                    <div className={styles.pipelineSteps}>
                        <div className={styles.step}>
                            <div className={styles.stepNumber}>1</div>
                            <div className={styles.stepContent}>
                                <h3>Data Ingestion</h3>
                                <p>
                                    We fetch play-by-play event data and shift charts from the NHL&apos;s public API
                                    for every game. This includes shot locations, event types, player shifts with
                                    second-level granularity, and boxscore validation data.
                                </p>
                            </div>
                        </div>

                        <div className={styles.step}>
                            <div className={styles.stepNumber}>2</div>
                            <div className={styles.stepContent}>
                                <h3>On-Ice Reconstruction</h3>
                                <p>
                                    We reconstruct which players were on the ice for every event by overlapping
                                    shift data with play-by-play timestamps. This handles boundary-second issues,
                                    goalie identification from boxscores, and strength-state classification.
                                </p>
                            </div>
                        </div>

                        <div className={styles.step}>
                            <div className={styles.stepNumber}>3</div>
                            <div className={styles.stepContent}>
                                <h3>Validation Gates</h3>
                                <p>
                                    Every game passes two validation gates before inclusion: Gate 1 verifies shift
                                    integrity and data completeness. Gate 2 validates that our on-ice reconstructions
                                    match official NHL +/- records, ensuring data accuracy.
                                </p>
                            </div>
                        </div>

                        <div className={styles.step}>
                            <div className={styles.stepNumber}>4</div>
                            <div className={styles.stepContent}>
                                <h3>Expected Goals Model</h3>
                                <p>
                                    Each shot is scored with an xG probability based on distance, angle, shot type,
                                    rebound status, rush context, and game state. The model is trained on historical
                                    NHL data and versioned for reproducibility.
                                </p>
                            </div>
                        </div>

                        <div className={styles.step}>
                            <div className={styles.stepNumber}>5</div>
                            <div className={styles.stepContent}>
                                <h3>RAPM Computation</h3>
                                <p>
                                    We build stint-level design matrices where each row is a continuous segment of play
                                    with the same 10 skaters on ice. Ridge regression isolates individual player effects
                                    while controlling for all linemates and opponents. Cross-validated alpha selection
                                    ensures optimal regularization strength.
                                </p>
                            </div>
                        </div>

                        <div className={styles.step}>
                            <div className={styles.stepNumber}>6</div>
                            <div className={styles.stepContent}>
                                <h3>Latent Skills &amp; Forecasting</h3>
                                <p>
                                    A Sparse Autoencoder decomposes the RAPM metric space into interpretable latent
                                    dimensions. A Dynamic Linear Model (Kalman filter) tracks how these latent skills
                                    evolve game-by-game, enabling real-time development tracking and multi-game-ahead
                                    forecasting.
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Data Sources */}
                <section id="data-sources" className={styles.dataSources}>
                    <div className={styles.categoryHeader}>
                        <h2>Data Sources</h2>
                        <p>Transparency about where our data comes from and coverage.</p>
                    </div>

                    <div className={styles.sourceGrid}>
                        <div className={styles.sourceCard}>
                            <h4>NHL API</h4>
                            <p>Play-by-play events, shift charts, boxscores, and schedule data from the NHL&apos;s publicly available API endpoints.</p>
                        </div>
                        <div className={styles.sourceCard}>
                            <h4>Coverage</h4>
                            <p>6 NHL seasons (2020-21 through 2025-26) with 800+ unique players and 300+ validated games per season.</p>
                        </div>
                        <div className={styles.sourceCard}>
                            <h4>Validation Rate</h4>
                            <p>Our on-ice reconstruction matches official NHL +/- data for 95%+ of player-games, ensuring high data accuracy.</p>
                        </div>
                    </div>
                </section>

                {/* FAQ */}
                <section id="faq" className={styles.faq}>
                    <div className={styles.categoryHeader}>
                        <h2>Frequently Asked Questions</h2>
                        <p>Common questions about our analytics and methodology.</p>
                    </div>

                    <div className={styles.faqList}>
                        {FAQ_ITEMS.map((item, i) => (
                            <details key={i} className={styles.faqItem}>
                                <summary className={styles.faqQuestion}>{item.q}</summary>
                                <p className={styles.faqAnswer}>{item.a}</p>
                            </details>
                        ))}
                    </div>
                </section>
            </div>
        </div>
    );
}
