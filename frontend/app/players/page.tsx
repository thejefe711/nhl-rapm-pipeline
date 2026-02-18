'use client';

import { useRouter } from 'next/navigation';
import SearchBox from '@/components/SearchBox';
import { PlayerSearchResult } from '@/lib/api';
import styles from './page.module.css';

const FEATURED_PLAYERS = [
    { id: 8478402, name: 'Connor McDavid', team: 'EDM', position: 'C' },
    { id: 8479318, name: 'Auston Matthews', team: 'TOR', position: 'C' },
    { id: 8477934, name: 'Leon Draisaitl', team: 'EDM', position: 'C' },
    { id: 8477492, name: 'Nathan MacKinnon', team: 'COL', position: 'C' },
    { id: 8478483, name: 'Cale Makar', team: 'COL', position: 'D' },
    { id: 8480800, name: 'David Pastrnak', team: 'BOS', position: 'RW' },
];

export default function PlayersPage() {
    const router = useRouter();

    const handlePlayerSelect = (player: PlayerSearchResult) => {
        router.push(`/players/${player.player_id}`);
    };

    return (
        <div className={styles.page}>
            <div className="container">
                {/* Header */}
                <header className={styles.header}>
                    <div className={styles.headerBadge}>
                        <span>Player Analytics</span>
                    </div>
                    <h1>Find Any Player</h1>
                    <p className={styles.subtitle}>
                        Search 800+ NHL players and explore their RAPM analytics,
                        performance trends, and AI-powered insights.
                    </p>
                </header>

                {/* Search */}
                <section className={styles.searchSection}>
                    <SearchBox
                        onSelect={handlePlayerSelect}
                        placeholder="Search by player name..."
                        autoFocus
                    />
                    <p className={styles.searchHint}>
                        Type at least 2 characters to search
                    </p>
                </section>

                {/* Featured Players */}
                <section className={styles.featured}>
                    <div className={styles.featuredHeader}>
                        <h2>Featured Players</h2>
                        <span className={styles.featuredTag}>Popular</span>
                    </div>

                    <div className={styles.playerGrid}>
                        {FEATURED_PLAYERS.map((player) => (
                            <button
                                key={player.id}
                                className={styles.playerCard}
                                onClick={() => router.push(`/players/${player.id}`)}
                            >
                                <div className={styles.playerAvatar}>
                                    {player.name.split(' ').map(n => n[0]).join('')}
                                </div>
                                <div className={styles.playerInfo}>
                                    <span className={styles.playerName}>{player.name}</span>
                                    <span className={styles.playerMeta}>
                                        {player.team} • {player.position}
                                    </span>
                                </div>
                                <svg className={styles.chevron} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M9 18l6-6-6-6" />
                                </svg>
                            </button>
                        ))}
                    </div>
                </section>

                {/* Info Section */}
                <section className={styles.info}>
                    <div className={styles.infoCard}>
                        <div className={styles.infoIcon}>?</div>
                        <div className={styles.infoContent}>
                            <h3>What is RAPM?</h3>
                            <p>
                                <strong>Regularized Adjusted Plus-Minus</strong> isolates a player&apos;s
                                individual impact by controlling for linemates, competition, and zone starts.
                                A positive RAPM means the player improves their team&apos;s shot differential.
                            </p>
                            <ul className={styles.infoList}>
                                <li>Accounts for quality of linemates</li>
                                <li>Controls for strength of competition</li>
                                <li>Uses ridge regression for stability</li>
                                <li>5v5 even-strength focus</li>
                            </ul>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
}
