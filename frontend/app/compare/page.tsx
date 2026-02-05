'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import SearchBox from '@/components/SearchBox';
import { PlayerSearchResult, getPlayerRAPM, RAPMRow } from '@/lib/api';
import styles from './page.module.css';

interface SelectedPlayer {
    id: number;
    name: string;
    rapm: RAPMRow[];
}

export default function ComparePage() {
    const router = useRouter();
    const [player1, setPlayer1] = useState<SelectedPlayer | null>(null);
    const [player2, setPlayer2] = useState<SelectedPlayer | null>(null);
    const [isLoading, setIsLoading] = useState<1 | 2 | null>(null);

    const loadPlayer = async (player: PlayerSearchResult, slot: 1 | 2) => {
        setIsLoading(slot);
        try {
            const rapmData = await getPlayerRAPM(player.player_id);
            const selectedPlayer: SelectedPlayer = {
                id: player.player_id,
                name: player.full_name,
                rapm: rapmData.rows,
            };

            if (slot === 1) {
                setPlayer1(selectedPlayer);
            } else {
                setPlayer2(selectedPlayer);
            }
        } catch (err) {
            console.error('Failed to load player:', err);
        } finally {
            setIsLoading(null);
        }
    };

    const commonSeasons = player1 && player2
        ? player1.rapm
            .map(r => r.season)
            .filter(season => player2.rapm.some(r => r.season === season))
            .sort()
            .reverse()
        : [];

    const getBetterPlayer = () => {
        if (!player1 || !player2 || commonSeasons.length === 0) return null;

        const p1Avg = commonSeasons.reduce((sum, season) =>
            sum + (player1.rapm.find(r => r.season === season)?.value || 0), 0) / commonSeasons.length;
        const p2Avg = commonSeasons.reduce((sum, season) =>
            sum + (player2.rapm.find(r => r.season === season)?.value || 0), 0) / commonSeasons.length;

        return p1Avg > p2Avg ? 1 : p2Avg > p1Avg ? 2 : null;
    };

    const winner = getBetterPlayer();

    return (
        <div className={styles.page}>
            <div className="container">
                <header className={styles.header}>
                    <div className={styles.headerBadge}>
                        <span>Head to Head</span>
                    </div>
                    <h1>Compare Players</h1>
                    <p className={styles.subtitle}>
                        See how two players stack up against each other
                    </p>
                </header>

                {/* Player Selection */}
                <section className={styles.selection}>
                    <div className={`${styles.playerSlot} ${player1 ? styles.hasPlayer : ''}`}>
                        <div className={styles.slotLabel}>Player 1</div>

                        {!player1 ? (
                            <div className={styles.slotContent}>
                                <SearchBox
                                    onSelect={(p) => loadPlayer(p, 1)}
                                    placeholder="Search first player..."
                                />
                                {isLoading === 1 && (
                                    <div className={styles.loadingState}>
                                        <div className="spinner" />
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className={styles.selectedPlayer}>
                                <div className={styles.playerAvatar}>
                                    {player1.name.split(' ').map(n => n[0]).join('')}
                                </div>
                                <h3>{player1.name}</h3>
                                <button
                                    className={styles.removeBtn}
                                    onClick={() => setPlayer1(null)}
                                >
                                    Remove
                                </button>
                            </div>
                        )}
                    </div>

                    <div className={styles.versus}>
                        <span>VS</span>
                    </div>

                    <div className={`${styles.playerSlot} ${player2 ? styles.hasPlayer : ''}`}>
                        <div className={styles.slotLabel}>Player 2</div>

                        {!player2 ? (
                            <div className={styles.slotContent}>
                                <SearchBox
                                    onSelect={(p) => loadPlayer(p, 2)}
                                    placeholder="Search second player..."
                                />
                                {isLoading === 2 && (
                                    <div className={styles.loadingState}>
                                        <div className="spinner" />
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className={styles.selectedPlayer}>
                                <div className={styles.playerAvatar}>
                                    {player2.name.split(' ').map(n => n[0]).join('')}
                                </div>
                                <h3>{player2.name}</h3>
                                <button
                                    className={styles.removeBtn}
                                    onClick={() => setPlayer2(null)}
                                >
                                    Remove
                                </button>
                            </div>
                        )}
                    </div>
                </section>

                {/* Comparison Results */}
                {player1 && player2 && commonSeasons.length > 0 && (
                    <section className={styles.comparison}>
                        <div className={styles.comparisonHeader}>
                            <h2>Corsi RAPM 5v5 Comparison</h2>
                            {winner && (
                                <div className={styles.winnerBadge}>
                                    🏆 {winner === 1 ? player1.name : player2.name} leads
                                </div>
                            )}
                        </div>

                        <div className={styles.comparisonTable}>
                            <div className={styles.tableHeader}>
                                <div className={`${styles.colPlayer} ${winner === 1 ? styles.winner : ''}`}>
                                    {player1.name}
                                </div>
                                <div className={styles.colSeason}>Season</div>
                                <div className={`${styles.colPlayer} ${winner === 2 ? styles.winner : ''}`}>
                                    {player2.name}
                                </div>
                            </div>

                            {commonSeasons.map((season) => {
                                const p1Value = player1.rapm.find(r => r.season === season)?.value ?? 0;
                                const p2Value = player2.rapm.find(r => r.season === season)?.value ?? 0;
                                const p1Better = p1Value > p2Value;
                                const maxVal = Math.max(Math.abs(p1Value), Math.abs(p2Value));

                                return (
                                    <div key={season} className={styles.tableRow}>
                                        <div className={styles.colPlayer}>
                                            <div className={styles.barContainer}>
                                                <div
                                                    className={`${styles.bar} ${styles.barLeft} ${p1Better ? styles.barWinner : ''}`}
                                                    style={{ width: `${maxVal > 0 ? (Math.abs(p1Value) / maxVal) * 100 : 0}%` }}
                                                />
                                            </div>
                                            <span className={`${styles.value} ${p1Better ? styles.highlight : ''}`}>
                                                {p1Value >= 0 ? '+' : ''}{p1Value.toFixed(2)}
                                            </span>
                                        </div>
                                        <div className={styles.colSeason}>{formatSeason(season)}</div>
                                        <div className={styles.colPlayer}>
                                            <span className={`${styles.value} ${!p1Better ? styles.highlight : ''}`}>
                                                {p2Value >= 0 ? '+' : ''}{p2Value.toFixed(2)}
                                            </span>
                                            <div className={styles.barContainer}>
                                                <div
                                                    className={`${styles.bar} ${styles.barRight} ${!p1Better ? styles.barWinner : ''}`}
                                                    style={{ width: `${maxVal > 0 ? (Math.abs(p2Value) / maxVal) * 100 : 0}%` }}
                                                />
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        <div className={styles.actions}>
                            <button
                                className="btn btn-secondary"
                                onClick={() => router.push(`/players/${player1.id}`)}
                            >
                                View {player1.name} →
                            </button>
                            <button
                                className="btn btn-secondary"
                                onClick={() => router.push(`/players/${player2.id}`)}
                            >
                                View {player2.name} →
                            </button>
                        </div>
                    </section>
                )}

                {player1 && player2 && commonSeasons.length === 0 && (
                    <div className={styles.noOverlap}>
                        <div className={styles.noOverlapIcon}>📅</div>
                        <h3>No Overlapping Seasons</h3>
                        <p>These players don&apos;t have data for the same seasons.</p>
                    </div>
                )}

                {(!player1 || !player2) && (
                    <div className={styles.prompt}>
                        <div className={styles.promptIcon}>⚔️</div>
                        <h3>Select Two Players</h3>
                        <p>Search and select two players above to compare their stats.</p>
                    </div>
                )}
            </div>
        </div>
    );
}

function formatSeason(season: string): string {
    if (season.length === 8) {
        return `${season.slice(2, 4)}-${season.slice(6, 8)}`;
    }
    return season;
}
