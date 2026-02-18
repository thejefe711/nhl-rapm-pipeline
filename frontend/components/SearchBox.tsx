'use client';

import { useState, useEffect, useCallback } from 'react';
import { searchPlayers, PlayerSearchResult } from '@/lib/api';
import styles from './SearchBox.module.css';

interface SearchBoxProps {
    onSelect: (player: PlayerSearchResult) => void;
    placeholder?: string;
    autoFocus?: boolean;
}

export default function SearchBox({ onSelect, placeholder = 'Search players...', autoFocus = false }: SearchBoxProps) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<PlayerSearchResult[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isOpen, setIsOpen] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(-1);

    const search = useCallback(async (q: string) => {
        if (q.length < 2) {
            setResults([]);
            return;
        }

        setIsLoading(true);
        try {
            const data = await searchPlayers(q);
            setResults(data.rows);
            setIsOpen(true);
            setSelectedIndex(-1);
        } catch (error) {
            console.error('Search error:', error);
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        const timeoutId = setTimeout(() => {
            search(query);
        }, 200);

        return () => clearTimeout(timeoutId);
    }, [query, search]);

    const handleSelect = (player: PlayerSearchResult) => {
        onSelect(player);
        setQuery('');
        setResults([]);
        setIsOpen(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (!isOpen || results.length === 0) return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSelectedIndex(prev => Math.min(prev + 1, results.length - 1));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSelectedIndex(prev => Math.max(prev - 1, 0));
        } else if (e.key === 'Enter' && selectedIndex >= 0) {
            e.preventDefault();
            handleSelect(results[selectedIndex]);
        } else if (e.key === 'Escape') {
            setIsOpen(false);
        }
    };

    return (
        <div className={styles.searchContainer}>
            <div className={styles.inputWrapper}>
                <svg className={styles.searchIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8" />
                    <path d="M21 21l-4.35-4.35" />
                </svg>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onFocus={() => results.length > 0 && setIsOpen(true)}
                    onBlur={() => setTimeout(() => setIsOpen(false), 200)}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    autoFocus={autoFocus}
                    className={styles.searchInput}
                />
                {isLoading && (
                    <div className={styles.loadingSpinner}>
                        <div className={styles.spinner} />
                    </div>
                )}
                {query && !isLoading && (
                    <button className={styles.clearButton} onClick={() => setQuery('')}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M18 6L6 18M6 6l12 12" />
                        </svg>
                    </button>
                )}
            </div>

            {isOpen && results.length > 0 && (
                <div className={styles.dropdown}>
                    <div className={styles.dropdownHeader}>
                        <span className={styles.resultCount}>{results.length} players found</span>
                    </div>
                    <ul className={styles.results}>
                        {results.map((player, index) => (
                            <li key={player.player_id}>
                                <button
                                    className={`${styles.resultItem} ${index === selectedIndex ? styles.selected : ''}`}
                                    onClick={() => handleSelect(player)}
                                    onMouseEnter={() => setSelectedIndex(index)}
                                >
                                    <div className={styles.playerAvatar}>
                                        {player.full_name.charAt(0)}
                                    </div>
                                    <div className={styles.playerInfo}>
                                        <span className={styles.playerName}>{player.full_name}</span>
                                        <span className={styles.playerId}>ID: {player.player_id}</span>
                                    </div>
                                    <svg className={styles.arrowIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M9 18l6-6-6-6" />
                                    </svg>
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {isOpen && query.length >= 2 && results.length === 0 && !isLoading && (
                <div className={styles.dropdown}>
                    <div className={styles.noResults}>
                        <p>No players found for &ldquo;{query}&rdquo;</p>
                    </div>
                </div>
            )}
        </div>
    );
}
