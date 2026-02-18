'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { searchPlayers, PlayerSearchResult } from '@/lib/api';
import styles from './CommandPalette.module.css';

const QUICK_LINKS = [
    { label: 'Leaderboards', path: '/leaderboards', icon: 'chart' },
    { label: 'Compare Players', path: '/compare', icon: 'compare' },
    { label: 'Glossary & Methodology', path: '/glossary', icon: 'book' },
    { label: 'Player Search', path: '/players', icon: 'search' },
];

export default function CommandPalette() {
    const [isOpen, setIsOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<PlayerSearchResult[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);
    const router = useRouter();

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                setIsOpen(prev => !prev);
            }
            if (e.key === 'Escape') {
                setIsOpen(false);
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, []);

    useEffect(() => {
        if (isOpen) {
            setQuery('');
            setResults([]);
            setSelectedIndex(0);
            setTimeout(() => inputRef.current?.focus(), 50);
        }
    }, [isOpen]);

    const search = useCallback(async (q: string) => {
        if (q.length < 2) {
            setResults([]);
            return;
        }
        setIsLoading(true);
        try {
            const data = await searchPlayers(q, 8);
            setResults(data.rows);
            setSelectedIndex(0);
        } catch {
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        const t = setTimeout(() => search(query), 150);
        return () => clearTimeout(t);
    }, [query, search]);

    const totalItems = query.length < 2
        ? QUICK_LINKS.length
        : results.length;

    const handleSelect = (index: number) => {
        if (query.length < 2) {
            router.push(QUICK_LINKS[index].path);
        } else if (results[index]) {
            router.push(`/players/${results[index].player_id}`);
        }
        setIsOpen(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSelectedIndex(prev => Math.min(prev + 1, totalItems - 1));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSelectedIndex(prev => Math.max(prev - 1, 0));
        } else if (e.key === 'Enter') {
            e.preventDefault();
            handleSelect(selectedIndex);
        }
    };

    if (!isOpen) return null;

    return (
        <div className={styles.overlay} onClick={() => setIsOpen(false)}>
            <div className={styles.palette} onClick={e => e.stopPropagation()}>
                <div className={styles.inputRow}>
                    <svg className={styles.searchIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" />
                        <path d="M21 21l-4.35-4.35" />
                    </svg>
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Search players, navigate pages..."
                        className={styles.input}
                    />
                    <kbd className={styles.kbd}>ESC</kbd>
                </div>

                <div className={styles.results}>
                    {query.length < 2 ? (
                        <>
                            <div className={styles.sectionLabel}>Quick Navigation</div>
                            {QUICK_LINKS.map((link, i) => (
                                <button
                                    key={link.path}
                                    className={`${styles.resultItem} ${i === selectedIndex ? styles.selected : ''}`}
                                    onClick={() => handleSelect(i)}
                                    onMouseEnter={() => setSelectedIndex(i)}
                                >
                                    <span className={styles.resultIcon}>
                                        {link.icon === 'chart' && (
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 20V10M12 20V4M6 20v-6" /></svg>
                                        )}
                                        {link.icon === 'compare' && (
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5" /></svg>
                                        )}
                                        {link.icon === 'book' && (
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2zM22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z" /></svg>
                                        )}
                                        {link.icon === 'search' && (
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" /></svg>
                                        )}
                                    </span>
                                    <span className={styles.resultLabel}>{link.label}</span>
                                    <span className={styles.resultMeta}>Page</span>
                                </button>
                            ))}
                        </>
                    ) : isLoading ? (
                        <div className={styles.loading}>
                            <div className="spinner" style={{ width: 24, height: 24, borderWidth: 2 }} />
                            <span>Searching...</span>
                        </div>
                    ) : results.length > 0 ? (
                        <>
                            <div className={styles.sectionLabel}>Players</div>
                            {results.map((player, i) => (
                                <button
                                    key={player.player_id}
                                    className={`${styles.resultItem} ${i === selectedIndex ? styles.selected : ''}`}
                                    onClick={() => handleSelect(i)}
                                    onMouseEnter={() => setSelectedIndex(i)}
                                >
                                    <span className={styles.playerAvatar}>
                                        {player.full_name.charAt(0)}
                                    </span>
                                    <span className={styles.resultLabel}>{player.full_name}</span>
                                    <span className={styles.resultMeta}>Player</span>
                                </button>
                            ))}
                        </>
                    ) : (
                        <div className={styles.empty}>
                            No results for &ldquo;{query}&rdquo;
                        </div>
                    )}
                </div>

                <div className={styles.footer}>
                    <span className={styles.hint}>
                        <kbd className={styles.kbdSmall}>&uarr;</kbd>
                        <kbd className={styles.kbdSmall}>&darr;</kbd>
                        to navigate
                    </span>
                    <span className={styles.hint}>
                        <kbd className={styles.kbdSmall}>Enter</kbd>
                        to select
                    </span>
                    <span className={styles.hint}>
                        <kbd className={styles.kbdSmall}>Esc</kbd>
                        to close
                    </span>
                </div>
            </div>
        </div>
    );
}
