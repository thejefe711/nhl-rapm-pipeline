'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './Navbar.module.css';

export default function Navbar() {
    const pathname = usePathname();

    const isActive = (path: string) => {
        if (path === '/') return pathname === '/';
        return pathname.startsWith(path);
    };

    const navItems = [
        { path: '/', label: 'Dashboard', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4' },
        { path: '/players', label: 'Players', icon: 'M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z' },
        { path: '/leaderboards', label: 'Leaderboards', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
        { path: '/compare', label: 'Compare', icon: 'M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4' },
        { path: '/glossary', label: 'Glossary', icon: 'M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253' },
    ];

    const triggerCommandPalette = () => {
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true }));
    };

    return (
        <nav className={styles.navbar}>
            <div className={styles.container}>
                <Link href="/" className={styles.logo}>
                    <div className={styles.logoMark}>
                        <svg className={styles.logoIcon} viewBox="0 0 16 16" fill="currentColor" width="16" height="16">
                            <ellipse cx="8" cy="8" rx="7" ry="5.5" />
                        </svg>
                    </div>
                    <div className={styles.logoText}>
                        <span className={styles.logoPrimary}>Data</span>
                        <span className={styles.logoDash}>-</span>
                        <span className={styles.logoPrimary}>Puck</span>
                    </div>
                </Link>

                <div className={styles.navCenter}>
                    <ul className={styles.nav}>
                        {navItems.map((item) => (
                            <li key={item.path}>
                                <Link
                                    href={item.path}
                                    className={`${styles.navLink} ${isActive(item.path) ? styles.active : ''}`}
                                >
                                    <svg className={styles.navIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                        <path d={item.icon} />
                                    </svg>
                                    <span className={styles.navLabel}>{item.label}</span>
                                    {isActive(item.path) && <span className={styles.activeIndicator} />}
                                </Link>
                            </li>
                        ))}
                    </ul>
                </div>

                <div className={styles.navRight}>
                    <button
                        className={styles.searchTrigger}
                        onClick={triggerCommandPalette}
                        title="Search (Ctrl+K)"
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={styles.searchTriggerIcon}>
                            <circle cx="11" cy="11" r="8" />
                            <path d="M21 21l-4.35-4.35" />
                        </svg>
                        <span className={styles.searchTriggerText}>Search...</span>
                        <kbd className={styles.searchKbd}>Ctrl K</kbd>
                    </button>
                    <div className={styles.statusBadge}>
                        <span className={styles.statusDot} />
                        <span className={styles.statusText}>Live</span>
                    </div>
                </div>
            </div>
        </nav>
    );
}
