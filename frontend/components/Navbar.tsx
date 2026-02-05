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
        { path: '/', label: 'Dashboard', icon: '◈' },
        { path: '/players', label: 'Players', icon: '◎' },
        { path: '/leaderboards', label: 'Leaderboards', icon: '◇' },
        { path: '/compare', label: 'Compare', icon: '⬡' },
    ];

    return (
        <nav className={styles.navbar}>
            <div className={styles.container}>
                <Link href="/" className={styles.logo}>
                    <div className={styles.logoMark}>
                        <span className={styles.logoIcon}>❄</span>
                    </div>
                    <div className={styles.logoText}>
                        <span className={styles.logoPrimary}>NHL</span>
                        <span className={styles.logoSecondary}>Analytics</span>
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
                                    <span className={styles.navIcon}>{item.icon}</span>
                                    <span className={styles.navLabel}>{item.label}</span>
                                    {isActive(item.path) && <span className={styles.activeIndicator} />}
                                </Link>
                            </li>
                        ))}
                    </ul>
                </div>

                <div className={styles.navRight}>
                    <div className={styles.statusBadge}>
                        <span className={styles.statusDot} />
                        <span className={styles.statusText}>Live Data</span>
                    </div>
                </div>
            </div>
        </nav>
    );
}
