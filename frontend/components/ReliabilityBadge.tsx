import React from 'react';
import styles from './ReliabilityBadge.module.css';

interface ReliabilityBadgeProps {
    isReliable: boolean;
    toiSeconds?: number;
    text?: string;
}

export default function ReliabilityBadge({ isReliable, toiSeconds, text }: ReliabilityBadgeProps) {
    if (isReliable && (!toiSeconds || toiSeconds >= 1800)) {
        return null;
    }

    const type = !isReliable ? 'danger' : 'warning';

    let defaultText = 'Low Sample Size';
    if (!isReliable) {
        defaultText = 'Unreliable Data';
    } else if (toiSeconds && toiSeconds < 1800) {
        defaultText = `< 30 min TOI`;
    }

    return (
        <div className={`${styles.badge} ${styles[type]}`} title={text || defaultText}>
            <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span>{text || defaultText}</span>
        </div>
    );
}
