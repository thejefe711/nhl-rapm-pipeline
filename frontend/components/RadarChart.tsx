'use client';

import styles from './RadarChart.module.css';

interface RadarDataPoint {
    label: string;
    value: number;
    percentile: number;
}

interface RadarChartProps {
    data: RadarDataPoint[];
    size?: number;
}

export default function RadarChart({ data, size = 300 }: RadarChartProps) {
    if (data.length < 3) return null;

    const cx = size / 2;
    const cy = size / 2;
    const radius = size * 0.36;
    const levels = 5;
    const angleStep = (2 * Math.PI) / data.length;

    const getPoint = (index: number, value: number) => {
        const angle = angleStep * index - Math.PI / 2;
        const r = (value / 100) * radius;
        return {
            x: cx + r * Math.cos(angle),
            y: cy + r * Math.sin(angle),
        };
    };

    const gridLevels = Array.from({ length: levels }, (_, i) => {
        const levelRadius = ((i + 1) / levels) * radius;
        const points = data.map((_, idx) => {
            const angle = angleStep * idx - Math.PI / 2;
            return `${cx + levelRadius * Math.cos(angle)},${cy + levelRadius * Math.sin(angle)}`;
        });
        return points.join(' ');
    });

    const axes = data.map((_, idx) => {
        const angle = angleStep * idx - Math.PI / 2;
        return {
            x2: cx + radius * Math.cos(angle),
            y2: cy + radius * Math.sin(angle),
        };
    });

    const dataPoints = data.map((d, idx) => getPoint(idx, d.percentile));
    const dataPath = dataPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + ' Z';

    const labels = data.map((d, idx) => {
        const angle = angleStep * idx - Math.PI / 2;
        const labelRadius = radius + 28;
        const x = cx + labelRadius * Math.cos(angle);
        const y = cy + labelRadius * Math.sin(angle);

        let anchor: 'start' | 'middle' | 'end' = 'middle';
        if (Math.cos(angle) > 0.3) anchor = 'start';
        else if (Math.cos(angle) < -0.3) anchor = 'end';

        return { x, y, anchor, label: d.label, value: d.value, percentile: d.percentile };
    });

    return (
        <div className={styles.container}>
            <svg viewBox={`0 0 ${size} ${size}`} className={styles.svg} width={size} height={size}>
                {/* Grid levels */}
                {gridLevels.map((points, i) => (
                    <polygon
                        key={i}
                        points={points}
                        fill="none"
                        stroke="rgba(0,0,0,0.07)"
                        strokeWidth="1"
                    />
                ))}

                {/* Axes */}
                {axes.map((axis, i) => (
                    <line
                        key={i}
                        x1={cx}
                        y1={cy}
                        x2={axis.x2}
                        y2={axis.y2}
                        stroke="rgba(0,0,0,0.07)"
                        strokeWidth="1"
                    />
                ))}

                {/* 50th percentile ring highlight */}
                <polygon
                    points={data.map((_, idx) => {
                        const angle = angleStep * idx - Math.PI / 2;
                        const r = 0.5 * radius;
                        return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
                    }).join(' ')}
                    fill="none"
                    stroke="rgba(0,0,0,0.15)"
                    strokeWidth="1"
                    strokeDasharray="3,3"
                />

                {/* Data area */}
                <path
                    d={dataPath}
                    fill="rgba(13, 148, 136, 0.10)"
                    stroke="rgba(13, 148, 136, 0.65)"
                    strokeWidth="2"
                />

                {/* Data points */}
                {dataPoints.map((p, i) => (
                    <circle
                        key={i}
                        cx={p.x}
                        cy={p.y}
                        r="4"
                        fill="var(--accent)"
                        stroke="#ffffff"
                        strokeWidth="2.5"
                    />
                ))}

                {/* Labels */}
                {labels.map((l, i) => (
                    <text
                        key={i}
                        x={l.x}
                        y={l.y}
                        textAnchor={l.anchor}
                        dominantBaseline="middle"
                        className={styles.label}
                    >
                        {l.label}
                    </text>
                ))}
            </svg>
        </div>
    );
}
