/**
 * Mock data for Data-Puck.com — used when the backend API is unreachable.
 * All values are realistic but synthetic.
 */

export const MOCK_SEASONS = ['20192020', '20202021', '20212022', '20222023', '20232024', '20242025'];

export interface MockPlayer {
    player_id: number;
    full_name: string;
    first_name: string;
    last_name: string;
    position: string;
    team: string;
    games_count: number;
    seasons_count: number;
}

export const MOCK_PLAYERS: MockPlayer[] = [
    { player_id: 8478402, full_name: 'Connor McDavid', first_name: 'Connor', last_name: 'McDavid', position: 'C', team: 'EDM', games_count: 80, seasons_count: 6 },
    { player_id: 8479318, full_name: 'Auston Matthews', first_name: 'Auston', last_name: 'Matthews', position: 'C', team: 'TOR', games_count: 74, seasons_count: 6 },
    { player_id: 8477934, full_name: 'Leon Draisaitl', first_name: 'Leon', last_name: 'Draisaitl', position: 'C', team: 'EDM', games_count: 81, seasons_count: 6 },
    { player_id: 8477492, full_name: 'Nathan MacKinnon', first_name: 'Nathan', last_name: 'MacKinnon', position: 'C', team: 'COL', games_count: 78, seasons_count: 6 },
    { player_id: 8478483, full_name: 'Cale Makar', first_name: 'Cale', last_name: 'Makar', position: 'D', team: 'COL', games_count: 77, seasons_count: 5 },
    { player_id: 8480800, full_name: 'David Pastrnak', first_name: 'David', last_name: 'Pastrnak', position: 'RW', team: 'BOS', games_count: 79, seasons_count: 6 },
    { player_id: 8479323, full_name: 'Mikko Rantanen', first_name: 'Mikko', last_name: 'Rantanen', position: 'RW', team: 'COL', games_count: 76, seasons_count: 6 },
    { player_id: 8478427, full_name: 'Sebastian Aho', first_name: 'Sebastian', last_name: 'Aho', position: 'C', team: 'CAR', games_count: 80, seasons_count: 6 },
    { player_id: 8477474, full_name: 'Adam Fox', first_name: 'Adam', last_name: 'Fox', position: 'D', team: 'NYR', games_count: 78, seasons_count: 4 },
    { player_id: 8476453, full_name: 'Nikita Kucherov', first_name: 'Nikita', last_name: 'Kucherov', position: 'RW', team: 'TBL', games_count: 75, seasons_count: 6 },
    { player_id: 8479339, full_name: 'Matthew Tkachuk', first_name: 'Matthew', last_name: 'Tkachuk', position: 'LW', team: 'FLA', games_count: 79, seasons_count: 6 },
    { player_id: 8476346, full_name: 'Roman Josi', first_name: 'Roman', last_name: 'Josi', position: 'D', team: 'NSH', games_count: 77, seasons_count: 6 },
    { player_id: 8478550, full_name: 'Mitch Marner', first_name: 'Mitch', last_name: 'Marner', position: 'RW', team: 'TOR', games_count: 81, seasons_count: 6 },
    { player_id: 8478420, full_name: 'Kyle Connor', first_name: 'Kyle', last_name: 'Connor', position: 'LW', team: 'WPG', games_count: 80, seasons_count: 5 },
    { player_id: 8476882, full_name: 'Sidney Crosby', first_name: 'Sidney', last_name: 'Crosby', position: 'C', team: 'PIT', games_count: 73, seasons_count: 6 },
    { player_id: 8479382, full_name: 'Quinn Hughes', first_name: 'Quinn', last_name: 'Hughes', position: 'D', team: 'VAN', games_count: 76, seasons_count: 5 },
    { player_id: 8481559, full_name: 'Kirill Kaprizov', first_name: 'Kirill', last_name: 'Kaprizov', position: 'LW', team: 'MIN', games_count: 78, seasons_count: 4 },
    { player_id: 8480803, full_name: 'Brady Tkachuk', first_name: 'Brady', last_name: 'Tkachuk', position: 'LW', team: 'OTT', games_count: 79, seasons_count: 5 },
    { player_id: 8477935, full_name: 'Mark Scheifele', first_name: 'Mark', last_name: 'Scheifele', position: 'C', team: 'WPG', games_count: 74, seasons_count: 6 },
    { player_id: 8480069, full_name: 'Jason Robertson', first_name: 'Jason', last_name: 'Robertson', position: 'LW', team: 'DAL', games_count: 77, seasons_count: 4 },
];

/* ── Seeded random for deterministic mock values ── */
function seededRandom(seed: number): () => number {
    let s = seed;
    return () => {
        s = (s * 16807 + 0) % 2147483647;
        return (s - 1) / 2147483646;
    };
}

/**
 * Generate a realistic RAPM-like value for a player/metric/season combo.
 * Top players get higher base values; each metric has a different scale.
 */
export function mockMetricValue(playerId: number, metric: string, season: string): number {
    const rand = seededRandom(playerId * 31 + hashString(metric) * 7 + hashString(season) * 3);

    const playerRank = MOCK_PLAYERS.findIndex(p => p.player_id === playerId);
    const baseTalent = playerRank >= 0 ? 1.0 - (playerRank / MOCK_PLAYERS.length) * 0.8 : 0.5;

    const isDefMetric = metric.includes('def');
    const isXG = metric.includes('xg');
    const isGoal = metric.includes('goals');
    const isHD = metric.includes('hd_');
    const isPenalty = metric.includes('penalties');
    const isAssist = metric.includes('assist');
    const isTurnover = metric.includes('turnover') || metric.includes('takeaway') || metric.includes('giveaway');

    let scale = 2.5;
    if (isXG) scale = 0.08;
    if (isGoal) scale = 0.05;
    if (isHD) scale = 0.04;
    if (isPenalty) scale = 0.15;
    if (isAssist) scale = 0.12;
    if (isTurnover) scale = 0.03;

    const noise = (rand() - 0.5) * scale * 0.7;
    let value = baseTalent * scale + noise;

    if (isDefMetric) value *= 0.6;
    if (metric.includes('penalties_committed')) value = -(Math.abs(value) * 0.5);
    if (metric.includes('giveaway')) value = -(Math.abs(value) * 0.4);

    return Math.round(value * 1000) / 1000;
}

function hashString(s: string): number {
    let h = 0;
    for (let i = 0; i < s.length; i++) {
        h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    }
    return Math.abs(h);
}

/* ── Pre-built responses ── */

export function getMockStatsOverview() {
    return {
        seasons: MOCK_SEASONS,
        seasons_count: MOCK_SEASONS.length,
        total_players: MOCK_PLAYERS.length * 40,
        total_metrics: 22,
        total_games: 1312,
        latest_season: '20242025',
        top_players: MOCK_PLAYERS.slice(0, 10).map(p => ({
            player_id: p.player_id,
            value: mockMetricValue(p.player_id, 'corsi_rapm_5v5', '20242025'),
            full_name: p.full_name,
        })).sort((a, b) => b.value - a.value),
    };
}

export function getMockSearchResults(query: string) {
    const q = query.toLowerCase();
    const rows = MOCK_PLAYERS
        .filter(p => p.full_name.toLowerCase().includes(q))
        .map(p => ({ player_id: p.player_id, full_name: p.full_name }));
    return { q: query, limit: 20, rows };
}

export function getMockSeasons() {
    return { seasons: MOCK_SEASONS };
}

export function getMockLeaderboard(metric: string, season: string, top: number) {
    const rows = MOCK_PLAYERS.map(p => ({
        season,
        player_id: p.player_id,
        value: mockMetricValue(p.player_id, metric, season),
        full_name: p.full_name,
        games_count: p.games_count,
        events_count: Math.floor(p.games_count * 48),
    }))
        .sort((a, b) => b.value - a.value)
        .slice(0, top);
    return { season, metric, top, rows };
}

export function getMockPlayer(playerId: number) {
    const p = MOCK_PLAYERS.find(mp => mp.player_id === playerId) || {
        player_id: playerId,
        full_name: `Player #${playerId}`,
        first_name: 'Unknown',
        last_name: 'Player',
        position: 'C',
        team: 'NHL',
        games_count: 60,
        seasons_count: 3,
    };
    return { player: { player_id: p.player_id, full_name: p.full_name, first_name: p.first_name, last_name: p.last_name, games_count: p.games_count, seasons_count: p.seasons_count } };
}

export function getMockPlayerRAPM(playerId: number, metric: string) {
    const player = MOCK_PLAYERS.find(p => p.player_id === playerId);
    const numSeasons = player?.seasons_count ?? 4;
    const seasons = MOCK_SEASONS.slice(-numSeasons);
    const rows = seasons.map(s => ({
        season: s,
        player_id: playerId,
        value: mockMetricValue(playerId, metric, s),
    }));
    return {
        player_id: playerId,
        full_name: player?.full_name ?? null,
        metric,
        rows,
    };
}

const ALL_METRICS = [
    'corsi_rapm_5v5', 'xg_rapm_5v5', 'goals_rapm_5v5',
    'corsi_off_rapm_5v5', 'xg_off_rapm_5v5',
    'corsi_def_rapm_5v5', 'xg_def_rapm_5v5',
    'hd_xg_rapm_5v5_ge020', 'hd_xg_off_rapm_5v5_ge020', 'hd_xg_def_rapm_5v5_ge020',
    'turnover_to_xg_swing_rapm_5v5_w10', 'takeaway_to_xg_swing_rapm_5v5_w10', 'giveaway_to_xg_swing_rapm_5v5_w10',
    'penalties_drawn_rapm_5v5', 'penalties_committed_rapm_5v5',
    'primary_assist_rapm_5v5', 'secondary_assist_rapm_5v5',
    'xg_primary_assist_on_goals_rapm_5v5', 'xg_secondary_assist_on_goals_rapm_5v5',
];

export function getMockPlayerProfile(playerId: number, season?: string) {
    const p = MOCK_PLAYERS.find(mp => mp.player_id === playerId);
    const s = season || '20242025';

    const metrics = ALL_METRICS.map(m => ({
        metric_name: m,
        value: mockMetricValue(playerId, m, s),
        games_count: p?.games_count ?? 60,
        toi_seconds: (p?.games_count ?? 60) * 1100,
        events_count: (p?.games_count ?? 60) * 48,
    }));

    const allPlayerValues = ALL_METRICS.reduce<Record<string, number[]>>((acc, m) => {
        acc[m] = MOCK_PLAYERS.map(pl => mockMetricValue(pl.player_id, m, s));
        return acc;
    }, {});

    const percentiles: Record<string, { percentile: number; total_players: number }> = {};
    for (const m of ALL_METRICS) {
        const playerVal = mockMetricValue(playerId, m, s);
        const vals = allPlayerValues[m].sort((a, b) => a - b);
        const rank = vals.filter(v => v <= playerVal).length;
        percentiles[m] = { percentile: (rank / vals.length) * 100, total_players: vals.length };
    }

    const numSeasons = p?.seasons_count ?? 4;
    const career = MOCK_SEASONS.slice(-numSeasons).flatMap(sz =>
        ALL_METRICS.map(m => ({
            season: sz,
            metric_name: m,
            value: mockMetricValue(playerId, m, sz),
        }))
    );

    return {
        player: {
            player_id: playerId,
            full_name: p?.full_name ?? `Player #${playerId}`,
            first_name: p?.first_name ?? 'Unknown',
            last_name: p?.last_name ?? 'Player',
            position: p?.position ?? 'C',
            games_count: p?.games_count ?? 60,
            seasons_count: p?.seasons_count ?? 4,
        },
        season: s,
        metrics,
        percentiles,
        career,
    };
}

export function getMockPlayerExplanation(playerId: number) {
    const p = MOCK_PLAYERS.find(mp => mp.player_id === playerId);
    const name = p?.full_name ?? 'This player';
    const corsi = mockMetricValue(playerId, 'corsi_rapm_5v5', '20242025');
    const xg = mockMetricValue(playerId, 'xg_rapm_5v5', '20242025');

    return {
        player_id: playerId,
        player_name: name,
        model: 'DLM + SAE',
        season: '20242025',
        explanation: `${name} projects as a ${corsi > 1.5 ? 'premier' : corsi > 0.5 ? 'strong' : 'developing'} two-way contributor with a Corsi RAPM of ${corsi >= 0 ? '+' : ''}${corsi.toFixed(2)} and xG impact of ${xg >= 0 ? '+' : ''}${xg.toFixed(3)}. ${corsi > 1 ? 'Their shot-driving ability ranks among the league elite.' : 'Their overall shot metrics show room for growth.'} Latent skill embeddings suggest ${p?.position === 'D' ? 'a high offensive ceiling for a defenseman' : 'elite transition play and offensive zone entry skills'}. The Kalman filter forecast projects ${corsi > 1 ? 'sustained elite production' : 'continued development'} over the next 1-2 seasons.`,
        data_quality: 'good' as const,
        stable_skills: corsi > 1 ? 5 : 3,
        emerging_skills: corsi > 1 ? 2 : 4,
    };
}
