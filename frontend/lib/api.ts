/**
 * API client for NHL Analytics backend
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${API_URL}${endpoint}`;

    const res = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
        },
    });

    if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`);
    }

    return res.json();
}

// Health check
export async function checkHealth() {
    return fetchAPI<{ ok: boolean; duckdb_exists: boolean }>('/api/health');
}

// Seasons
export async function getSeasons() {
    return fetchAPI<{ seasons: string[] }>('/api/seasons');
}

// Metrics
export async function getMetrics() {
    return fetchAPI<{ metrics: Array<{ metric_name: string; rows: number }> }>('/api/metrics');
}

// Player search
export interface PlayerSearchResult {
    player_id: number;
    full_name: string;
}

export async function searchPlayers(query: string, limit = 20) {
    return fetchAPI<{ q: string; limit: number; rows: PlayerSearchResult[] }>(
        `/api/players/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );
}

// Player detail
export interface PlayerDetail {
    player_id: number;
    full_name?: string;
    first_name?: string;
    last_name?: string;
    games_count?: number;
    seasons_count?: number;
}

export async function getPlayer(playerId: number) {
    return fetchAPI<{ player: PlayerDetail }>(`/api/players/${playerId}`);
}

// Player RAPM
export interface RAPMRow {
    season: string;
    player_id: number;
    value: number;
}

export async function getPlayerRAPM(playerId: number, metric = 'corsi_rapm_5v5') {
    return fetchAPI<{ player_id: number; full_name: string | null; metric: string; rows: RAPMRow[] }>(
        `/api/player/${playerId}/rapm?metric=${encodeURIComponent(metric)}`
    );
}

// Leaderboard
export interface LeaderboardRow {
    season: string;
    player_id: number;
    value: number;
    full_name: string | null;
    games_count?: number;
    events_count?: number;
}

export async function getLeaderboard(metric: string, season?: string, top = 20) {
    let url = `/api/leaderboards?metric=${encodeURIComponent(metric)}&top=${top}`;
    if (season) {
        url += `&season=${encodeURIComponent(season)}`;
    }
    return fetchAPI<{ season: string | null; metric: string; top: number; rows: LeaderboardRow[] }>(url);
}

// Player explanation
export interface PlayerExplanation {
    player_id: number;
    player_name: string;
    model: string;
    season: string;
    explanation: string;
    data_quality: 'good' | 'limited' | 'insufficient';
    stable_skills: number;
    emerging_skills: number;
}

export async function getPlayerExplanation(playerId: number, season = '20242025') {
    return fetchAPI<PlayerExplanation>(
        `/api/explanations/player/${playerId}?season=${encodeURIComponent(season)}`
    );
}
