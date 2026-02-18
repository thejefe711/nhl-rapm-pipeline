/**
 * API client for Data-Puck.com backend
 * Falls back to mock data when the API is unreachable.
 */

import {
    getMockStatsOverview,
    getMockSearchResults,
    getMockSeasons,
    getMockLeaderboard,
    getMockPlayer,
    getMockPlayerRAPM,
    getMockPlayerProfile,
    getMockPlayerExplanation,
} from './mock-data';

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
    try {
        return await fetchAPI<{ ok: boolean; duckdb_exists: boolean }>('/api/health');
    } catch {
        return { ok: false, duckdb_exists: false };
    }
}

// Seasons
export async function getSeasons() {
    try {
        return await fetchAPI<{ seasons: string[] }>('/api/seasons');
    } catch {
        return getMockSeasons();
    }
}

// Metrics
export async function getMetrics() {
    try {
        return await fetchAPI<{ metrics: Array<{ metric_name: string; rows: number }> }>('/api/metrics');
    } catch {
        return { metrics: [] };
    }
}

// Player search
export interface PlayerSearchResult {
    player_id: number;
    full_name: string;
}

export async function searchPlayers(query: string, limit = 20) {
    try {
        return await fetchAPI<{ q: string; limit: number; rows: PlayerSearchResult[] }>(
            `/api/players/search?q=${encodeURIComponent(query)}&limit=${limit}`
        );
    } catch {
        return getMockSearchResults(query);
    }
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
    try {
        return await fetchAPI<{ player: PlayerDetail }>(`/api/players/${playerId}`);
    } catch {
        return getMockPlayer(playerId);
    }
}

// Player RAPM
export interface RAPMRow {
    season: string;
    player_id: number;
    value: number;
}

export async function getPlayerRAPM(playerId: number, metric = 'corsi_rapm_5v5') {
    try {
        return await fetchAPI<{ player_id: number; full_name: string | null; metric: string; rows: RAPMRow[] }>(
            `/api/player/${playerId}/rapm?metric=${encodeURIComponent(metric)}`
        );
    } catch {
        return getMockPlayerRAPM(playerId, metric);
    }
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
    try {
        return await fetchAPI<{ season: string | null; metric: string; top: number; rows: LeaderboardRow[] }>(url);
    } catch {
        return getMockLeaderboard(metric, season || '20242025', top);
    }
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
    try {
        return await fetchAPI<PlayerExplanation>(
            `/api/explanations/player/${playerId}?season=${encodeURIComponent(season)}`
        );
    } catch {
        return getMockPlayerExplanation(playerId);
    }
}

// Player Profile (multi-metric)
export interface PlayerMetric {
    metric_name: string;
    value: number;
    games_count?: number;
    toi_seconds?: number;
    events_count?: number;
}

export interface MetricPercentile {
    percentile: number;
    total_players: number;
}

export interface CareerRow {
    season: string;
    metric_name: string;
    value: number;
}

export interface PlayerProfile {
    player: PlayerDetail & { position?: string };
    season: string;
    metrics: PlayerMetric[];
    percentiles: Record<string, MetricPercentile>;
    career: CareerRow[];
}

export async function getPlayerProfile(playerId: number, season?: string) {
    let url = `/api/player/${playerId}/profile`;
    if (season) url += `?season=${encodeURIComponent(season)}`;
    try {
        return await fetchAPI<PlayerProfile>(url);
    } catch {
        return getMockPlayerProfile(playerId, season) as PlayerProfile;
    }
}

// Stats overview (dashboard)
export interface StatsOverview {
    seasons: string[];
    seasons_count: number;
    total_players: number;
    total_metrics: number;
    total_games: number;
    latest_season: string | null;
    top_players: Array<{ player_id: number; value: number; full_name: string | null }>;
}

export async function getStatsOverview() {
    try {
        return await fetchAPI<StatsOverview>('/api/stats/overview');
    } catch {
        return getMockStatsOverview();
    }
}
