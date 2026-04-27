import { apiFetch } from "./client";

// All insights routes follow the same shape: either {"available": false}
// or {"available": true, ...payload}. Each fetcher narrows the payload
// to a tile-friendly shape; absent/unavailable returns null so the tile
// can render an empty-state message instead of throwing.

export interface DriftState {
  available: true;
  state: "warming_up" | "stable" | "drift";
  n: number;
  drift_count: number;
  last_drift_at: number | null;
}

export interface ShadowState {
  available: true;
  n: number;
  level: "ok" | "watch" | "diverged";
  metrics: {
    n: number;
    mean_diff_pct: number;
    last_diff_pct: number;
    max_drawdown_diff: number;
    paired_t_score: number;
    direction: "live_better" | "live_worse" | "even";
  } | null;
  ts: string;
}

export interface RegretSummary {
  n: number;
  mean_regret: number;
  median_regret: number;
  pct_high_regret: number;
  missed_edge_count: number;
  tail_loss_count: number;
  by_symbol: Record<string, number>;
}

export interface ThesisAttribution {
  rows: Array<{
    tag: string;
    n_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    avg_pnl_pct: number;
  }>;
  kill_candidates: string[];
}

export interface WhaleFlow {
  available: true;
  flows: Array<{
    symbol: string;
    inflow_to_exchange_usd: number;
    outflow_from_exchange_usd: number;
    inflow_pressure: number;
    n_transfers: number;
    label: string;
  }>;
}

export interface VelocityResults {
  available: true;
  results: Array<{
    symbol: string;
    n_recent: number;
    n_older: number;
    n_total: number;
    velocity: number;
    novelty: number;
    label: string;
  }>;
}

export interface PurificationSummary {
  available: true;
  total_usd: number;
  by_symbol: Record<string, number>;
  disbursed_total_usd: number;
  n_entries: number;
}

async function fetchOptional<T>(path: string): Promise<T | null> {
  const res = await apiFetch<{ available?: boolean } & Record<string, unknown>>(path);
  if (res.available === false) return null;
  return res as T;
}

export async function fetchDrift(): Promise<DriftState | null> {
  return fetchOptional<DriftState>("/api/insights/drift");
}

export async function fetchShadow(): Promise<ShadowState | null> {
  return fetchOptional<ShadowState>("/api/insights/shadow");
}

export async function fetchRegret(limit = 200): Promise<RegretSummary | null> {
  try {
    const r = await apiFetch<RegretSummary | { error: string }>(
      `/api/insights/regret?limit=${limit}`,
    );
    if ("error" in r) return null;
    return r;
  } catch {
    return null;
  }
}

export async function fetchThesis(limit = 200): Promise<ThesisAttribution | null> {
  try {
    const r = await apiFetch<ThesisAttribution | { error: string }>(
      `/api/insights/thesis?limit=${limit}`,
    );
    if ("error" in r) return null;
    return r;
  } catch {
    return null;
  }
}

export async function fetchWhale(): Promise<WhaleFlow | null> {
  return fetchOptional<WhaleFlow>("/api/insights/whale");
}

export async function fetchVelocity(): Promise<VelocityResults | null> {
  return fetchOptional<VelocityResults>("/api/insights/velocity");
}

export async function fetchPurification(): Promise<PurificationSummary | null> {
  return fetchOptional<PurificationSummary>("/api/insights/purification");
}
