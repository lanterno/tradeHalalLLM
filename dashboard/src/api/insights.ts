import { apiFetch } from "./client";

// All insights routes follow the same shape: either {"available": false}
// or {"available": true, ...payload}. Each fetcher narrows the payload
// to a tile-friendly shape; absent/unavailable returns null so the tile
// can render an empty-state message instead of throwing.

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

export async function fetchPurification(): Promise<PurificationSummary | null> {
  return fetchOptional<PurificationSummary>("/api/insights/purification");
}
