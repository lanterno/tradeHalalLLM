import { apiFetch } from "./client";
import type { Market, Trade } from "./types";

export interface TradeFilters {
  limit?: number;
  offset?: number;
  pair?: string;
  side?: string;
  status?: string;
  from_date?: string;
  to_date?: string;
  market?: Market;
}

export async function fetchTrades(filters: TradeFilters = {}): Promise<Trade[]> {
  const params = new URLSearchParams();
  if (filters.limit) params.set("limit", String(filters.limit));
  if (filters.offset) params.set("offset", String(filters.offset));
  if (filters.pair) params.set("pair", filters.pair);
  if (filters.side) params.set("side", filters.side);
  if (filters.status) params.set("status", filters.status);
  if (filters.from_date) params.set("from_date", filters.from_date);
  if (filters.to_date) params.set("to_date", filters.to_date);
  // Always send the market; the backend otherwise defaults to crypto and a
  // stock operator sees the empty crypto table.
  if (filters.market) params.set("market", filters.market);
  const qs = params.toString();
  return apiFetch<Trade[]>(`/api/trades${qs ? `?${qs}` : ""}`);
}
