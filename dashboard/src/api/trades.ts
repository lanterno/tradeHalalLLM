import { apiFetch } from "./client";
import type { Trade } from "./types";

export interface TradeFilters {
  limit?: number;
  offset?: number;
  symbol?: string;
  side?: string;
  status?: string;
  from_date?: string;
  to_date?: string;
}

export async function fetchTrades(filters: TradeFilters = {}): Promise<Trade[]> {
  const params = new URLSearchParams();
  if (filters.limit) params.set("limit", String(filters.limit));
  if (filters.offset) params.set("offset", String(filters.offset));
  if (filters.symbol) params.set("symbol", filters.symbol);
  if (filters.side) params.set("side", filters.side);
  if (filters.status) params.set("status", filters.status);
  if (filters.from_date) params.set("from_date", filters.from_date);
  if (filters.to_date) params.set("to_date", filters.to_date);
  const qs = params.toString();
  return apiFetch<Trade[]>(`/api/trades${qs ? `?${qs}` : ""}`);
}
