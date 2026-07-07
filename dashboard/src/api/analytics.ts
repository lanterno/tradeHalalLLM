import { apiFetch } from "./client";
import type { AnalyticsStats, DailyPnl, Market } from "./types";

export async function fetchAnalytics(
  days = 7,
  market: Market = "stocks",
): Promise<AnalyticsStats> {
  return apiFetch<AnalyticsStats>(`/api/analytics?days=${days}&market=${market}`);
}

export async function fetchDailyPnl(
  days = 30,
  market: Market = "stocks",
): Promise<DailyPnl[]> {
  return apiFetch<DailyPnl[]>(`/api/pnl/daily?days=${days}&market=${market}`);
}
