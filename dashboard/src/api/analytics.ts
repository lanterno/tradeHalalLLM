import { apiFetch } from "./client";
import type { AnalyticsStats, DailyPnl } from "./types";

export async function fetchAnalytics(days = 7): Promise<AnalyticsStats> {
  return apiFetch<AnalyticsStats>(`/api/analytics?days=${days}`);
}

export async function fetchDailyPnl(days = 30): Promise<DailyPnl[]> {
  return apiFetch<DailyPnl[]>(`/api/pnl/daily?days=${days}`);
}
