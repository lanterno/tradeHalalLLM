import { apiFetch } from "./client";
import type { StockOfTheDay } from "./types";

export async function fetchStockOfTheDay(): Promise<StockOfTheDay> {
  return apiFetch<StockOfTheDay>("/api/recommendation");
}

export async function fetchRecommendationHistory(
  limit = 30,
): Promise<StockOfTheDay[]> {
  return apiFetch<StockOfTheDay[]>(`/api/recommendation/history?limit=${limit}`);
}
