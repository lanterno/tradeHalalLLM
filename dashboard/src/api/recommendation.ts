import { apiFetch } from "./client";
import type { StockOfTheDay, RecommendationScorecard } from "./types";

export async function fetchStockOfTheDay(): Promise<StockOfTheDay> {
  return apiFetch<StockOfTheDay>("/api/recommendation");
}

export async function fetchRecommendationHistory(
  limit = 30,
): Promise<StockOfTheDay[]> {
  return apiFetch<StockOfTheDay[]>(`/api/recommendation/history?limit=${limit}`);
}

export async function fetchRecommendationScorecard(): Promise<RecommendationScorecard> {
  return apiFetch<RecommendationScorecard>("/api/recommendation/scorecard");
}
