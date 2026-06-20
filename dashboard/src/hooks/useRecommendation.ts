import { useQuery } from "@tanstack/react-query";
import {
  fetchStockOfTheDay,
  fetchRecommendationHistory,
} from "../api/recommendation";

// The daily pick changes at most once per pre-market run, so a slow poll
// is plenty — it mainly picks up a fresh pick after generation.
const REFRESH_MS = 60_000;

export function useStockOfTheDay() {
  return useQuery({
    queryKey: ["recommendation", "daily"],
    queryFn: fetchStockOfTheDay,
    refetchInterval: REFRESH_MS,
  });
}

export function useRecommendationHistory(limit = 30) {
  return useQuery({
    queryKey: ["recommendation", "history", limit],
    queryFn: () => fetchRecommendationHistory(limit),
    refetchInterval: REFRESH_MS,
  });
}
