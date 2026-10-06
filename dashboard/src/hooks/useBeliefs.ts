import { useQuery } from "@tanstack/react-query";
import { fetchBeliefBoard, fetchBeliefOverview, fetchShadowDecisions } from "../api/halabot";

export function useBeliefBoard() {
  return useQuery({
    queryKey: ["halabot", "beliefs"],
    queryFn: fetchBeliefBoard,
    refetchInterval: 30_000,
  });
}

/** The trust strip and side column: the record, the book, health, cost. */
export function useBeliefOverview() {
  return useQuery({
    queryKey: ["halabot", "overview"],
    queryFn: fetchBeliefOverview,
    refetchInterval: 60_000,
  });
}

export function useShadowDecisions(limit = 30) {
  return useQuery({
    queryKey: ["halabot", "decisions", limit],
    queryFn: () => fetchShadowDecisions(limit),
    refetchInterval: 30_000,
  });
}

/** One name's recent decisions, fetched only while its detail is open. */
export function useAssetDecisions(asset: string, limit = 8) {
  return useQuery({
    queryKey: ["halabot", "decisions", asset, limit],
    queryFn: () => fetchShadowDecisions(limit, asset),
    refetchInterval: 60_000,
  });
}
