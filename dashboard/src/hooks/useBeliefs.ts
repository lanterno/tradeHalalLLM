import { useQuery } from "@tanstack/react-query";
import { fetchBeliefBoard, fetchShadowDecisions } from "../api/halabot";

export function useBeliefBoard() {
  return useQuery({
    queryKey: ["halabot", "beliefs"],
    queryFn: fetchBeliefBoard,
    refetchInterval: 30_000,
  });
}

export function useShadowDecisions(limit = 30) {
  return useQuery({
    queryKey: ["halabot", "decisions", limit],
    queryFn: () => fetchShadowDecisions(limit),
    refetchInterval: 30_000,
  });
}
