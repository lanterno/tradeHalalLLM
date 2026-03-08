import { useQuery } from "@tanstack/react-query";
import { fetchDecisions, fetchAdjustments } from "../api/decisions";

export function useDecisions(limit = 50) {
  return useQuery({
    queryKey: ["decisions", limit],
    queryFn: () => fetchDecisions(limit),
    refetchInterval: 30_000,
  });
}

export function useAdjustments() {
  return useQuery({
    queryKey: ["adjustments"],
    queryFn: fetchAdjustments,
    refetchInterval: 30_000,
  });
}
