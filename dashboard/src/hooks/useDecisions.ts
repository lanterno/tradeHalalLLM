import { useQuery } from "@tanstack/react-query";
import {
  fetchDecisions,
  fetchAdjustments,
  fetchDecisionQuality,
} from "../api/decisions";

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

/** Computed once a night by the evening run; an hour's cache is plenty. */
export function useDecisionQuality() {
  return useQuery({
    queryKey: ["decision-quality"],
    queryFn: fetchDecisionQuality,
    staleTime: 3_600_000,
  });
}
