import { useQuery } from "@tanstack/react-query";
import { fetchCycleMetrics, fetchLlmMetrics } from "../api/metrics";

export function useCycleMetrics(windowSeconds: number) {
  return useQuery({
    queryKey: ["metrics", "cycles", windowSeconds],
    queryFn: () => fetchCycleMetrics(windowSeconds),
    refetchInterval: 30_000,
  });
}

export function useLlmMetrics(windowSeconds: number) {
  return useQuery({
    queryKey: ["metrics", "llm", windowSeconds],
    queryFn: () => fetchLlmMetrics(windowSeconds),
    refetchInterval: 30_000,
  });
}
