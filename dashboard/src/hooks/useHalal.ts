import { useQuery } from "@tanstack/react-query";
import { fetchHalalCompliance, fetchPurification, fetchZakat } from "../api/halal";

export function useHalalCompliance() {
  return useQuery({
    queryKey: ["halal", "compliance"],
    queryFn: fetchHalalCompliance,
    refetchInterval: 60_000,
  });
}

export function useZakat() {
  return useQuery({
    queryKey: ["halal", "zakat"],
    queryFn: fetchZakat,
    refetchInterval: 300_000,
  });
}

export function usePurification(year?: number) {
  return useQuery({
    queryKey: ["halal", "purification", year ?? "current"],
    queryFn: () => fetchPurification(year),
    refetchInterval: 300_000,
  });
}
