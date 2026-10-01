import { useQuery } from "@tanstack/react-query";
import { fetchShadow, fetchPurification } from "../api/insights";

// Refresh cadence: 30s — these endpoints reflect cycle-level state
// that doesn't change faster than that. Tiles re-fetch on focus too,
// which gives operator-driven freshness when actively investigating.
const REFRESH_MS = 30_000;

export function useShadow() {
  return useQuery({
    queryKey: ["insights", "shadow"],
    queryFn: fetchShadow,
    refetchInterval: REFRESH_MS,
  });
}

export function usePurification() {
  return useQuery({
    queryKey: ["insights", "purification"],
    queryFn: fetchPurification,
    refetchInterval: 5 * 60_000,
  });
}
