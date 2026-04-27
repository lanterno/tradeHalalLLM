import { useQuery } from "@tanstack/react-query";
import {
  fetchDrift,
  fetchShadow,
  fetchRegret,
  fetchThesis,
  fetchWhale,
  fetchVelocity,
  fetchPurification,
} from "../api/insights";

// Refresh cadence: 30s — these endpoints reflect cycle-level state
// that doesn't change faster than that. Tiles re-fetch on focus too,
// which gives operator-driven freshness when actively investigating.
const REFRESH_MS = 30_000;

export function useDrift() {
  return useQuery({
    queryKey: ["insights", "drift"],
    queryFn: fetchDrift,
    refetchInterval: REFRESH_MS,
  });
}

export function useShadow() {
  return useQuery({
    queryKey: ["insights", "shadow"],
    queryFn: fetchShadow,
    refetchInterval: REFRESH_MS,
  });
}

export function useRegret(limit = 200) {
  return useQuery({
    queryKey: ["insights", "regret", limit],
    queryFn: () => fetchRegret(limit),
    refetchInterval: REFRESH_MS,
  });
}

export function useThesis(limit = 200) {
  return useQuery({
    queryKey: ["insights", "thesis", limit],
    queryFn: () => fetchThesis(limit),
    refetchInterval: REFRESH_MS,
  });
}

export function useWhale() {
  return useQuery({
    queryKey: ["insights", "whale"],
    queryFn: fetchWhale,
    refetchInterval: REFRESH_MS,
  });
}

export function useVelocity() {
  return useQuery({
    queryKey: ["insights", "velocity"],
    queryFn: fetchVelocity,
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
