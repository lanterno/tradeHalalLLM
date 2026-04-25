import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  clearHalt,
  fetchBackups,
  fetchHaltStatus,
  fetchReconcileRecent,
  fetchRiskState,
  setHalt,
} from "../api/risk";

export function useRiskState() {
  return useQuery({
    queryKey: ["risk", "state"],
    queryFn: fetchRiskState,
    refetchInterval: 15_000,
  });
}

export function useHaltStatus() {
  return useQuery({
    queryKey: ["halt", "status"],
    queryFn: fetchHaltStatus,
    refetchInterval: 10_000,
  });
}

export function useSetHalt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: setHalt,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["halt", "status"] });
    },
  });
}

export function useClearHalt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: clearHalt,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["halt", "status"] });
    },
  });
}

export function useReconcileRecent(limit = 25) {
  return useQuery({
    queryKey: ["reconcile", "recent", limit],
    queryFn: () => fetchReconcileRecent(limit),
    refetchInterval: 30_000,
  });
}

export function useBackups() {
  return useQuery({
    queryKey: ["backups"],
    queryFn: fetchBackups,
    refetchInterval: 60_000,
  });
}
