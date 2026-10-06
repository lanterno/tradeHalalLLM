import { useQuery } from "@tanstack/react-query";
import { fetchHealth, fetchSystemStatus, fetchConfig, fetchCoreConfig } from "../api/system";

export function useCoreConfig() {
  return useQuery({
    queryKey: ["config", "core"],
    queryFn: fetchCoreConfig,
    staleTime: 300_000,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 10_000,
  });
}

export function useSystemStatus() {
  return useQuery({
    queryKey: ["systemStatus"],
    queryFn: fetchSystemStatus,
    refetchInterval: 10_000,
  });
}

export function useConfig() {
  return useQuery({
    queryKey: ["config"],
    queryFn: fetchConfig,
    staleTime: 300_000,
  });
}
