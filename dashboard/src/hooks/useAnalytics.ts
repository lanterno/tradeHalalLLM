import { useQuery } from "@tanstack/react-query";
import { fetchAnalytics, fetchDailyPnl } from "../api/analytics";

export function useAnalytics(days = 7) {
  return useQuery({
    queryKey: ["analytics", days],
    queryFn: () => fetchAnalytics(days),
    refetchInterval: 30_000,
  });
}

export function useDailyPnl(days = 30) {
  return useQuery({
    queryKey: ["dailyPnl", days],
    queryFn: () => fetchDailyPnl(days),
    refetchInterval: 60_000,
  });
}
