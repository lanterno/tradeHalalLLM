import { useQuery } from "@tanstack/react-query";
import { fetchAnalytics, fetchDailyPnl } from "../api/analytics";
import { useMarket } from "../lib/market";

export function useAnalytics(days = 7) {
  const { market } = useMarket();
  return useQuery({
    queryKey: ["analytics", days, market],
    queryFn: () => fetchAnalytics(days, market),
    refetchInterval: 30_000,
  });
}

export function useDailyPnl(days = 30) {
  const { market } = useMarket();
  return useQuery({
    queryKey: ["dailyPnl", days, market],
    queryFn: () => fetchDailyPnl(days, market),
    refetchInterval: 60_000,
  });
}
