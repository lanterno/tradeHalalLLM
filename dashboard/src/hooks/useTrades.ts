import { useQuery } from "@tanstack/react-query";
import { fetchTrades, type TradeFilters } from "../api/trades";

export function useTrades(filters: TradeFilters = {}) {
  return useQuery({
    queryKey: ["trades", filters],
    queryFn: () => fetchTrades(filters),
    refetchInterval: 15_000,
  });
}
