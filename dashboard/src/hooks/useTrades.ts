import { useQuery } from "@tanstack/react-query";
import { fetchTrades, type TradeFilters } from "../api/trades";
import { useMarket } from "../lib/market";

export function useTrades(filters: TradeFilters = {}) {
  const { market } = useMarket();
  const merged: TradeFilters = { ...filters, market };
  return useQuery({
    queryKey: ["trades", merged],
    queryFn: () => fetchTrades(merged),
    refetchInterval: 15_000,
  });
}
