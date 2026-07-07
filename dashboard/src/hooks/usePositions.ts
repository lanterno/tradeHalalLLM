import { useQuery } from "@tanstack/react-query";
import { fetchPositions } from "../api/positions";
import { useMarket } from "../lib/market";

export function usePositions() {
  const { market } = useMarket();
  return useQuery({
    queryKey: ["positions", market],
    queryFn: () => fetchPositions(market),
    refetchInterval: 5_000,
  });
}
