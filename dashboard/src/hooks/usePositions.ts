import { useQuery } from "@tanstack/react-query";
import { fetchPositions } from "../api/positions";

export function usePositions() {
  return useQuery({
    queryKey: ["positions"],
    queryFn: fetchPositions,
    refetchInterval: 5_000,
  });
}
