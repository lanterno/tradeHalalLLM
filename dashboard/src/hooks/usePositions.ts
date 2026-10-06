import { useQuery } from "@tanstack/react-query";
import { fetchPositions } from "../api/positions";

export function usePositions() {
  return useQuery({
    queryKey: ["positions"],
    queryFn: fetchPositions,
    // The bot snapshots each account once a minute; polling faster shows nothing new.
    refetchInterval: 30_000,
  });
}
