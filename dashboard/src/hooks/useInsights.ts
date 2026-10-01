import { useQuery } from "@tanstack/react-query";
import { fetchPurification } from "../api/insights";

// Purification accrues once per closed winning trade, so a 5-minute poll
// is plenty. The tile also re-fetches on window focus.
export function usePurification() {
  return useQuery({
    queryKey: ["insights", "purification"],
    queryFn: fetchPurification,
    refetchInterval: 5 * 60_000,
  });
}
