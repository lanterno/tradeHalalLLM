import { useQuery } from "@tanstack/react-query";
import { fetchHalalCompliance } from "../api/halal";

export function useHalalCompliance() {
  return useQuery({
    queryKey: ["halal", "compliance"],
    queryFn: fetchHalalCompliance,
    refetchInterval: 60_000,
  });
}
