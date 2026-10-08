import { useQuery } from "@tanstack/react-query";
import { fetchOperations } from "../api/operations";

export function useOperations() {
  return useQuery({
    queryKey: ["operations"],
    queryFn: fetchOperations,
    refetchInterval: 30_000,
  });
}
