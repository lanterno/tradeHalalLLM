import { useQuery } from "@tanstack/react-query";
import { fetchCore } from "../api/core";

export function useCore() {
  return useQuery({
    queryKey: ["core"],
    queryFn: fetchCore,
    refetchInterval: 60_000,
  });
}
