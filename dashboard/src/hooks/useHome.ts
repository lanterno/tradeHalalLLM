import { useQuery } from "@tanstack/react-query";
import { fetchHome } from "../api/home";

export function useHome() {
  return useQuery({
    queryKey: ["home"],
    queryFn: fetchHome,
    // The bot refreshes the figures once a minute in the session.
    refetchInterval: 30_000,
  });
}
