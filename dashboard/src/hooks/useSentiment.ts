import { useQuery } from "@tanstack/react-query";
import { fetchSentiment } from "../api/sentiment";

export function useSentiment() {
  return useQuery({
    queryKey: ["sentiment"],
    queryFn: fetchSentiment,
    refetchInterval: 60_000,
  });
}
