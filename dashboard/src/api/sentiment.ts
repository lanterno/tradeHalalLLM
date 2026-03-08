import { apiFetch } from "./client";
import type { SentimentSignal } from "./types";

export async function fetchSentiment(): Promise<SentimentSignal[]> {
  return apiFetch<SentimentSignal[]>("/api/sentiment");
}
