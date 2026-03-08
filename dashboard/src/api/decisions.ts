import { apiFetch } from "./client";
import type { LlmDecision, StrategyAdjustment } from "./types";

export async function fetchDecisions(limit = 50): Promise<LlmDecision[]> {
  return apiFetch<LlmDecision[]>(`/api/decisions?limit=${limit}`);
}

export async function fetchAdjustments(): Promise<StrategyAdjustment[]> {
  return apiFetch<StrategyAdjustment[]>("/api/adjustments");
}
