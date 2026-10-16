import { apiFetch } from "./client";
import type { DecisionQuality, LlmDecision, StrategyAdjustment } from "./types";

export async function fetchDecisions(limit = 50): Promise<LlmDecision[]> {
  return apiFetch<LlmDecision[]>(`/api/decisions?limit=${limit}`);
}

export async function fetchAdjustments(): Promise<StrategyAdjustment[]> {
  return apiFetch<StrategyAdjustment[]>("/api/adjustments");
}

export async function fetchDecisionQuality(): Promise<DecisionQuality | null> {
  return apiFetch<DecisionQuality | null>("/api/decisions/quality");
}
