import { apiFetch } from "./client";
import type { CycleMetrics, LlmMetrics } from "./types";

export async function fetchCycleMetrics(window: number): Promise<CycleMetrics> {
  return apiFetch<CycleMetrics>(`/api/metrics/cycles?window=${window}`);
}

export async function fetchLlmMetrics(window: number): Promise<LlmMetrics> {
  return apiFetch<LlmMetrics>(`/api/metrics/llm?window=${window}`);
}
