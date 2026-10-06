import { apiFetch } from "./client";
import type { BeliefBoard, BeliefOverview, ShadowDecision } from "./types";

export function fetchBeliefBoard(): Promise<BeliefBoard> {
  return apiFetch<BeliefBoard>("/api/halabot/beliefs");
}

export function fetchBeliefOverview(): Promise<BeliefOverview> {
  return apiFetch<BeliefOverview>("/api/halabot/overview");
}

export function fetchShadowDecisions(limit = 30, asset?: string): Promise<ShadowDecision[]> {
  const q = new URLSearchParams({ limit: String(limit) });
  if (asset) q.set("asset", asset);
  return apiFetch<ShadowDecision[]>(`/api/halabot/decisions?${q}`);
}
