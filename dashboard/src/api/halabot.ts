import { apiFetch } from "./client";
import type { BeliefBoard, ShadowDecision } from "./types";

export function fetchBeliefBoard(): Promise<BeliefBoard> {
  return apiFetch<BeliefBoard>("/api/halabot/beliefs");
}

export function fetchShadowDecisions(limit = 30): Promise<ShadowDecision[]> {
  return apiFetch<ShadowDecision[]>(`/api/halabot/decisions?limit=${limit}`);
}
