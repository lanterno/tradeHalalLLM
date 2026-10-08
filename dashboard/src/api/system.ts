import { apiFetch } from "./client";
import type { HealthStatus } from "./types";

export async function fetchHealth(): Promise<HealthStatus> {
  return apiFetch<HealthStatus>("/api/health");
}
