import { apiFetch } from "./client";
import type { HealthStatus, SystemStatus, AppConfig } from "./types";

export async function fetchHealth(): Promise<HealthStatus> {
  return apiFetch<HealthStatus>("/api/health");
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  return apiFetch<SystemStatus>("/api/system/status");
}

export async function fetchConfig(): Promise<AppConfig> {
  return apiFetch<AppConfig>("/api/config");
}
