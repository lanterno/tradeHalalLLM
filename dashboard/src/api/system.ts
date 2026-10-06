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

/** The core portfolio's settings and rule constants (no keys). */
export type CoreConfig = Record<string, string | number | boolean | null>;

export async function fetchCoreConfig(): Promise<CoreConfig> {
  return apiFetch<CoreConfig>("/api/system/core-config");
}
