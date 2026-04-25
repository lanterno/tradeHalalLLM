import { apiFetch } from "./client";
import type { BackupRow, HaltStatus, ReconcileLogRow, RiskState } from "./types";

export async function fetchRiskState(): Promise<RiskState> {
  return apiFetch<RiskState>("/api/risk/state");
}

export async function fetchHaltStatus(): Promise<HaltStatus> {
  return apiFetch<HaltStatus>("/api/system/halt");
}

export async function setHalt(reason: string): Promise<HaltStatus> {
  return apiFetch<HaltStatus>("/api/system/halt", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Halt-Confirm": "yes",
    },
    body: JSON.stringify({ reason }),
  });
}

export async function clearHalt(): Promise<HaltStatus> {
  return apiFetch<HaltStatus>("/api/system/halt", {
    method: "DELETE",
    headers: {
      "X-Halt-Confirm": "yes",
    },
  });
}

export async function fetchReconcileRecent(
  limit: number,
): Promise<ReconcileLogRow[]> {
  return apiFetch<ReconcileLogRow[]>(`/api/system/reconcile/recent?limit=${limit}`);
}

export async function fetchBackups(): Promise<BackupRow[]> {
  return apiFetch<BackupRow[]>("/api/system/backups");
}
