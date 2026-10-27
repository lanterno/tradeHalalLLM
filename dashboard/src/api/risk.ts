import { apiFetch } from "./client";
import type { CoreRisk, HaltStatus, ReconcileLogRow, RiskMeterStatus, RiskState } from "./types";

export async function fetchRiskState(): Promise<RiskState> {
  return apiFetch<RiskState>("/api/risk/state");
}

export async function fetchCoreRisk(): Promise<CoreRisk> {
  return apiFetch<CoreRisk>("/api/risk/core");
}

export async function fetchHaltStatus(): Promise<HaltStatus> {
  return apiFetch<HaltStatus>("/api/system/halt");
}

export async function setHalt(reason: string): Promise<HaltStatus> {
  return apiFetch<HaltStatus>("/api/system/halt", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Trader-Confirm": "true",
    },
    body: JSON.stringify({ reason }),
  });
}

export async function clearHalt(): Promise<HaltStatus> {
  return apiFetch<HaltStatus>("/api/system/halt", {
    method: "DELETE",
    headers: {
      "X-Trader-Confirm": "true",
    },
  });
}

export async function fetchRiskMeter(): Promise<RiskMeterStatus> {
  return apiFetch<RiskMeterStatus>("/api/system/risk-meter");
}

export async function setRiskMeter({
  level,
  reason,
}: {
  level: number;
  reason: string;
}): Promise<RiskMeterStatus> {
  return apiFetch<RiskMeterStatus>("/api/system/risk-meter", {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      "X-Trader-Confirm": "true",
    },
    body: JSON.stringify({ level, reason }),
  });
}

export async function fetchReconcileRecent(
  limit: number,
): Promise<ReconcileLogRow[]> {
  return apiFetch<ReconcileLogRow[]>(`/api/system/reconcile/recent?limit=${limit}`);
}
