import { apiFetch } from "./client";
import type { HalalCompliance, PurificationYear, ZakatStatus } from "./types";

export async function fetchHalalCompliance(): Promise<HalalCompliance> {
  return apiFetch<HalalCompliance>("/api/halal/compliance");
}

export async function fetchZakat(): Promise<ZakatStatus> {
  return apiFetch<ZakatStatus>("/api/halal/zakat");
}

export async function fetchPurification(year?: number): Promise<PurificationYear> {
  return apiFetch<PurificationYear>(
    year ? `/api/halal/purification?year=${year}` : "/api/halal/purification",
  );
}
