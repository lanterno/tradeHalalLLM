import { apiFetch } from "./client";
import type { HalalCompliance } from "./types";

export async function fetchHalalCompliance(): Promise<HalalCompliance> {
  return apiFetch<HalalCompliance>("/api/halal/compliance");
}
