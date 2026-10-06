import { apiFetch } from "./client";
import type { PositionsResponse } from "./types";

export async function fetchPositions(): Promise<PositionsResponse> {
  return apiFetch<PositionsResponse>("/api/positions");
}
