import { apiFetch } from "./client";
import type { OpenPosition } from "./types";

export async function fetchPositions(): Promise<OpenPosition[]> {
  return apiFetch<OpenPosition[]>("/api/positions");
}
