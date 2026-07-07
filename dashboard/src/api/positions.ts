import { apiFetch } from "./client";
import type { Market, OpenPosition } from "./types";

export async function fetchPositions(
  market: Market = "stocks",
): Promise<OpenPosition[]> {
  return apiFetch<OpenPosition[]>(`/api/positions?market=${market}`);
}
