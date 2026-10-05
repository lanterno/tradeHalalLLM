import { apiFetch } from "./client";
import type { HomeStatus } from "./types";

export async function fetchHome(): Promise<HomeStatus> {
  return apiFetch<HomeStatus>("/api/home");
}
