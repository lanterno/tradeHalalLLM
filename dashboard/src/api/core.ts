import { apiFetch } from "./client";
import type { CoreStatus } from "./types";

export async function fetchCore(): Promise<CoreStatus> {
  return apiFetch<CoreStatus>("/api/core");
}
