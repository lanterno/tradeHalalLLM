import { apiFetch } from "./client";
import type { OperationsStatus } from "./types";

export async function fetchOperations(): Promise<OperationsStatus> {
  return apiFetch<OperationsStatus>("/api/operations");
}
