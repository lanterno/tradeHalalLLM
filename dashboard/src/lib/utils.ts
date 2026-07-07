import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatQty(value: number, decimals = 6): string {
  // Trim trailing zeros so stock share counts render as "44" (not
  // "44.000000") while crypto fractional sizes keep their precision.
  return parseFloat(value.toFixed(decimals)).toString();
}

/**
 * The per-asset key for a row: stocks carry ``symbol``, crypto carry
 * ``pair``. Returns whichever is present so one template renders both.
 */
export function entityOf(row: { symbol?: string | null; pair?: string | null }): string {
  return row.symbol ?? row.pair ?? "";
}

export function formatTime(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function formatDate(iso: string): string {
  if (!iso) return "";
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

export function pnlColor(value: number): string {
  if (value > 0) return "text-accent";
  if (value < 0) return "text-loss";
  return "text-muted";
}

export function pnlBg(value: number): string {
  if (value > 0) return "bg-accent/10";
  if (value < 0) return "bg-loss/10";
  return "bg-surface";
}

export function relativeTime(iso: string): string {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
