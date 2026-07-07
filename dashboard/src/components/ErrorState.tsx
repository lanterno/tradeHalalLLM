import { AlertTriangle } from "lucide-react";
import { cn } from "../lib/utils";

interface ErrorStateProps {
  error?: unknown;
  onRetry?: () => void;
  className?: string;
  /** Compact variant for inline panels (no vertical padding). */
  compact?: boolean;
}

/**
 * Uniform "failed to load" state so a 5xx / network error is visually
 * distinct from an empty result or a perpetual loading spinner. Used on
 * every page's `isError` branch and by the ErrorBoundary.
 */
export function ErrorState({ error, onRetry, className, compact }: ErrorStateProps) {
  const message =
    error instanceof Error ? error.message : "The request failed. Try again.";
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-center justify-center gap-2 text-center",
        compact ? "py-6" : "py-12",
        className,
      )}
    >
      <AlertTriangle className="h-6 w-6 text-warning" aria-hidden />
      <p className="text-sm font-medium text-loss">Failed to load</p>
      <p className="max-w-md text-xs text-muted">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-1 rounded-lg border border-border px-3 py-1 text-xs font-medium text-muted transition-colors hover:bg-surface-hover hover:text-white"
        >
          Retry
        </button>
      )}
    </div>
  );
}
