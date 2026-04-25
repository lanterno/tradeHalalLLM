/**
 * Lightweight shimmer shown while a lazy-loaded route bundle is fetching.
 * Plain CSS gradient so it renders instantly with no extra deps.
 */
export function PageSkeleton() {
  return (
    <div className="p-6 space-y-4">
      <div className="h-8 w-48 rounded-md bg-surface-hover animate-pulse" />
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="h-24 rounded-md bg-surface-hover animate-pulse"
          />
        ))}
      </div>
      <div className="h-64 rounded-md bg-surface-hover animate-pulse" />
    </div>
  );
}
