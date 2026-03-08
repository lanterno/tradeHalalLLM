import { useSentiment } from "../hooks/useSentiment";
import { SentimentGauge } from "../components/SentimentGauge";

export default function Sentiment() {
  const { data: signals, isLoading } = useSentiment();

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold text-white">Sentiment</h1>

      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="h-20 animate-pulse rounded-xl border border-border bg-surface"
            />
          ))}
        </div>
      ) : !signals?.length ? (
        <div className="rounded-xl border border-border bg-surface p-8">
          <p className="text-center text-sm text-muted">
            No sentiment data available. Sentiment feeds may not be running.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Buzz alerts */}
          {signals.some((s) => s.buzz > 2) && (
            <div className="rounded-xl border border-yellow-400/30 bg-yellow-400/5 p-4">
              <h3 className="mb-2 text-sm font-semibold text-yellow-400">
                High Buzz Alert
              </h3>
              <div className="flex flex-wrap gap-2">
                {signals
                  .filter((s) => s.buzz > 2)
                  .map((s) => (
                    <span
                      key={s.pair}
                      className="rounded-md bg-yellow-400/10 px-2 py-1 text-xs font-medium text-yellow-400"
                    >
                      {s.pair} ({s.buzz.toFixed(1)}x buzz)
                    </span>
                  ))}
              </div>
            </div>
          )}

          {/* Sentiment table */}
          <div className="rounded-xl border border-border bg-surface p-4">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                    <th className="px-3 py-2">Pair</th>
                    <th className="px-3 py-2">Score</th>
                    <th className="px-3 py-2 text-right">Buzz</th>
                    <th className="px-3 py-2 text-right">Confidence</th>
                    <th className="px-3 py-2">Sources</th>
                    <th className="px-3 py-2">Headlines</th>
                  </tr>
                </thead>
                <tbody>
                  {signals.map((s) => (
                    <tr
                      key={s.pair}
                      className="border-b border-border/50 hover:bg-surface-hover/50 transition-colors"
                    >
                      <td className="px-3 py-2 font-medium">{s.pair}</td>
                      <td className="px-3 py-2">
                        <SentimentGauge score={s.score} />
                      </td>
                      <td className="px-3 py-2 text-right font-mono">
                        <span
                          className={
                            s.buzz > 2
                              ? "text-yellow-400"
                              : s.buzz > 1
                                ? "text-white"
                                : "text-muted"
                          }
                        >
                          {s.buzz.toFixed(1)}x
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-muted">
                        {(s.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex gap-1">
                          {s.data_sources.map((src) => (
                            <span
                              key={src}
                              className="rounded bg-surface-hover px-1.5 py-0.5 text-[10px] font-medium text-muted"
                            >
                              {src}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="max-w-xs px-3 py-2">
                        {s.news_headlines.length > 0 ? (
                          <ul className="space-y-0.5">
                            {s.news_headlines.slice(0, 2).map((h, i) => (
                              <li
                                key={i}
                                className="truncate text-xs text-muted"
                                title={h}
                              >
                                {h}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <span className="text-xs text-muted">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Narratives */}
          {signals.some((s) => s.top_narratives.length > 0) && (
            <div className="rounded-xl border border-border bg-surface p-4">
              <h3 className="mb-4 text-sm font-medium uppercase tracking-wider text-muted">
                Top Narratives
              </h3>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {signals
                  .filter((s) => s.top_narratives.length > 0)
                  .map((s) => (
                    <div
                      key={s.pair}
                      className="rounded-lg border border-border/50 p-3"
                    >
                      <div className="mb-2 flex items-center justify-between">
                        <span className="font-medium">{s.pair}</span>
                        <SentimentGauge score={s.score} size="sm" />
                      </div>
                      <ul className="space-y-1">
                        {s.top_narratives.map((n, i) => (
                          <li key={i} className="text-xs text-muted">
                            {n}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
