import { cn } from "../lib/utils";

interface SentimentGaugeProps {
  score: number;
  size?: "sm" | "md";
}

export function SentimentGauge({ score, size = "md" }: SentimentGaugeProps) {
  const pct = ((score + 1) / 2) * 100;
  const label =
    score > 0.3 ? "Bullish" : score < -0.3 ? "Bearish" : "Neutral";
  const color =
    score > 0.3
      ? "text-accent bg-accent"
      : score < -0.3
        ? "text-loss bg-loss"
        : "text-yellow-400 bg-yellow-400";

  return (
    <div className="flex items-center gap-2">
      <div
        className={cn(
          "relative overflow-hidden rounded-full bg-surface-hover",
          size === "sm" ? "h-1.5 w-16" : "h-2 w-24",
        )}
      >
        <div
          className={cn("absolute inset-y-0 left-0 rounded-full", color.split(" ")[1])}
          style={{ width: `${pct}%`, opacity: 0.6 }}
        />
      </div>
      <span
        className={cn(
          "text-xs font-medium",
          color.split(" ")[0],
          size === "sm" && "text-[10px]",
        )}
      >
        {score > 0 ? "+" : ""}
        {score.toFixed(2)} {label}
      </span>
    </div>
  );
}
