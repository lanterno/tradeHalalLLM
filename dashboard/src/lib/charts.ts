// Shared Recharts styling so all charts read as one system. These mirror the
// CSS theme tokens in app.css (Recharts needs literal color strings, so we
// can't hand it the CSS vars directly — keep the two in sync).

export const CHART = {
  accent: "#4ade80", // --color-accent (positive / P&L up)
  loss: "#f87171", // --color-loss (negative / P&L down)
  warning: "#facc15", // --color-warning
  muted: "#6b7280", // --color-muted (axis ticks / legend)
  gridStroke: "#1a1a2e", // --color-border
  axisText: "#e0e0e0",
} as const;

/** Categorical palette for pies / multi-series (accent-led, brand-consistent). */
export const CHART_COLORS = [
  "#4ade80",
  "#60a5fa",
  "#facc15",
  "#c084fc",
  "#fb923c",
  "#f87171",
] as const;

/** Positive/negative fill for P&L bars. */
export function pnlFill(value: number): string {
  return value >= 0 ? CHART.accent : CHART.loss;
}

/** Recharts <Tooltip contentStyle> — dark card matching --color-surface/border. */
export const CHART_TOOLTIP = {
  background: "#111118",
  border: "1px solid #1a1a2e",
  borderRadius: 8,
  fontSize: 12,
} as const;

/** Recharts axis <XAxis/YAxis tick> style. */
export const AXIS_TICK = { fill: CHART.muted, fontSize: 11 } as const;
