export interface AnalyticsStats {
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_win_pct: number;
  avg_loss_pct: number;
  total_pnl: number;
  profit_factor: number;
  max_drawdown_pct: number;
  avg_hold_minutes: number;
  best_pair: string;
  worst_pair: string;
  streak: number;
  streak_type: string;
  by_exit_reason: Record<string, number>;
}

export interface Trade {
  id: number;
  timestamp: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  order_id: string;
  status: string;
  llm_reasoning: string;
  stop_loss: number | null;
  target_price: number | null;
  exit_price: number | null;
  exit_reason: string | null;
  closed_at: string | null;
  // Fill fields.
  filled_price?: number | null;
  filled_quantity?: number | null;
  submitted_at?: string | null;
  filled_at?: string | null;
  entry_type?: string | null;
  paper_slippage_pct?: number | null;
  predicted_slippage_pct?: number | null;
  live_slippage_pct?: number | null;
}

export interface DailyPnl {
  id: number;
  date: string;
  starting_equity: number;
  ending_equity: number;
  realized_pnl: number;
  return_pct: number;
  trades_count: number;
}

export interface LlmDecision {
  id: number;
  timestamp: string;
  provider: string;
  model: string;
  prompt_summary: string;
  raw_response: string;
  parsed_action: string;
  symbols: string;
  execution_ms: number;
}

export interface StrategyAdjustment {
  id: number;
  timestamp: string;
  parameter: string;
  old_value: string;
  new_value: string;
  reasoning: string;
}

/** One holding, marked by the broker (GET /api/positions). */
export interface Holding {
  symbol: string;
  qty: number | null;
  avg_entry: number | null;
  price: number | null;
  market_value: number | null;
  cost_basis: number | null;
  unrealized_pl: number | null;
  unrealized_pl_pct: number | null;
  change_today: number | null; // a fraction: 0.012 is +1.2%
  weight: number | null; // of the account's equity
  // The day-trader's own exit levels (its ledger); absent on the core.
  stop_loss?: number | null;
  target_price?: number | null;
  opened_at?: string;
}

export interface AccountPositions {
  account: "core" | "paper" | string;
  label: string;
  status: "active" | "disabled" | "retired" | string;
  /** "snapshot": the broker's marks, as of `as_of`; "ledger": fills at the last close. */
  source: "snapshot" | "ledger";
  as_of: string | null;
  age_seconds: number | null;
  equity: number | null;
  cash: number | null;
  invested: number;
  unrealized_pl: number | null;
  positions: Holding[];
}

export interface PositionsResponse {
  accounts: AccountPositions[];
}

export interface HealthStatus {
  status: string; // the web process; always "running" while it answers
  timestamp: string;
  version: string;
  /** The bot's own heartbeat verdict: the real "is it running?". */
  bot_alive?: boolean;
  bot?: Record<
    string,
    {
      stale?: boolean;
      age_seconds?: number;
      reason?: string | null;
      // "disabled": the component is switched off on purpose (e.g. the retired
      // day-trader's cycle), so its old beat is not a fault.
      status?: string;
    }
  >;
}

/** The day-trader's cycle beat; it stops for good when the day-trader is retired. */
export const DAY_TRADER_CYCLE = "stock.cycle";

/**
 * Components whose heartbeat is stale for real: bot alive and none of these is
 * what "healthy" means. A component the backend reports as disabled is skipped,
 * and so is the day-trader's cycle while the day-trader is switched off (an
 * older backend reports that beat as plainly stale).
 */
export function staleComponents(
  h: HealthStatus,
  opts: { dayTraderEnabled?: boolean } = {},
): string[] {
  return Object.entries(h.bot ?? {})
    .filter(([name, b]) => {
      if (name.startsWith("_") || !b.stale) return false;
      if (b.status === "disabled") return false;
      if (name === DAY_TRADER_CYCLE && opts.dayTraderEnabled === false) return false;
      return true;
    })
    .map(([name]) => name);
}

export interface SystemStatus {
  bot_running: boolean;
  last_cycle: string | null;
  stocks_cycle_interval_seconds: number;
  uptime_seconds: number | null;
  day_trader_enabled?: boolean;
  core_enabled?: boolean;
}

export interface AppConfig {
  llm_provider: string;
  llm_model: string;
  stocks_trading_interval_minutes: number;
  stocks_max_position_pct: number;
  stocks_daily_loss_limit: number;
  stocks_daily_return_target: number;
  database: string;
}

export interface CycleMetrics {
  window_seconds: number;
  count: number;
  p50_ms: number | null;
  p95_ms: number | null;
  p99_ms: number | null;
  failed: number;
  halted: number;
}

export interface LlmMetrics {
  window_seconds: number;
  calls: number;
  total_tokens: number;
  total_cost_usd: number;
  p50_ms: number | null;
  p95_ms: number | null;
}

// AAOIFI halal-compliance summary (GET /api/halal/compliance). DB-backed.
export interface HalalCompliance {
  status: "compliant" | "attention" | "violation" | string;
  is_compliant: boolean;
  quarter_start: string;
  month_start: string;
  today_start: string;
  trades_today: number;
  trades_this_month: number;
  trades_this_quarter: number;
  halal_screenings_quarter: number;
  doubtful_screenings_quarter: number;
  not_halal_screenings_quarter: number;
  non_halal_fills_quarter: number;
  purification_accrued_usd: number;
  purification_disbursed_usd: number;
  purification_outstanding_usd: number;
}

// A guard/rejection: the cycle proposed a trade but a guard blocked it.
export interface RejectionRow {
  timestamp: string;
  cycle_id: string | null;
  symbol: string | null;
  reason: string;
  category: string;
}

export interface RiskState {
  available: boolean;
  is_halted?: boolean;
  halt_reason?: string | null;
  portfolio_heat_pct?: number | null;
  drawdown_pct?: number | null;
  avg_correlation?: number | null;
  summary?: string;
  // Which bot wrote the snapshot (always "stocks" now).
  market?: string;
  // ISO timestamp of when the cycle wrote this snapshot — useful
  // for surfacing staleness if the cycle has stopped running.
  pushed_at?: string;
}

export interface HaltStatus {
  enabled: boolean;
  reason: string | null;
  set_by: string | null;
  set_at: string | null;
}

export interface ReconcileLogRow {
  id: number;
  timestamp: string;
  // A free string: rows written before 2026-10-01 may name "crypto".
  market: string;
  symbol: string;
  db_quantity: number;
  broker_quantity: number;
  drift_pct: number;
  drift_usd: number | null;
  notes: string | null;
}

export interface BackupRow {
  path: string;
  size_bytes: number;
  backed_up_at: string;
}

// Daily halal "stock of the day" recommendation (advisory — never traded).
// The latest endpoint returns { available: false } when none has been
// generated yet; otherwise available is true and the fields are populated.
// Per-candidate quantitative range grounding, stored in the `candidates`
// JSONB at generation time (see quant/outlook + expected_move). Only the
// fields the UI reads are typed; the object carries more.
export interface CandidateQuant {
  band5d_lo?: number;
  band5d_hi?: number;
  vol_pctl?: number | null;
  impl_move_pct?: number;
  impl_dte?: number;
  impl_low?: number;
  impl_high?: number;
  quant_bands?: { calibrated?: boolean; calibration_version?: string | null };
  [key: string]: unknown;
}

export interface StockOfTheDay {
  available: boolean;
  id?: number;
  date?: string;
  symbol?: string;
  conviction?: number;
  thesis?: string;
  halal_note?: string;
  suggested_entry?: number | null;
  suggested_target?: number | null;
  suggested_stop?: number | null;
  catalysts?: string | null;
  risks?: string | null;
  universe_size?: number;
  model?: string | null;
  created_at?: string;
  // Per-symbol context the model weighed, incl. quant range fields.
  candidates?: Record<string, CandidateQuant>;
  // Outcome tracking (populated by the scorecard backfill once matured).
  outcome_status?: string;
  fwd_return_1d?: number | null;
  fwd_return_5d?: number | null;
  fwd_return_20d?: number | null;
  benchmark_return_5d?: number | null;
}

// Aggregate track record for the daily recommendation (forward returns).
export interface RecommendationScorecard {
  available: boolean;
  /** Days with a pick (one per day: a re-generated day counts once). */
  n_total: number;
  n_scored: number;
  n_duplicates?: number;
  /** The record in a word; "negative" while picks lag the benchmark or IC < 0. */
  verdict?: "negative" | "unproven" | "positive";
  conviction_ic?: number | null;
  conviction_mode?: { value: number; n: number; of: number } | null;
  sufficient?: boolean;
  min_samples?: number;
  hit_rate_5d?: number;
  avg_fwd_1d?: number | null;
  avg_fwd_5d?: number | null;
  avg_fwd_20d?: number | null;
  avg_excess_5d?: number | null;
  benchmark?: string;
  best?: { symbol: string; date: string; fwd_5d: number };
  worst?: { symbol: string; date: string; fwd_5d: number };
  // Plan quality (LLM suggested_target/stop vs the realized path).
  n_with_levels?: number;
  levels_sufficient?: boolean;
  target_hit_rate?: number | null;
  stop_hit_rate?: number | null;
  avg_mfe_5d?: number | null;
  avg_mae_5d?: number | null;
  avg_plan_return_5d?: number | null;
  // Quant band coverage (stored band vs realized 5d path).
  band_n?: number;
  band_coverage_5d?: number | null;
  candidate_band_n?: number;
  candidate_band_coverage_5d?: number | null;
  // Counterfactual: where the pick ranked among its candidates (0.5 = random).
  pick_percentile_n?: number;
  avg_pick_percentile_5d?: number | null;
}

// ── halabot shadow engine (belief board) ────────────────────────

export interface BeliefCatalyst {
  kind: string;
  scheduled_for: string;
  expected_impact: number;
  detail: string;
}

export interface BeliefEvidence {
  source: string;
  direction: number;
  weight: number;
  detail: string;
}

export interface Belief {
  asset: string;
  version: number;
  regime: string;
  regime_confidence: number;
  direction: string;
  conviction: number;
  conviction_raw: number;
  thesis: string;
  invalidation: number | null;
  stop: number | null;
  support: number | null;
  resistance: number | null;
  horizon: string;
  catalysts_pending: BeliefCatalyst[];
  halal: string | null;
  n_evidence: number;
  top_evidence: BeliefEvidence[];
  last_updated: string | null;
}

export interface BeliefBoard {
  available: boolean;
  beliefs: Belief[];
}

export interface ShadowDecision {
  id: string;
  type: string;
  asset: string | null;
  ts: string;
  source: string;
  payload: Record<string, unknown>;
  correlation_id: string | null;
}

export interface ZakatAssessment {
  period_start: string;
  hawl_date: string;
  market_value: number;
  trade_goods_zakat: number;
  dividends: number;
  purified: number;
  income_zakat: number;
  chosen: "trade goods" | "income";
  amount: number;
}

export interface ZakatStatus {
  configured: boolean;
  source: string;
  hawl_hijri?: string;
  last_hawl?: string;
  next_hawl?: string;
  next_hawl_hijri?: string;
  days_to_next?: number;
  accounts?: ZakatAccount[];
}

export interface ZakatAccount {
  account: string;
  label: string;
  if_due_today: ZakatAssessment;
  last_recorded: {
    hawl_date: string;
    hawl_hijri: string;
    amount: number;
    chosen: string;
  } | null;
}

export interface PurificationLine {
  account: string;
  symbol: string;
  dividends: number;
  amount: number;
  payments: number;
  assumed: number;
}

export interface PurificationYear {
  year: number;
  lines: PurificationLine[];
  dividends: number;
  amount: number;
}

export interface CoreHolding {
  symbol: string;
  shares: number | null;
  value: number | null;
  weight: number | null;
  target: number | null;
  drift: number | null;
}

export interface CoreOrder {
  at: string;
  symbol: string;
  side: string;
  qty: number | null;
  price: number | null;
  notional: number | null;
  reason: string;
  screen_as_of: string | null;
  status: string;
  fill_price: number | null;
  fill_status: string | null;
  vs_arrival_bps: number | null;
}

export interface CoreExecution {
  start: string;
  end: string;
  orders: number;
  filled: number;
  partial: number;
  unfilled: number;
  filled_notional: number;
  vs_arrival_bps: number | null;
  vs_close_bps: number | null;
  book_cost_bps: number;
  cost_vs_close_usd: number | null;
  worst: { symbol: string; side: string; vs_arrival_bps: number | null }[];
}

export interface CoreRun {
  run_on: string;
  monthly: boolean;
  executed: boolean;
  equity: number | null;
  cash: number | null;
  orders: number;
  halted: string | null;
  screen_as_of: string | null;
}

export interface CoreStatus {
  enabled: boolean;
  paper: boolean;
  equity: number | null;
  equity_day: string | null;
  holdings: CoreHolding[];
  series: { date: string; account: number | null; book: number | null }[];
  readiness: {
    ready: boolean;
    days: number;
    min_days: number;
    monthly_runs: number;
    tracking_error: number | null;
    max_tracking_error: number;
    gap: number | null;
    max_gap: number;
    refused: number;
    unfilled: number;
    halted: number;
    failures: string[];
  };
  execution: CoreExecution | null;
  orders: CoreOrder[];
  runs: CoreRun[];
}

export interface HomeAccount {
  account: string;
  label: string;
  status: "active" | "paused" | "disabled";
  paper: boolean;
  equity: number;
  cash: number | null;
  invested: number | null;
  change: number | null;
  change_pct: number | null;
  positions: number;
  symbols: string[];
  as_of: string;
  source: "live" | "ledger" | "run";
}

export interface HomeBenchmark {
  symbol: string;
  price: number;
  change_pct: number | null;
  as_of: string;
  live: boolean;
}

export interface HomeStatus {
  now: string;
  today: string;
  today_hijri: string;
  market: {
    session: Record<string, string>;
    holidays: string[];
    early_closes: string[];
    next_holiday: { date: string; name: string } | null;
    next_early_close: string | null;
    trading_days_month: number;
    trading_days_left: number;
    benchmarks: HomeBenchmark[];
  };
  accounts: HomeAccount[];
  total: {
    equity: number;
    change: number | null;
    change_pct: number | null;
    series: { date: string; equity: number }[];
    change_30d_pct: number | null;
  };
  set_aside: {
    purification_unpaid: number;
    zakat: { amount: number; chosen: string; next_hawl: string; next_hawl_hijri: string } | null;
  };
  portfolio: {
    holdings: {
      symbol: string;
      name: string | null;
      value: number;
      weight: number;
      change_today: number | null;
    }[];
    count: number;
    top_weight: number | null;
    sectors: { sector: string; weight: number }[];
    screen: { as_of: string | null; halal: number; screened: number; failing: string[] };
  };
  upcoming: { at: string; label: string; detail: string }[];
  gate: {
    ready: boolean;
    days: number;
    min_days: number;
    monthly_runs: number;
    tracking_error: number | null;
    max_tracking_error: number;
    gap: number | null;
    max_gap: number;
    clean: boolean;
  };
}
