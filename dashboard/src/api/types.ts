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
  // Stock rows carry ``symbol``; ``pair`` survives from the old crypto
  // shape. Use ``entityOf()`` (lib/utils) to read whichever the row has.
  pair?: string;
  symbol?: string;
  side: string;
  quantity: number;
  price: number;
  order_id: string;
  exchange?: string;
  status: string;
  llm_reasoning: string;
  entry_price?: number | null;
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

export interface OpenPosition {
  id: number;
  // The backend aliases ``pair = symbol`` on stock positions.
  pair: string;
  symbol?: string;
  quantity: number;
  entry_price: number;
  stop_loss: number | null;
  target_price: number | null;
  timestamp: string;
  current_price?: number;
  unrealized_pnl?: number;
  unrealized_pnl_pct?: number;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  version: string;
}

export interface SystemStatus {
  bot_running: boolean;
  last_cycle: string | null;
  stocks_cycle_interval_seconds: number;
  uptime_seconds: number | null;
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

// ── Phase 2 / 3 surfaces ─────────────────────────────────────

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
  // Which bot wrote the snapshot — populated by the cycle's
  // runtime push so the dashboard can show whose risk this is.
  market?: "crypto" | "stocks" | string;
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
  market: "crypto" | "stocks";
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
  n_total: number;
  n_scored: number;
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
    halted: number;
    failures: string[];
  };
  orders: CoreOrder[];
  runs: CoreRun[];
}
