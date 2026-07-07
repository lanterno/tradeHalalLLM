// Which bot's data a market-aware endpoint should return. The backend
// discriminates trades/positions/analytics/pnl on this; the SPA sends it
// explicitly (see lib/market.tsx) rather than relying on the crypto default.
export type Market = "stocks" | "crypto";

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
  // Crypto rows carry ``pair``; stocks rows carry ``symbol``. Use
  // ``entityOf()`` (lib/utils) to read whichever the row has.
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
  // Stocks-side fill fields (absent on crypto rows).
  filled_price?: number | null;
  filled_quantity?: number | null;
  submitted_at?: string | null;
  filled_at?: string | null;
  entry_type?: string | null;
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
  // Stocks positions are aliased ``pair = symbol`` by the backend; crypto
  // rows carry a native ``pair``. ``symbol`` is present on stocks rows.
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

export interface SentimentSignal {
  pair: string;
  score: number;
  buzz: number;
  confidence: number;
  top_narratives: string[];
  news_headlines: string[];
  data_sources: string[];
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  version: string;
}

export interface SystemStatus {
  bot_running: boolean;
  last_cycle: string | null;
  cycle_interval_seconds: number;
  ws_health: Record<string, unknown>;
  uptime_seconds: number | null;
}

export interface AppConfig {
  llm_provider: string;
  llm_model: string;
  crypto_pairs: string[];
  crypto_trading_interval_seconds: number;
  crypto_max_position_pct: number;
  crypto_daily_loss_limit: number;
  crypto_daily_return_target: number;
  db_path: string;
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
