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
  pair: string;
  side: string;
  quantity: number;
  price: number;
  order_id: string;
  exchange: string;
  status: string;
  llm_reasoning: string;
  entry_price: number | null;
  stop_loss: number | null;
  target_price: number | null;
  exit_price: number | null;
  exit_reason: string | null;
  closed_at: string | null;
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
  pair: string;
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

export interface LlmProviderStats {
  calls: number;
  tokens: number;
  p50_ms: number;
}

export interface LlmMetrics {
  window_seconds: number;
  calls: number;
  total_tokens: number;
  p50_ms: number | null;
  p95_ms: number | null;
  by_provider: Record<string, LlmProviderStats>;
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
