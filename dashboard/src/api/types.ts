/** How the round trips that closed one way did (portfolio/analytics.py:ExitStats). */
export interface ExitStats {
  reason: string;
  trades: number;
  avg_pct: number;
  total_pnl: number;
  win_rate: number;
  avg_hold_minutes: number;
  first_hour_trades: number;
  first_hour_avg_pct: number | null;
}

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
  best_symbol: string;
  worst_symbol: string;
  streak: number;
  streak_type: string;
  by_exit_reason: Record<string, number>;
  exits: ExitStats[];
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
  ending_equity: number | null;
  /** Misnamed in the ledger: the writer stores the equity change, and some old
   *  rows something else. Plot equity_change instead. */
  realized_pnl: number;
  /** Ending minus starting equity; null while the day is open. */
  equity_change: number | null;
  return_pct: number | null;
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
  status: "active" | "disabled" | string;
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
      // "disabled": the component is not configured (the core's daily trade
      // without its own keys), so its old beat is not a fault.
      status?: string;
    }
  >;
}

/**
 * Components whose heartbeat is stale for real: bot alive and none of these is
 * what "healthy" means. A component the backend reports as disabled is skipped.
 */
export function staleComponents(h: HealthStatus): string[] {
  return Object.entries(h.bot ?? {})
    .filter(([name, b]) => !name.startsWith("_") && b.stale && b.status !== "disabled")
    .map(([name]) => name);
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

export interface NonHalalBuy {
  symbol: string;
  day: string;
  verdict: "doubtful" | "not_halal" | "unscreened" | string;
  screen_as_of: string | null;
}

/** One account's quarter, its buys judged by the in-house screen on the trade's day. */
export interface AccountCompliance {
  account: string;
  label: string;
  status: "compliant" | "violation";
  trades_today: number;
  trades_this_month: number;
  trades_this_quarter: number;
  buys_this_quarter: number;
  buy_verdicts: { halal: number; doubtful: number; not_halal: number; unscreened: number };
  non_halal_buys_quarter: number;
  non_halal_buys: NonHalalBuy[];
}

// Halal-compliance summary (GET /api/halal/compliance). DB-backed.
export interface HalalCompliance {
  status: "compliant" | "attention" | "violation" | string;
  is_compliant: boolean;
  // New York calendar days, "YYYY-MM-DD".
  quarter_start: string;
  month_start: string;
  today_start: string;
  trades_today: number;
  trades_this_month: number;
  trades_this_quarter: number;
  non_halal_fills_quarter: number;
  accounts: AccountCompliance[];
  // This quarter's.
  purification_accrued_usd: number;
  purification_disbursed_usd: number;
  // Everything still owed, whenever it accrued.
  purification_outstanding_usd: number;
  purification_unpaid_by_account: Record<string, number>;
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

/** The core portfolio's risk (GET /api/risk/core). Weights are of equity. */
export interface CoreRisk {
  available: boolean;
  source?: "snapshot" | "ledger";
  as_of?: string | null;
  age_seconds?: number | null;
  equity?: number | null;
  cash?: number | null;
  cash_pct?: number | null;
  positions?: number;
  top10_weight?: number | null;
  largest?: { symbol: string; weight: number | null } | null;
  sectors?: { sector: string; weight: number | null }[];
  failing_screen?: string[];
  drawdown_pct?: number | null;
  peak_equity?: number;
  peak_day?: string | null;
  history_from?: string | null;
  history_days?: number;
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
  // "stocks": the only market since crypto was removed.
  market: string;
  symbol: string;
  db_quantity: number;
  broker_quantity: number;
  drift_pct: number;
  drift_usd: number | null;
  notes: string | null;
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
  /** The name a reader knows the source by ("Relative strength"). */
  label: string;
  direction: number;
  weight: number;
  detail: string;
  /** The evidence as a sentence ("Outperforming SPY by 4.3%"). */
  plain: string;
}

/** What the engine would do with a name now (server-computed). */
export type Stance = "long" | "leaning" | "none" | "excluded" | "benchmark";

/** The strict in-house screen's verdict on a name. */
export type StrictVerdict = "halal" | "not_halal" | "doubtful" | "unscreened";

export interface ShadowHolding {
  weight: number;
  entry_price: number;
  last_price: number;
  return_pct: number;
  opened_at: string;
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
  /** The strongest evidence for and against, in plain words. */
  main_reason: string | null;
  counter_reason: string | null;
  /** A note when a very high RSI is counted in the name's favour. */
  caution: string | null;
  last_updated: string | null;
  price: number | null;
  price_at: string | null;
  strict: StrictVerdict;
  /** The core account's weight in the name (null: not held, or no snapshot). */
  core_weight: number | null;
  shadow: ShadowHolding | null;
  stance: Stance;
}

export interface BeliefBoard {
  available: boolean;
  beliefs: Belief[];
  entry_band: number;
  exit_band: number;
  benchmark: string;
  screen_as_of: string | null;
  screen_stale: boolean;
  core_as_of: string | null;
}

export interface CohortRecord {
  closed: number;
  wins: number;
  win_rate: number | null;
  mean_return_pct: number | null;
}

export interface ShadowBookPosition extends ShadowHolding {
  asset: string;
  marked_at: string;
  strict: StrictVerdict;
}

export interface MacroEvent {
  kind: string;
  plain: string;
  scheduled_for: string;
  expected_impact: number;
  detail: string;
  names: number;
}

export interface BeliefOverview {
  available: boolean;
  cohort: {
    cohort: number;
    started: string | null;
    current: CohortRecord;
    earlier: CohortRecord;
    median_hold_s: number | null;
  };
  verdict: {
    status: "unproven" | "beating" | "not_beating";
    label: string;
    age_days: number;
    min_closed: number;
    min_days: number;
  };
  baseline_win_rate: number;
  baseline_note: string;
  baseline_measured: boolean;
  baseline_trades: number;
  calibration: {
    status: "fitted" | "identity" | "unknown";
    scored_24h: number;
    moved_24h: number;
    samples: number;
    min_samples: number;
  };
  book: {
    positions: ShadowBookPosition[];
    invested: number;
    cash: number;
    return_pct: number | null;
    contribution_pct: number;
  };
  engine: {
    status: "live" | "stale" | "missing";
    heartbeat_at: string | null;
    heartbeat_age_s: number | null;
    last_bar_at: string | null;
    last_event_at: string | null;
    names: number;
    refreshed_at: string | null;
  };
  llm: { per_day_usd: number | null; days: number; today_usd: number };
  calendar: MacroEvent[];
  entry_band: number;
  exit_band: number;
}

export interface ShadowDecision {
  id: string;
  type: string;
  asset: string | null;
  ts: string;
  source: string;
  payload: Record<string, unknown>;
  correlation_id: string | null;
  /** Why it was proposed, in plain words. */
  plain: string;
  /** Proposed while the regular session was shut: no fill was possible. */
  outside_session: boolean;
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
  name: string | null;
  sector: string;
  shares: number | null;
  value: number | null;
  weight: number | null;
  target: number | null;
  drift: number | null;
  band: number | null; // how far it may drift from its target before it trades
  in_band: boolean | null;
  today: number | null;
  since_buy: number | null;
  verdict: string | null;
  to_sell: boolean;
  reason: string | null;
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
  screen_method: string | null;
  status: string;
  fill_price: number | null;
  filled_qty: number | null;
  fill_status: string | null;
  close: number | null;
  vs_arrival_bps: number | null;
  vs_close_bps: number | null;
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
  histogram: { from_bps: number; to_bps: number; value: number }[];
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
  notes: string[];
  notional: number | null;
  filled: number;
  vs_arrival_bps: number | null;
  vs_close_bps: number | null;
  order_rows: CoreOrder[];
}

export interface CoreSector {
  sector: string;
  value: number | null;
  weight: number | null;
  count: number;
  names: string[];
}

export interface CoreStatus {
  now: string;
  enabled: boolean;
  paper: boolean;
  account: string;
  equity: number | null;
  equity_day: string | null;
  equity_source: "live" | "ledger";
  cash: number | null;
  invested: number | null;
  change: number | null;
  change_pct: number | null;
  today_vs: { symbol: string; change_pct: number | null; diff_pts: number | null }[];
  since: {
    since: string;
    start_equity: number | null;
    change: number | null;
    change_pct: number | null;
    book_pct: number | null;
    spus_pct?: number | null;
    hlal_pct?: number | null;
    spy_pct?: number | null;
  } | null;
  positions: number;
  top10_weight: number | null;
  to_sell: { symbol: string; name: string | null; reason: string | null }[];
  screen_as_of: string | null;
  next_check: string;
  next_rebalance: string;
  monthly_due: boolean;
  holdings: CoreHolding[];
  sectors: CoreSector[];
  series: {
    date: string;
    account: number | null;
    book: number | null;
    spus?: number | null;
    hlal?: number | null;
    spy?: number | null;
  }[];
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
    partial: number;
    halted: number;
    missing_runs: string[];
    failures: string[];
    window: { day: string; status: "run" | "missed" | "pending" | "before" }[];
    earliest: string;
    earliest_days: string;
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
    zakat: {
      amount: number;
      chosen: string;
      estimate: boolean; // valued today, before the hawl: not yet owed
      as_of: string;
      next_hawl: string;
      next_hawl_hijri: string;
    } | null;
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

// ── Operations (/api/operations, web/operations.py) ──

export type OpsStatus = "ok" | "stale" | "missing" | "disabled" | "unknown";

export interface OpsJob {
  component: string;
  label: string;
  at: string; // "15:40" ET
  weekday: string | null; // "Fri" for a weekly job
  today_at: string | null;
  due_today: boolean;
  last: string | null;
  ran_today: boolean;
  summary: string | null;
  next: string | null;
  status: OpsStatus;
  reason: string | null;
}

export interface OpsProcess {
  component: string;
  label: string;
  cadence: string;
  due: boolean;
  last: string | null;
  age_seconds: number | null;
  status: OpsStatus;
  reason: string | null;
  detail: Record<string, unknown> | null;
}

export interface OpsPool {
  pool: string;
  members: string[];
  today: number;
  daily_cap: number | null;
  month: number;
  monthly_cap: number | null;
  pace: number;
}

export interface OpsConsumer {
  consumer: string;
  pool: string;
  today: number;
  calls_today: number;
  yesterday: number;
  month: number;
  calls_month: number;
}

export interface OpsBackup {
  at: string | null;
  detail: Record<string, unknown> | null;
  status: "ok" | "stale" | "missing";
}

export interface OpsFreshness {
  name: string;
  as_of: string | null;
  detail: string;
  status: "ok" | "stale" | "unknown" | "info";
}

export interface OpsTable {
  name: string;
  bytes: number;
  rows: number;
  dead: number;
  vacuumed: string | null;
  shadow: boolean;
  dumped: boolean;
}

export interface OperationsStatus {
  now: string;
  paper: boolean;
  market: {
    today: string;
    trading_day: boolean;
    open: boolean;
    opens: string | null;
    closes: string | null;
    next_open: string;
  };
  fleet: {
    verdict: "healthy" | "degraded" | "down";
    alive: boolean;
    reason: string | null;
    beating: number;
    jobs_due: number;
    jobs_ran: number;
    problems: string[];
    watchdog: { suspect?: string[]; alerting?: string[] } | null;
  };
  halt: { enabled: boolean; reason: string | null; set_by: string | null; set_at: string | null };
  jobs: OpsJob[];
  processes: OpsProcess[];
  llm: {
    day: string;
    today: number;
    calls_today: number;
    enforced: boolean;
    pools: OpsPool[];
    consumers: OpsConsumer[];
    days: ({ day: string } & Record<string, number | string>)[];
  };
  backups: {
    nightly: OpsBackup;
    offsite: OpsBackup;
    drill: OpsBackup;
    max_age_hours: number;
    drill_max_age_days: number;
    next_nightly: string;
    next_drill: string;
  };
  freshness: OpsFreshness[];
  database: {
    bytes: number;
    version: string;
    tables: number;
    connections: number;
    active: number;
    dead_ratio: number | null;
    xid_age: number;
    revision: string | null;
    shadow_share: number;
    biggest: OpsTable[];
    rest: { count: number; bytes: number };
    dead_most: OpsTable[];
    shadow_events: { day: string; events: number }[];
  };
  deploy: {
    version: string;
    revision: string | null;
    expected_revision: string;
    schema_ok: boolean;
    web_started: string | null;
  };
  config: { core: [string, string][]; llm: [string, string][]; day_trader: [string, string][] };
}
