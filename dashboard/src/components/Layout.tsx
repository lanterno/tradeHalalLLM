import { NavLink, Outlet } from "react-router-dom";
import {
  LayoutDashboard,
  Crosshair,
  ArrowLeftRight,
  BarChart3,
  Radar,
  Brain,
  Settings,
  Activity,
  Gauge,
  ShieldAlert,
  Microscope,
  Star,
  Telescope,
} from "lucide-react";
import { cn } from "../lib/utils";
import { useMarket } from "../lib/market";
import type { Market } from "../api/types";
import { useHealth } from "../hooks/useSystem";

const NAV_ITEMS = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard", end: true },
  { to: "/recommendation", icon: Star, label: "Stock of the Day" },
  { to: "/beliefs", icon: Telescope, label: "Belief Board" },
  { to: "/positions", icon: Crosshair, label: "Positions" },
  { to: "/trades", icon: ArrowLeftRight, label: "Trades" },
  { to: "/analytics", icon: BarChart3, label: "Analytics" },
  { to: "/sentiment", icon: Radar, label: "Sentiment" },
  { to: "/decisions", icon: Brain, label: "Decisions" },
  { to: "/risk", icon: ShieldAlert, label: "Risk & Halt" },
  { to: "/insights", icon: Microscope, label: "Insights" },
  { to: "/observability", icon: Gauge, label: "Observability" },
  { to: "/system", icon: Settings, label: "System" },
] as const;

const MARKETS: readonly Market[] = ["stocks", "crypto"] as const;

export function Layout() {
  const { data: health } = useHealth();
  const isLive = health?.status === "running";
  const { market, setMarket } = useMarket();

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 shrink-0 flex flex-col border-r border-border bg-surface">
        {/* Brand */}
        <div className="flex items-center gap-2.5 px-5 py-5">
          <Activity className="h-6 w-6 text-accent" />
          <span className="text-lg font-bold tracking-tight text-white">
            Halal Trader
          </span>
        </div>

        {/* Market switch — which bot's data the market-aware pages show */}
        <div className="px-3 pb-3">
          <div
            role="tablist"
            aria-label="Market"
            className="flex gap-0.5 rounded-lg border border-border bg-bg p-0.5"
          >
            {MARKETS.map((m) => (
              <button
                key={m}
                role="tab"
                aria-selected={market === m}
                onClick={() => setMarket(m)}
                className={cn(
                  "flex-1 rounded-md px-2 py-1 text-xs font-medium capitalize transition-colors",
                  market === m
                    ? "bg-accent/15 text-accent"
                    : "text-muted hover:text-white hover:bg-surface-hover",
                )}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 flex flex-col gap-0.5 px-3 py-2">
          {NAV_ITEMS.map(({ to, icon: Icon, label, ...rest }) => (
            <NavLink
              key={to}
              to={to}
              end={"end" in rest}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-accent/10 text-accent"
                    : "text-muted hover:text-white hover:bg-surface-hover",
                )
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Status */}
        <div className="border-t border-border px-5 py-4">
          <div className="flex items-center gap-2 text-xs">
            <span
              className={cn(
                "h-2 w-2 rounded-full",
                isLive ? "bg-accent animate-pulse" : "bg-muted",
              )}
            />
            <span className={isLive ? "text-accent" : "text-muted"}>
              {isLive ? "Bot Running" : "Offline"}
            </span>
          </div>
          {health && (
            <p className="mt-1 text-[10px] text-muted">v{health.version}</p>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
