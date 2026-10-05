import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  Crosshair,
  ArrowLeftRight,
  BarChart3,
  Brain,
  Settings,
  Activity,
  Gauge,
  ShieldAlert,
  ShieldCheck,
  Microscope,
  Star,
  PieChart,
  Telescope,
  Menu,
  X,
} from "lucide-react";
import { cn } from "../lib/utils";
import { useHealth } from "../hooks/useSystem";
import { ErrorBoundary } from "./ErrorBoundary";

const NAV_ITEMS = [
  { to: "/", icon: LayoutDashboard, label: "Home", end: true },
  { to: "/core", icon: PieChart, label: "Core portfolio" },
  { to: "/recommendation", icon: Star, label: "Stock of the Day" },
  { to: "/beliefs", icon: Telescope, label: "Belief Board" },
  { to: "/positions", icon: Crosshair, label: "Positions" },
  { to: "/trades", icon: ArrowLeftRight, label: "Trades" },
  { to: "/analytics", icon: BarChart3, label: "Analytics" },
  { to: "/decisions", icon: Brain, label: "Decisions" },
  { to: "/halal", icon: ShieldCheck, label: "Halal" },
  { to: "/risk", icon: ShieldAlert, label: "Risk & Halt" },
  { to: "/insights", icon: Microscope, label: "Insights" },
  { to: "/observability", icon: Gauge, label: "Observability" },
  { to: "/system", icon: Settings, label: "System" },
] as const;

function StatusDot({ isLive }: { isLive: boolean }) {
  return (
    <span
      className={cn(
        "h-2 w-2 rounded-full",
        isLive ? "bg-accent animate-pulse" : "bg-muted",
      )}
    />
  );
}

export function Layout() {
  const { data: health } = useHealth();
  const isLive = health?.status === "running";
  const location = useLocation();
  // Phones: the sidebar is a drawer behind a top bar. It is open only on the
  // page it was opened on, so any navigation (a link, the back button) closes it.
  const [openOn, setOpenOn] = useState<string | null>(null);
  const menuOpen = openOn === location.pathname;
  const setMenuOpen = (open: boolean) => setOpenOn(open ? location.pathname : null);

  return (
    <div className="flex h-dvh flex-col overflow-hidden md:flex-row">
      {/* Top bar (phones only) */}
      <header className="flex shrink-0 items-center justify-between border-b border-border bg-surface px-4 py-3 md:hidden">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-accent" />
          <span className="font-bold tracking-tight text-white">Halal Trader</span>
          <StatusDot isLive={isLive} />
        </div>
        <button
          type="button"
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen(!menuOpen)}
          className="rounded-lg p-2 text-muted hover:bg-surface-hover hover:text-white"
        >
          {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </header>

      {/* Backdrop behind the open drawer (phones only) */}
      {menuOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 md:hidden"
          onClick={() => setMenuOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar: a fixed column from md up, a slide-in drawer below */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-border bg-surface transition-transform duration-200",
          "md:static md:z-auto md:w-56 md:shrink-0 md:translate-x-0",
          menuOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {/* Brand */}
        <div className="flex items-center gap-2.5 px-5 py-5">
          <Activity className="h-6 w-6 text-accent" />
          <span className="text-lg font-bold tracking-tight text-white">
            Halal Trader
          </span>
        </div>

        {/* Nav */}
        <nav className="flex flex-1 flex-col gap-0.5 overflow-y-auto px-3 py-2">
          {NAV_ITEMS.map(({ to, icon: Icon, label, ...rest }) => (
            <NavLink
              key={to}
              to={to}
              end={"end" in rest}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors md:py-2",
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
            <StatusDot isLive={isLive} />
            <span className={isLive ? "text-accent" : "text-muted"}>
              {isLive ? "Bot Running" : "Offline"}
            </span>
          </div>
          {health && (
            <p className="mt-1 text-[10px] text-muted">v{health.version}</p>
          )}
        </div>
      </aside>

      {/* Main content — an ErrorBoundary keyed by path so one crashed page
          shows a recoverable error (sidebar survives) and navigating away
          resets it. */}
      <main className="min-w-0 flex-1 overflow-y-auto">
        <ErrorBoundary key={location.pathname}>
          <Outlet />
        </ErrorBoundary>
      </main>
    </div>
  );
}
