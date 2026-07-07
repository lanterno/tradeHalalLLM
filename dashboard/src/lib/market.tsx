/* eslint-disable react-refresh/only-export-components --
 * This is the market context module: it deliberately co-locates the provider
 * component with its `useMarket` hook and the `entityLabel` helper. The
 * fast-refresh-only rule wants one component per file, which doesn't apply to
 * a shared context module.
 */
import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from "react";
import type { Market } from "../api/types";

// The dashboard serves two bots (stocks + crypto) off the same API, which
// discriminates every market-aware endpoint on a ``market`` query param. The
// backend defaults to crypto for back-compat, so the SPA must send the choice
// explicitly or the (stock) operator sees the empty crypto tables. This
// provider holds the current market, persists it, and defaults to **stocks**
// (the only live bot today).

const STORAGE_KEY = "ht.market";

function initialMarket(): Market {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "stocks" || v === "crypto") return v;
  } catch {
    // localStorage unavailable (SSR/private mode) — fall through to default.
  }
  return "stocks";
}

interface MarketContextValue {
  market: Market;
  setMarket: (m: Market) => void;
}

const MarketContext = createContext<MarketContextValue | null>(null);

export function MarketProvider({ children }: { children: ReactNode }) {
  const [market, setMarketState] = useState<Market>(initialMarket);
  const setMarket = useCallback((m: Market) => {
    setMarketState(m);
    try {
      localStorage.setItem(STORAGE_KEY, m);
    } catch {
      // Persistence is best-effort; the in-memory value still switches.
    }
  }, []);
  return (
    <MarketContext.Provider value={{ market, setMarket }}>
      {children}
    </MarketContext.Provider>
  );
}

export function useMarket(): MarketContextValue {
  const ctx = useContext(MarketContext);
  if (!ctx) {
    throw new Error("useMarket must be used within a MarketProvider");
  }
  return ctx;
}

/** Column/label word for the per-asset key: "Symbol" for stocks, "Pair" for crypto. */
export function entityLabel(market: Market): string {
  return market === "stocks" ? "Symbol" : "Pair";
}
