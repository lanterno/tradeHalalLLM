import { useEffect, useRef, useState, useCallback } from "react";

interface PriceMap {
  [symbol: string]: number;
}

export function usePriceStream(symbols: string[]) {
  const [prices, setPrices] = useState<PriceMap>({} as PriceMap);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  // Set on intentional teardown (unmount, or a symbols change — e.g. switching
  // the market to stocks) so the socket's own onclose doesn't schedule a
  // reconnect that resurrects a stale (crypto) connection.
  const closingRef = useRef(false);

  const connect = useCallback(() => {
    if (!symbols.length) return;
    closingRef.current = false;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const params = symbols.map((s) => `symbols=${s}`).join("&");
    const url = `${protocol}//${host}/ws/prices?${params}`;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      if (!closingRef.current) {
        reconnectTimer.current = setTimeout(connect, 3000);
      }
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as PriceMap;
        setPrices((prev) => ({ ...prev, ...data }));
      } catch {
        // ignore malformed messages
      }
    };
  }, [symbols]);

  useEffect(() => {
    connect();
    return () => {
      closingRef.current = true;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { prices, connected };
}
