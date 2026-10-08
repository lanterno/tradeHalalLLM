const BASE = "";
const TOKEN_KEY = "trader-token";

// The dashboard token (WEB_API_TOKEN) every change to the bot needs, kept in
// this browser once it has been typed. Storage can be unavailable (a private
// window); the token is then asked for again next time.
function storedToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function storeToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // nothing to keep it in
  }
}

function withToken(init: RequestInit | undefined, token: string | null): RequestInit | undefined {
  if (!token) return init;
  const headers = new Headers(init?.headers);
  headers.set("X-Trader-Token", token);
  return { ...init, headers };
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const mutation = (init?.method ?? "GET").toUpperCase() !== "GET";
  let res = await fetch(`${BASE}${path}`, mutation ? withToken(init, storedToken()) : init);
  if (mutation && res.status === 401) {
    // No token yet, or a stale one: ask once, then retry with it.
    storeToken(null);
    const token = window.prompt("Changing the bot needs the dashboard token (WEB_API_TOKEN):");
    if (token) {
      res = await fetch(`${BASE}${path}`, withToken(init, token.trim()));
      if (res.ok) storeToken(token.trim());
    }
  }
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}
