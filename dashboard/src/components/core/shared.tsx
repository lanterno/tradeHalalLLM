import { formatPct } from "../../lib/utils";
import { toneOf } from "./format";

export function SignedPct({ v, digits = 2 }: { v: number | null | undefined; digits?: number }) {
  return <span className={toneOf(v)}>{formatPct(v, digits, { signed: true })}</span>;
}
