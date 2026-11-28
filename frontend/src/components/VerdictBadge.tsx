import type { Verdict } from "../api/types";

export default function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return (
    <p className={`verdict ${verdict}`} role="status">
      {verdict}
    </p>
  );
}
