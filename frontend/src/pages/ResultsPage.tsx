import { useEffect, useState } from "react";

import { getResults, USE_MOCK } from "../api/client";
import type { ResultTable } from "../api/types";
import ResultsTable from "../components/ResultsTable";

export default function ResultsPage() {
  const [tables, setTables] = useState<ResultTable[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getResults()
      .then(setTables)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main className="page">
      <header className="page-head">
        <h1>Results</h1>
        <p>False-positive rate, recall and AUROC for each memory setting, at the calibrated threshold.</p>
        {USE_MOCK && (
          <p className="notice">Sample tables with made-up numbers. No experiment has run yet.</p>
        )}
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      {tables?.length === 0 && <p className="empty">No result tables yet.</p>}
      {tables?.map((t) => <ResultsTable key={t.name} table={t} />)}
    </main>
  );
}
