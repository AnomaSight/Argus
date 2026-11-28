import { useEffect, useState } from "react";

import { getHealth } from "./api/client";

export default function App() {
  const [status, setStatus] = useState("checking…");

  useEffect(() => {
    getHealth()
      .then((h) => setStatus(h.status))
      .catch(() => setStatus("unreachable"));
  }, []);

  return (
    <main>
      <h1>Argus</h1>
      <p>Backend: {status}</p>
    </main>
  );
}
