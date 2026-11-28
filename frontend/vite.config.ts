import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Backend paths have no prefix (C5). The /api prefix exists only so the dev
// proxy can tell API calls from page loads; it is stripped before forwarding.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true, // CORS on the backend allows exactly this port
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
