import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/earth-system-trend-detective/",
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        lab: "lab/index.html",
      },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
