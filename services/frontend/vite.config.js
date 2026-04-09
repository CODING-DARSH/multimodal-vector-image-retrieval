import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // In dev mode, proxy /search/* and /images/* to the API
      // This avoids CORS issues when running outside Docker
      "/search": "http://localhost:8000",
      "/feedback": "http://localhost:8000",
      "/images": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/stats": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
  },
})