import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// HeatSense AI frontend. Proxies /api to the FastAPI backend during
// development so the browser doesn't need CORS-exempt requests.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    // Split the large vendor bundle into smaller chunks so the initial load
    // is faster. Leaflet + react-leaflet and Recharts are the two big deps.
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor-react": ["react", "react-dom"],
          "vendor-leaflet": ["leaflet", "react-leaflet"],
          "vendor-recharts": ["recharts"],
          "vendor-axios": ["axios"],
        },
      },
    },
    chunkSizeWarningLimit: 600,
  },
});
