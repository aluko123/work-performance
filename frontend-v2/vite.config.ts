import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/analyses': { target: 'http://localhost:8000', changeOrigin: true },
      '/analyze_text': { target: 'http://localhost:8000', changeOrigin: true },
      '/analysis_status': { target: 'http://localhost:8000', changeOrigin: true },
    }
  }
})
