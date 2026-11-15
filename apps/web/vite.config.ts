import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins: [react()],
  server: { port: Number(process.env.WEB_PORT ?? 5174), strictPort: true, proxy: { '/api': 'http://127.0.0.1:8787', '/data': 'http://127.0.0.1:8787' } },
  build: { outDir: 'dist', sourcemap: true },
});
