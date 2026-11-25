import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  build: { outDir: '../dwarl/src/dwarl/web/static', emptyOutDir: true },
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8765', changeOrigin: true,
        configure(proxy) {
          proxy.on('proxyReq', (request) => request.removeHeader('origin'))
        },
      },
      '/reports': { target: 'http://127.0.0.1:8765', changeOrigin: true },
      '/frames': { target: 'http://127.0.0.1:8765', changeOrigin: true },
      '/docs': { target: 'http://127.0.0.1:8765', changeOrigin: true },
    },
  },
})
