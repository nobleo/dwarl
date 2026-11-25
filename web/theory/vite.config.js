import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'

const here = path => fileURLToPath(new URL(path, import.meta.url))
export default defineConfig({
  plugins: [vue()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  build: {
    lib: { entry: here('./main.js'), name: 'DwarlTheory', formats: ['iife'], fileName: () => 'theory-explorer.js' },
    outDir: here('../docs/assets'), emptyOutDir: false,
  },
})
