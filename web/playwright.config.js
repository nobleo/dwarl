import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:18765', headless: true },
  webServer: {
    command: 'pixi run python web/tests/dashboard_server.py',
    cwd: '..',
    url: 'http://127.0.0.1:18765',
    timeout: 30000,
    reuseExistingServer: false,
  },
})
