import { test, expect } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.goto('/#token=dwarl-browser-test')
  await expect(page.getByText('Local workspace connected', { exact: true })).toBeVisible()
})

test('visual obstacle comparison uses the normal test command', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Workflow', exact: true }).selectOption('test')
  await page.getByRole('combobox', { name: 'Simulator', exact: true }).selectOption('gazebo')
  await page.getByRole('checkbox', { name: 'Show simulator windows', exact: true }).check()
  await page.getByRole('checkbox', { name: 'Compare clean and obstructed routes', exact: true }).check()
  const preview = page.getByText(/^dwarl test /)
  await expect(preview).toContainText('--obstacle-comparison')
  await expect(preview).toContainText('--show')
  await page.getByRole('combobox', { name: 'Simulator', exact: true }).selectOption('mujoco')
  await expect(preview).not.toContainText('--obstacle-comparison')
  await expect(page.getByRole('checkbox', { name: 'Compare clean and obstructed routes', exact: true })).toHaveCount(0)
})

test('reports and metrics stay in the workspace while documentation opens as its own site', async ({ page }) => {
  await page.getByRole('button', { name: 'View full report & plots', exact: true }).click()
  const image = page.getByRole('img', { name: 'Route plot', exact: true })
  await image.scrollIntoViewIfNeeded()
  await expect.poll(() => image.evaluate(img => img.complete && img.naturalWidth > 0)).toBe(true)
  await expect(page.locator('iframe')).toHaveCount(0)
  await page.getByRole('tab', { name: 'Metrics', exact: true }).click()
  await expect(page.getByRole('img', { name: 'Train/mean_reward over training steps', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Switch to dark mode', exact: true }).click()
  const opened = page.waitForEvent('popup')
  await page.getByRole('link', { name: 'Documentation' }).click()
  const docs = await opened
  await expect(docs).toHaveURL(/\/docs\/$/)
  await expect(docs.getByRole('heading', { name: 'Get started', exact: true })).toBeVisible()
  await expect(docs.locator('body')).toHaveAttribute('data-md-color-scheme', 'slate')
  await docs.getByRole('link', { name: 'CLI', exact: true }).first().click()
  await expect(docs.getByRole('heading', { name: 'CLI reference', exact: true })).toBeVisible()
  await expect(page.locator('iframe')).toHaveCount(0)
})

test('real-time playback is offered for tests and sent with the viewer lease', async ({ page }) => {
  const toggle = page.getByRole('checkbox', { name: 'Play tests in real time', exact: true })
  await expect(toggle).toHaveCount(0)
  await page.getByRole('tab', { name: 'Evaluation', exact: true }).click()
  const lease = page.waitForRequest(request => request.url().endsWith('/api/view')
    && request.postDataJSON().lane === 'evaluation' && request.postDataJSON().realtime === true)
  await toggle.check()
  await lease
})

test('an old backend gets an upgrade message instead of unsupported API requests', async ({ page }) => {
  await page.route('**/api/state', route => route.fulfill({ json: { runs: [], jobs: {}, context: {}, live: {} } }))
  let unsupported = 0
  page.on('request', request => {
    if (/\/api\/(metrics|view)(\?|$)/.test(request.url())) unsupported++
  })
  await page.reload()
  await expect(page.getByRole('alert')).toContainText('server is still an older version')
  await page.getByRole('tab', { name: 'Metrics', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('without interrupting')
  expect(unsupported).toBe(0)
})

test('Studio runs in the browser and shuts down when off or hidden', async ({ page }) => {
  test.setTimeout(90000)
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const streams = new Set()
  let states = 0
  page.on('websocket', socket => socket.on('framereceived', () => {
    const path = new URL(socket.url()).pathname
    streams.add(path)
    if (path.endsWith('/state')) states++
  }))
  const toggle = page.getByRole('checkbox', { name: 'Enable MuJoCo rendering', exact: true })
  await toggle.scrollIntoViewIfNeeded()
  await expect(toggle).not.toBeChecked()
  await toggle.check()
  const viewer = page.locator('iframe[title="MuJoCo Studio live simulation"]')
  await expect(viewer).toBeVisible({ timeout: 20000 })
  const canvas = page.frameLocator('iframe[title="MuJoCo Studio live simulation"]').locator('canvas')
  await expect(canvas).toBeVisible({ timeout: 30000 })
  await expect.poll(() => [...streams].sort(), { timeout: 45000 }).toEqual([
    '/studio/training/state', '/studio/training/ui',
  ])
  await expect.poll(() => canvas.evaluate(() => typeof currentModelCrc !== 'undefined' && currentModelCrc > 0), {
    timeout: 30000,
  }).toBe(true)
  await expect.poll(() => states).toBeGreaterThan(10)
  expect(errors).toEqual([])
  await canvas.screenshot({ path: 'test-results/studio-live.png' })
  const firstSession = await viewer.getAttribute('src')
  await toggle.uncheck()
  await expect(viewer).toHaveCount(0)
  const state = () => page.evaluate(async () => (await (await fetch('/api/state', {
    headers: { 'X-DWARL-Token': sessionStorage.getItem('dwarl-token') },
  })).json()).render.training.updated_at)
  // Grace period exceeds the lease lifetime; then no new frames should be generated.
  await page.waitForTimeout(3500)
  const stopped = await state()
  await page.waitForTimeout(600)
  expect(await state()).toBe(stopped)
  await toggle.check()
  await expect.poll(state).toBeGreaterThan(stopped)
  await expect(viewer).toBeVisible({ timeout: 20000 })
  await expect.poll(() => viewer.getAttribute('src')).not.toBe(firstSession)
  await expect.poll(() => canvas.evaluate(() => typeof currentModelCrc !== 'undefined' && currentModelCrc > 0), {
    timeout: 30000,
  }).toBe(true)
  await page.getByRole('tab', { name: 'Metrics', exact: true }).click()
  await page.waitForTimeout(3500)
  const hidden = await state()
  await page.waitForTimeout(600)
  expect(await state()).toBe(hidden)
})
