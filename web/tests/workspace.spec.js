import { test, expect } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.goto('/#token=dwarl-browser-test')
  await expect(page.getByText('Local workspace connected', { exact: true })).toBeVisible()
})

test('run picker follows the newest run until a run is chosen manually', async ({ page }) => {
  const names = ['new', 'old']
  await page.route('**/api/state', route => route.fulfill({ json: {
    runs: names.map(name => ({name, state:{state:'completed'}})), jobs: {}, live: {},
    context: {robot:'diffbot', runs:{diffbot:{navigation:'runs/old'}}},
  } }))
  await page.route('**/api/run?*', route => route.fulfill({json:{evaluations:[], checkpoints:[]}}))
  await page.reload()
  const picker = page.getByLabel('Selected run', {exact:true})
  await expect(picker).toHaveValue('new')
  names.unshift('newer')
  await expect(picker).toHaveValue('newer')
  await picker.selectOption('old')
  names.unshift('newest')
  await expect(picker.locator('option[value="newest"]')).toHaveCount(1)
  await expect(picker).toHaveValue('old')
})

test('prepare maps previews and runs datagen without starting training', async ({ page }) => {
  await page.getByLabel('Workflow', {exact:true}).selectOption('datagen')
  await page.getByLabel('Robot integration').fill('diffbot')
  await page.getByLabel('Recipe', {exact:true}).selectOption('smoke')
  await expect(page.getByLabel('train maps', {exact:true})).toHaveValue('1')
  await page.getByRole('button', {name:'Preview dataset settings'}).click()
  await expect(page.getByText('Dataset settings resolved')).toBeVisible()
  const response = page.waitForResponse(r => r.url().endsWith('/api/command'))
  await page.getByRole('button', {name:'Run datagen'}).click()
  const job = await (await response).json()
  expect(job.argv).toEqual(['datagen','diffbot','--preset','smoke'])
  await expect(page.locator('pre').filter({hasText:'Dataset: generating'})).toBeVisible({timeout:20000})
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

test('a shared setup restores the wall and typed overrides without starting a command', async ({ page }) => {
  const setup = { schema:'dwarl-dashboard/1', plots:[], smoothing:0,
    settings:{train:[{path:'train.early_stop.min_delta',value:0.25}]} }
  const headers = {'X-DWARL-Token':'dwarl-browser-test'}
  const original = await (await page.request.get('/api/setup', {headers})).json()
  try {
    await page.getByText('Dashboard setup',{exact:true}).click()
    await page.getByLabel('Import setup').setInputFiles({name:'shared.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(setup))})
    await expect(page.getByText('Imported and saved workspace setup.')).toBeVisible()
    await page.reload()
    await expect(page.getByLabel('min delta',{exact:true})).toHaveValue('0.25')
    await expect(page.locator('aside .command-preview')).toContainText('train.early_stop.min_delta=0.25')
    await page.getByRole('button',{name:'Reset to inherited'}).click()
    await expect(page.locator('aside .command-preview')).not.toContainText('min_delta')
    await page.getByRole('tab',{name:'Metrics',exact:true}).click()
    await expect(page.getByText('Open the metric library to pin a plot.')).toBeVisible()
    await page.getByRole('button',{name:/Metric library/}).click()
    await expect(page.getByRole('dialog').getByRole('img')).toBeVisible()
    await page.getByRole('button',{name:'Pin to wall'}).click()
    await expect.poll(async () => (await (await page.request.get('/api/setup',{headers})).json()).plots)
      .toEqual(['Train/mean_reward'])
  } finally {
    await page.request.post('/api/setup',{headers,data:original})
  }
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
