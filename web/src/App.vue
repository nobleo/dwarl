<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { Activity, ArrowUpRight, BookOpen, ChartNoAxesCombined, CircleHelp, Moon, Play, ShieldCheck, Square, Sun, Terminal } from 'lucide-vue-next'
import logo from '../shared/assets/dwarl-logo.png'
import logoDark from '../shared/assets/dwarl-logo-dark.png'
import MujocoView from './MujocoView.vue'
import MetricsView from './MetricsView.vue'
import WorkspaceOverrides from './WorkspaceOverrides.vue'
import { dashboardSetup } from './dashboardSetup'
import DocumentView from './DocumentView.vue'
import CaseReplay from './CaseReplay.vue'
import Comparison from './Comparison.vue'

const fragment = new URLSearchParams(location.hash.slice(1))
if (fragment.has('token')) {
  sessionStorage.setItem('dwarl-token', fragment.get('token'))
  history.replaceState(null, '', location.pathname)
}
const token = sessionStorage.getItem('dwarl-token') || ''
const state = ref({ runs: [], context: {}, jobs: {}, live: {} })
const selected = ref(''), detail = ref(null), tab = ref('workspace')
const command = ref('train'), preset = ref(''), simulator = ref('mujoco')
const checkpoint = ref('latest'), robot = ref(''), resume = ref(false)
const extra = ref(''), busy = ref(false), error = ref(''), connected = ref(false)
const lane = ref('training'), evaluation = ref(''), reportOpen = ref(false)
const readiness = ref(null), previewSettings = ref(null), replay = ref(null), historyId = ref(''), historyJob = ref(null)
const savedTheme = localStorage.getItem('dwarl-theme')
const dark = ref(savedTheme ? savedTheme === 'slate' : matchMedia('(prefers-color-scheme: dark)').matches)
watch(dark, value => {
  const scheme = value ? 'slate' : 'default'
  document.documentElement.setAttribute('data-md-color-scheme', scheme)
  localStorage.setItem('dwarl-theme', scheme)
  // The bundled MkDocs site at /docs/ keeps its palette under this key; open it in the same scheme.
  localStorage.setItem('/docs/.__palette', JSON.stringify({
    index: value ? 1 : 0, color: { media: '', scheme, primary: 'custom', accent: 'custom' } }))
}, { immediate: true })
const guide = computed(() => `/docs/${['datagen', 'train', 'test', 'deploy'].includes(command.value) ? command.value : 'troubleshooting'}/`)
const terminal = ref(null), follow = ref(true)
let timer, stopped = false
const run = computed(() => state.value.runs.find(r => r.name === selected.value))
const job = computed(() => state.value.jobs[lane.value])
const running = computed(() => Boolean(job.value?.running))
const commandLane = computed(() => command.value === 'train' ? 'training' : 'evaluation')
const commandRunning = computed(() => Boolean(state.value.jobs[commandLane.value]?.running))
const compatible = computed(() => state.value.api_version === 5)
const { setup, message: setupMessage, failed: setupFailed, pending, load, save, update, importFile, download } = dashboardSetup(api)
watch(compatible, value => { if (value) load() })
const configurable = computed(() => ['datagen','train','test'].includes(command.value))
const usesRecipe = computed(() => command.value === 'datagen' || (command.value === 'train' && !resume.value))
const fields = ref(null)
watch([compatible, command, preset, resume], async (_, __, onCleanup) => {
  let current = true
  onCleanup(() => { current = false })
  fields.value = null
  if (!compatible.value || !configurable.value) return
  try {
    const query = new URLSearchParams({workflow:command.value,preset:usesRecipe.value?preset.value:''})
    const data = await api(`settings?${query}`)
    if (current) fields.value = data
  } catch(e) { if (current) error.value = e.message }
})
const starters = {datagen:['train_maps','validation_maps','test_maps','pairs_per_map'].map(key=>({path:`train.data.${key}`})),
  train:[{path:'train.max_iterations'}], test:[
  {path:'test.sim'}, {path:'test.suite',value:'validation'}, {path:'test.show'}, {path:'test.obstacle_comparison'}]}
const rows = computed(() => setup.value?.settings?.[command.value] ?? starters[command.value] ?? [])
const testSim = computed(() => rows.value.find(r=>r.path==='test.sim')?.value ?? fields.value?.['test.sim']?.value)
function updateRows(value) { update({settings:{...setup.value.settings,[command.value]:value}}) }
const waitingForSettings = computed(() => configurable.value && (!setup.value || !fields.value))
const progress = computed(() => run.value?.progress || {})
const argv = computed(() => {
  const args = [command.value]
  if (command.value === 'train' || command.value === 'datagen') {
    if (command.value === 'train' && resume.value && selected.value) args.push('--resume', `runs/${selected.value}`)
    else {
      if (robot.value.trim()) args.push(robot.value.trim())
      if (preset.value) args.push('--preset', preset.value)
    }
  }
  if (['test', 'deploy'].includes(command.value)) {
    if (selected.value) args.push('--run', `runs/${selected.value}`)
    args.push('--checkpoint', checkpoint.value)
    if (command.value === 'deploy' && robot.value.trim()) args.push(robot.value.trim())
  }
  for (const {path,value} of rows.value) {
    if (value === undefined || !fields.value?.[path]) continue
    if (path === 'test.obstacle_comparison' && testSim.value !== 'gazebo') continue
    if (value === true && ['test.show','test.obstacle_comparison'].includes(path)) args.push('--'+path.slice(5).replaceAll('_','-'))
    else args.push('--set', `${path}=${JSON.stringify(value)}`)
  }
  if (command.value === 'doctor') args.push('--sim', simulator.value)
  // Each advanced line is one argv element; never passed through a shell.
  return [...args, ...extra.value.split('\n').map(s => s.trim()).filter(Boolean)]
})
const preview = computed(() => `dwarl ${argv.value.map(a => /^[\w./=-]+$/.test(a) ? a : JSON.stringify(a)).join(' ')}`)
watch(argv, () => { previewSettings.value = null })
watch(robot, () => { readiness.value = null })
watch([selected, evaluation], () => { replay.value = null })
watch(selected, () => { checkpoint.value = 'latest' })
const selectedEvaluation = computed(() => detail.value?.evaluations.find(e => e.name === evaluation.value))
const episodes = computed(() => Object.entries(selectedEvaluation.value?.episodes || {})
  .flatMap(([backend, rows]) => rows.map(row => ({ ...row, backend }))))
const reportUrl = computed(() => `/reports/${encodeURIComponent(selected.value)}/tests/${encodeURIComponent(evaluation.value)}/report/index.html`)
function duration(seconds) {
  if (seconds == null) return '—'
  const s = Math.max(0, Math.round(seconds)), h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60)
  return h ? `${h} h ${String(m).padStart(2, '0')} min` : m ? `${m} min ${String(s % 60).padStart(2, '0')} s` : `${s} s`
}

async function api(path, body) {
  const response = await fetch(`/api/${path}`, {
    headers: { 'X-DWARL-Token': token, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    ...(body ? { method: 'POST', body: JSON.stringify(body) } : {}),
  })
  const data = await response.json()
  if (!response.ok) throw new Error(data.error || response.statusText)
  return data
}
async function loadDetail() {
  const name = selected.value
  const data = name ? await api(`run?name=${encodeURIComponent(name)}`) : null
  if (selected.value === name) {
    detail.value = data
    if (!data?.evaluations.some(e => e.name === evaluation.value)) {
      evaluation.value = data?.evaluations[0]?.name || ''
      reportOpen.value = false
    }
  }
}
async function refresh() {
  try {
    state.value = await api('state')
    connected.value = true
    if (!selected.value && state.value.runs.length) {
      const context = state.value.context
      const choices = context.runs?.[context.robot] || {}
      const remembered = (choices.navigation || choices.smoke || '').split('/').at(-1)
      selected.value = state.value.runs.some(r => r.name === remembered) ? remembered : state.value.runs[0].name
    }
    await loadDetail()
    if (follow.value) {
      await nextTick()
      terminal.value?.scrollTo(0, terminal.value.scrollHeight)
    }
  } catch (e) {
    connected.value = false
    error.value = e.message
  }
}
async function poll() {
  await refresh()
  if (!stopped) timer = setTimeout(poll, 1000)
}
async function act(path, body = {}) {
  busy.value = true
  error.value = ''
  try {
    if (path === 'command') lane.value = commandLane.value
    await api(path, body)
    await refresh()
  }
  catch (e) { error.value = e.message }
  finally { busy.value = false }
}
async function inspect(path, body, target) {
  busy.value = true
  error.value = ''
  try { target.value = await api(path, body) }
  catch (e) { error.value = e.message }
  finally { busy.value = false }
}
function caseRequest(row) {
  return { run: selected.value, evaluation: evaluation.value, case: row.case_id, backend: row.backend }
}
async function retryCase(row) {
  lane.value = 'evaluation'
  await act('retry', caseRequest(row))
}
const checkReadiness = () => inspect('readiness', { robot: robot.value }, readiness)
const checkPreview = () => inspect('preview', { argv: argv.value }, previewSettings)
const inspectCase = row => inspect('case', caseRequest(row), replay)
async function loadHistory() {
  try { historyJob.value = historyId.value ? await api(`history?id=${encodeURIComponent(historyId.value)}`) : null }
  catch (e) { error.value = e.message }
}
onMounted(poll)
onUnmounted(() => { stopped = true; clearTimeout(timer) })
</script>

<template>
  <header class="app-header sticky top-0 z-30 border-b border-base-300 bg-base-100/90 backdrop-blur">
    <div class="max-w-[1600px] mx-auto px-4 sm:px-8 py-3 flex flex-wrap items-center gap-x-6 gap-y-3">
      <div class="flex items-center gap-3">
        <img :src="dark ? logoDark : logo" alt="DWARL" class="w-28 h-auto" />
        <span class="hidden md:block h-7 border-l border-base-300"></span>
        <div class="hidden md:block leading-tight"><h1 class="text-sm font-semibold">Workspace</h1>
          <p class="text-xs muted">Train. Evaluate. Deploy.</p></div>
      </div>
      <nav class="segmented" role="tablist" aria-label="Workspace views">
        <button role="tab" :aria-selected="tab === 'workspace'" @click="tab = 'workspace'"><Activity :size="15" /> Workspace</button>
        <button role="tab" :aria-selected="tab === 'metrics'" @click="tab = 'metrics'"><ChartNoAxesCombined :size="15" /> Metrics</button>
      </nav>
      <a v-if="state.docs" href="/docs/" target="_blank" rel="noopener" class="btn btn-sm btn-ghost">
        <BookOpen :size="15" /> Documentation <ArrowUpRight :size="14" class="muted" /></a>
      <span v-else class="btn btn-sm btn-ghost btn-disabled" title="Build the documentation with pixi run web-build">
        <BookOpen :size="15" /> Documentation</span>
      <div class="flex flex-wrap items-center gap-3 lg:ml-auto text-xs">
        <details v-if="compatible" class="relative"><summary class="btn btn-sm">Dashboard setup</summary>
          <div class="panel absolute z-40 w-64 p-3 space-y-2 shadow-lg"><p>Workspace preferences · autosaved</p>
            <div class="flex flex-wrap gap-2"><button class="btn btn-xs" :disabled="pending || !setup" @click="save">Save setup</button>
              <button class="btn btn-xs" :disabled="pending" @click="load">Load setup</button>
              <button class="btn btn-xs" :disabled="!setup" @click="download">Export JSON</button></div>
            <label>Import setup<input type="file" accept=".json,application/json" class="file-input file-input-xs w-full" :disabled="pending" @change="importFile($event.target.files[0]); $event.target.value=''"></label>
            <p role="status">{{ setupMessage }}</p></div></details>
        <select v-model="selected" class="select select-sm w-64 max-w-full text-xs" aria-label="Selected run" @change="loadDetail().catch(e => error = e.message)">
          <option value="">No run selected</option><option v-for="r in state.runs" :key="r.name" :value="r.name">{{ r.name }}</option>
        </select>
        <span class="flex items-center gap-2 whitespace-nowrap"><span class="status" :class="connected ? 'status-success' : 'status-error'"></span>
          {{ connected ? 'Local workspace connected' : 'Connecting to CLI' }}</span>
        <button class="btn btn-sm btn-ghost btn-square" :aria-label="dark ? 'Switch to light mode' : 'Switch to dark mode'" @click="dark = !dark"><Sun v-if="dark" :size="17" /><Moon v-else :size="17" /></button>
      </div>
    </div>
  </header>

  <div class="max-w-[1600px] mx-auto px-4 sm:px-8 py-6">
    <div v-if="setupFailed" class="alert alert-error mb-5" role="alert">Dashboard setup: {{ setupMessage }}</div>
    <div v-if="error" class="alert alert-error mb-5" role="alert"><span>{{ error }}</span>
      <button class="btn btn-sm btn-ghost" @click="error = ''">Dismiss</button></div>
    <div v-if="connected && !compatible" class="alert alert-warning mb-5" role="alert">
      This page was rebuilt, but its server is still an older version. Open a new terminal and run
      <code>pixi run dwarl web --port 0</code>, then use the newly printed URL.
      It reconnects existing dashboard jobs without interrupting them. Keep the old terminal running until its jobs finish.
    </div>

    <main v-show="tab === 'workspace'" class="grid lg:grid-cols-[320px_1fr] gap-6 items-start">
      <aside class="panel lg:sticky lg:top-24 lg:max-h-[calc(100vh-7.5rem)] lg:overflow-y-auto">
        <div class="panel-header">
          <div><p class="eyebrow">Command centre</p><h2 class="text-lg font-semibold mt-1">What’s next?</h2></div>
          <a :href="guide" target="_blank" rel="noopener" class="btn btn-xs btn-ghost"><BookOpen :size="14" /> Help with {{ command }} <ArrowUpRight :size="12" /></a>
        </div>
        <div class="p-5 space-y-5">
          <label class="block"><span class="text-xs font-medium block mb-2">Workflow</span>
            <select v-model="command" class="select w-full" aria-label="Workflow">
              <option value="datagen">Prepare maps (datagen)</option>
              <option value="train">Train a policy</option><option value="test">Test a checkpoint</option>
              <option value="deploy">Deploy a policy</option><option value="doctor">Check environment</option>
              <option value="status">Inspect status</option><option value="next">Suggest next step</option>
            </select></label>
          <label v-if="['datagen', 'train', 'deploy'].includes(command)" class="block">
            <span class="text-xs font-medium block mb-2">Robot integration</span>
            <input v-model="robot" class="input w-full" placeholder="Use saved selection" aria-label="Robot integration" />
            <p class="text-xs muted mt-2 break-words">{{ state.context.robot || 'No saved robot. Run pixi run setup to get started.' }}</p>
          </label>
          <template v-if="command === 'train' || command === 'datagen'">
            <label v-if="command === 'train' && selected" class="flex items-center gap-2 text-sm"><input v-model="resume" type="checkbox" class="checkbox checkbox-sm" /> Resume selected run</label>
            <label v-if="usesRecipe" class="block"><span class="text-xs font-medium block mb-2">Recipe</span>
              <select v-model="preset" class="select w-full" aria-label="Recipe"><option value="">Configured defaults</option>
                <option value="smoke">Smoke</option><option value="navigation">Navigation</option></select></label>
          </template>
          <label v-if="['test', 'deploy'].includes(command)" class="block"><span class="text-xs font-medium block mb-2">Checkpoint</span>
            <select v-model="checkpoint" class="select w-full" aria-label="Checkpoint">
              <option value="latest">Latest completed · pin on start</option>
              <option value="best">Best validation · pin on start</option>
              <option v-for="cp in detail?.checkpoints || []" :key="cp.sha256" :value="String(cp.iteration)">
                Iteration {{ cp.iteration }}{{ cp.best ? ' · best' : '' }} · {{ cp.sha256.slice(0, 8) }}
              </option>
            </select></label>
          <label v-if="command==='doctor'" class="block"><span class="text-xs font-medium block mb-2">Simulator</span>
            <select v-model="simulator" class="select w-full" aria-label="Simulator"><option value="mujoco">MuJoCo</option><option value="gazebo">Gazebo</option></select></label>
          <WorkspaceOverrides v-if="configurable && fields && setup" :key="command" :fields="fields" :rows="rows" :gazebo="testSim==='gazebo'" @update="updateRows" />
          <details class="text-sm">
            <summary class="cursor-pointer muted">Advanced arguments</summary>
            <p class="text-xs muted my-2">One argument per line, including option values. Example: --set on one line, train.device=cpu on the next.</p>
            <textarea v-model="extra" class="textarea w-full font-mono text-xs" rows="4" aria-label="Advanced arguments"></textarea>
          </details>
          <div class="bg-base-200 rounded-lg p-3 command-preview">{{ preview }}</div>
          <button v-if="['datagen', 'train', 'test', 'deploy'].includes(command)" class="btn btn-outline w-full" :disabled="busy || !connected || waitingForSettings" @click="checkPreview">{{ command === 'datagen' ? 'Preview dataset settings' : 'Preview & check inputs' }}</button>
          <div v-if="previewSettings" class="text-xs bg-base-200 p-3 rounded-lg space-y-2">
            <p class="font-semibold text-primary">{{ command === 'datagen' ? 'Dataset settings resolved' : 'Preflight passed' }}</p>
            <p class="break-words">Robot: {{ previewSettings.config.robot || (command === 'datagen' ? 'No robot selected' : 'Frozen run profile') }}</p>
            <p v-if="command === 'train'">Device: {{ previewSettings.config.train.device }} · {{ previewSettings.config.train.num_envs }} environments · {{ previewSettings.config.train.max_iterations }} iterations</p>
            <p v-if="command === 'test'">{{ previewSettings.config.test.sim }} · {{ previewSettings.config.test.suite }} · up to {{ previewSettings.config.test.max_cases || 'all' }} cases per controller · checkpoint {{ previewSettings.config.test.checkpoint }}</p>
            <details><summary class="cursor-pointer">Resolved workflow settings</summary><pre class="terminal console rounded-lg p-3 mt-2 max-h-64 overflow-auto">{{ JSON.stringify(command === 'datagen' ? previewSettings.config.train.data : previewSettings.config[command], null, 2) }}</pre></details>
          </div>
          <div class="grid grid-cols-[1fr_auto] gap-2">
            <button class="btn btn-primary" :disabled="busy || commandRunning || !connected || waitingForSettings" @click="act('command', { argv: previewSettings?.argv || argv, telemetry: false })">
              <Play :size="16" /> Run {{ command }}</button>
            <button class="btn btn-outline" :disabled="busy || !commandRunning" :title="`Stop ${commandLane}`" @click="act('stop', { lane: commandLane })"><Square :size="14" /> Stop</button>
          </div>
          <p class="text-xs muted flex gap-2"><CircleHelp :size="16" class="shrink-0" />Stop requests a graceful interrupt. Completed checkpoints and results are retained.</p>
        </div>
        <div class="p-5 border-t border-base-300 space-y-3">
          <p class="text-sm font-medium flex items-center gap-2"><ShieldCheck :size="16" class="muted" /> Robot readiness</p>
          <div v-if="readiness" class="text-xs space-y-2">
            <p class="font-semibold" :class="readiness.passed ? 'text-success' : 'text-error'">{{ readiness.passed ? 'Robot contract checks passed' : 'Robot needs attention' }}</p>
            <p v-for="(finding, i) in readiness.findings || []" :key="i" :class="finding.level === 'fail' ? 'text-error' : 'muted'">
              {{ finding.message }} <span v-if="finding.hint" class="block mt-1">{{ finding.hint }}</span>
            </p>
            <p v-if="readiness.error" class="text-error">{{ readiness.error }}</p>
          </div>
          <p v-else class="text-xs muted">Checks the selected robot integration against the training contract.</p>
          <button class="btn btn-sm btn-outline w-full" :disabled="busy" @click="checkReadiness">Check robot readiness</button>
        </div>
      </aside>

      <div class="space-y-6 min-w-0">
        <section class="panel">
          <div class="panel-header">
            <div><p class="eyebrow">Run evidence</p><h2 class="text-lg font-semibold mt-1">Your policy, at a glance</h2>
              <p class="text-xs muted mt-1 font-mono break-all">{{ selected || 'No run selected' }}</p></div>
            <button class="btn btn-sm btn-ghost" @click="tab = 'metrics'"><ChartNoAxesCombined :size="15" /> View metrics</button>
          </div>
          <div class="p-5">
            <div class="grid grid-cols-2 xl:grid-cols-4 gap-5">
              <div><p class="stat-label">Run state</p><p class="stat-value capitalize">{{ run?.state.state || 'Ready to begin' }}</p></div>
              <div><p class="stat-label">Iterations</p><p class="stat-value">{{ progress.iteration != null ? `${progress.iteration + 1} / ${progress.target}` : '—' }}</p></div>
              <div><p class="stat-label">Checkpoints recorded</p><p class="stat-value">{{ run?.checkpoints ?? '—' }}</p></div>
              <div><p class="stat-label">Elapsed / ETA · wall time</p><p class="stat-value">{{ progress.elapsed_s != null ? `${duration(progress.elapsed_s)} / ${duration(progress.eta_s || 0)}` : '—' }}</p></div>
            </div>
            <progress v-if="progress.target" class="progress progress-primary w-full mt-5" :value="progress.iteration + 1" :max="progress.target" aria-label="Training progress"></progress>
            <p v-if="run?.state.error" class="text-error text-sm mt-3">{{ run.state.error }}</p>
          </div>
        </section>

        <div class="flex flex-wrap gap-3 items-center">
          <span class="eyebrow">Live view</span>
          <div class="segmented" role="tablist" aria-label="Live view">
            <button v-for="name in ['training', 'evaluation']" :key="name" role="tab" :aria-selected="lane === name" @click="lane = name">
              <span v-if="state.jobs[name]?.running" class="status status-success"></span>
              {{ name === 'training' ? 'Training' : 'Evaluation' }}
            </button>
          </div>
          <span v-if="state.jobs.training?.running && state.jobs.evaluation?.running" class="text-xs muted">Concurrent work affects compute timings.</span>
        </div>
        <MujocoView v-if="compatible" :lane="lane" :running="running" :supported="job?.viewer_protocol === 3"
          :testing="job?.argv?.[0] === 'test'" :status="state.render?.[lane]" :live="state.live?.[lane]" :active="tab === 'workspace'" />

        <section class="panel overflow-hidden">
          <div class="panel-header">
            <h2 class="font-semibold flex items-center gap-2"><Terminal :size="17" /> Command output <span class="text-xs font-normal muted">· {{ lane }}</span></h2>
            <div class="flex gap-3 items-center text-xs">
              <span v-if="job" class="badge badge-sm badge-soft" :class="running ? 'badge-primary' : job.returncode === 0 ? 'badge-success' : 'badge-warning'">{{ running ? 'Running' : `Exit ${job.returncode}` }}</span>
              <label class="flex items-center gap-2"><input v-model="follow" type="checkbox" class="checkbox checkbox-xs" /> Follow</label>
            </div>
          </div>
          <div v-if="job" class="px-5 py-2 bg-base-200 border-b border-base-300 command-preview">dwarl {{ job.argv.join(' ') }}</div>
          <pre ref="terminal" class="terminal console h-80 overflow-auto p-5">{{ job?.output || (lane === 'training' ? detail?.output : '') || 'Ready. Choose a workflow and run a command to see its output here.' }}</pre>
        </section>

        <section class="panel overflow-hidden">
          <div class="panel-header">
            <div><p class="eyebrow">Evaluation</p><h2 class="font-semibold mt-1">Results & reports</h2></div>
            <select v-model="evaluation" class="select select-sm w-auto max-w-full" aria-label="Evaluation results" @change="reportOpen = false">
              <option value="">No evaluation selected</option>
              <option v-for="e in detail?.evaluations || []" :key="e.name" :value="e.name">
                {{ e.name }} · iteration {{ e.summary?.iteration ?? 'in progress' }} · {{ e.summary?.suite || 'running' }}
              </option>
            </select>
          </div>
          <div v-if="selectedEvaluation?.summary" class="p-5 border-b border-base-300 space-y-3">
            <div class="flex flex-wrap justify-between items-center gap-3">
              <div class="text-sm"><span class="badge badge-soft mr-2" :class="selectedEvaluation.summary.passed ? 'badge-success' : 'badge-warning'">{{ selectedEvaluation.summary.status }} · {{ selectedEvaluation.summary.passed ? 'passed' : 'not passed' }}</span>
                <span class="muted">Checkpoint {{ selectedEvaluation.summary.checkpoint?.slice(0, 12) }}</span></div>
              <div v-if="selectedEvaluation.report" class="flex gap-2">
                <button class="btn btn-sm btn-outline" @click="reportOpen = !reportOpen">{{ reportOpen ? 'Hide report' : 'View full report & plots' }}</button>
                <a :href="reportUrl" target="_blank" rel="noopener" class="btn btn-sm btn-ghost" aria-label="Open report in a new tab"><ArrowUpRight :size="15" /></a>
              </div>
            </div>
            <p v-if="selectedEvaluation.summary.error" class="text-error text-sm">{{ selectedEvaluation.summary.error }}</p>
            <div v-for="(stats, backend) in selectedEvaluation.summary.simulators" :key="backend" class="text-sm flex flex-wrap gap-4">
              <strong>{{ backend }}</strong><span>{{ stats.completed }}/{{ stats.expected }} completed</span>
              <span>Success {{ stats.success_rate == null ? '—' : `${(stats.success_rate * 100).toFixed(1)}%` }}</span>
              <span>Collision {{ stats.collision_rate == null ? '—' : `${(stats.collision_rate * 100).toFixed(1)}%` }}</span>
            </div>
          </div>
          <DocumentView v-if="reportOpen && selectedEvaluation?.report" :src="reportUrl" />
          <div v-if="episodes.length" class="overflow-auto max-h-80">
            <table class="table table-sm table-pin-rows"><thead><tr><th>Case</th><th>Backend</th><th>Outcome</th><th>Time · sim s</th><th>Clearance · m</th><th>Inspect</th></tr></thead>
              <tbody><tr v-for="(row, i) in episodes" :key="i" class="hover:bg-base-200">
                <td>{{ row.case_name }}</td><td>{{ row.backend }}</td><td><span class="badge badge-sm badge-soft" :class="row.outcome === 'success' ? 'badge-success' : 'badge-warning'">{{ row.outcome }}</span></td>
                <td>{{ row.completion_time?.toFixed(1) ?? '—' }}</td><td>{{ row.clearance?.toFixed(2) ?? '—' }}</td>
                <td class="whitespace-nowrap"><button class="btn btn-xs btn-ghost" :disabled="busy" @click="inspectCase(row)">Replay</button>
                  <button class="btn btn-xs btn-ghost" :disabled="busy || state.jobs.evaluation?.running || !selectedEvaluation?.summary" @click="retryCase(row)">Rerun case</button></td>
              </tr></tbody></table>
          </div>
          <p v-else class="p-5 text-sm muted">Completed cases appear here as testing progresses. Process completion alone does not mean successful navigation.</p>
          <CaseReplay :data="replay" />
        </section>
        <Comparison :evaluations="detail?.evaluations || []" />
        <section class="panel overflow-hidden">
          <div class="p-5 flex flex-wrap items-center justify-between gap-3">
            <div><p class="eyebrow">Recent commands</p><h2 class="font-semibold mt-1">Persistent history</h2></div>
            <select v-model="historyId" class="select select-sm w-auto max-w-full" aria-label="Command history" @change="loadHistory">
              <option value="">Choose a command</option>
              <option v-for="item in state.history || []" :key="item.id" :value="item.id">
                {{ new Date(item.started_at * 1000).toLocaleString() }} · {{ item.argv[0] }} · {{ item.returncode == null ? 'no recorded exit' : `exit ${item.returncode}` }}
              </option>
            </select>
          </div>
          <template v-if="historyJob">
            <p class="px-5 py-2 bg-base-200 border-y border-base-300 command-preview">dwarl {{ historyJob.argv.join(' ') }}</p>
            <pre class="terminal console p-5 max-h-72 overflow-auto">{{ historyJob.output || 'No output recorded.' }}</pre>
          </template>
        </section>
      </div>
    </main>

    <MetricsView v-if="tab === 'metrics' && compatible && setup" :run="selected" :setup="setup" @update="update" />
    <footer class="py-6 text-xs muted flex justify-between">Local tools. Recorded evidence. <span>DWARL</span></footer>
  </div>
</template>
