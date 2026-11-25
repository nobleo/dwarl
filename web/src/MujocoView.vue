<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { Box, LoaderCircle, Maximize, Timer } from 'lucide-vue-next'
const props = defineProps({
  lane: String, running: Boolean, supported: Boolean, testing: Boolean, status: Object, live: Object, active: Boolean,
})
const enabled = ref(false), visible = ref(false), host = ref(null), viewport = ref(null), error = ref('')
const mode = ref('environment'), environment = ref(0), pageVisible = ref(!document.hidden)
const realtime = ref(localStorage.getItem('dwarl-realtime') === 'true')
watch(realtime, value => localStorage.setItem('dwarl-realtime', String(value)))
const rendering = computed(() => enabled.value && visible.value && props.active && props.running && props.supported && pageVisible.value)
const frame = computed(() => rendering.value && props.status?.backend === 'studio' && props.status?.port
  ? `/studio/${props.lane}/?session=${props.status.session}` : '')
const pacing = computed(() => props.lane === 'evaluation' && realtime.value)
const caseLabel = computed(() => props.testing && props.live?.kind === 'testing' && props.live.label
  ? `${props.live.label}${props.live.time_s != null ? ` · ${props.live.time_s.toFixed(1)} s` : ''}` : '')
const state = computed(() => rendering.value ? 'Live · up to 20 Hz' : enabled.value ? 'Paused while hidden' : 'Off')
let timer, observer
async function update(lane = props.lane, forceOff = false) {
  try {
    const response = await fetch('/api/view', { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-DWARL-Token': sessionStorage.getItem('dwarl-token') || '' },
      body: JSON.stringify({ lane, enabled: !forceOff && rendering.value, realtime: lane === 'evaluation' && realtime.value,
        camera: { mode: mode.value, environment: environment.value } }),
    })
    if (!response.ok) throw new Error((await response.json()).error)
    error.value = ''
  } catch (e) { error.value = e.message }
}
watch(() => props.lane, (_, old) => { update(old, true); enabled.value = false })
watch([rendering, mode, environment, realtime], () => update())
const visibility = () => { pageVisible.value = !document.hidden }
onMounted(() => {
  observer = new IntersectionObserver(entries => { visible.value = entries[0].isIntersecting })
  observer.observe(host.value)
  document.addEventListener('visibilitychange', visibility)
  timer = setInterval(() => update(props.lane, document.hidden), 1000)
})
onUnmounted(() => { clearInterval(timer); observer?.disconnect(); document.removeEventListener('visibilitychange', visibility); update(props.lane, true) })
</script>
<template>
  <section ref="host" class="panel overflow-hidden">
    <div class="panel-header">
      <div><p class="eyebrow">MuJoCo Studio</p><h2 class="font-semibold mt-1">Live simulator</h2></div>
      <div class="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <label v-if="lane === 'evaluation'" class="flex items-center gap-2 cursor-pointer"
          title="Slow MuJoCo testing to wall-clock speed while you watch. Scores are unchanged; full speed resumes when the viewer closes.">
          <Timer :size="16" class="muted" /> Real time
          <input v-model="realtime" type="checkbox" class="toggle toggle-sm toggle-secondary" aria-label="Play tests in real time" />
        </label>
        <label class="flex items-center gap-2 cursor-pointer">
          <span class="text-xs muted">{{ state }}</span> Viewer
          <input v-model="enabled" type="checkbox" class="toggle toggle-primary" :disabled="!running || !supported" aria-label="Enable MuJoCo rendering" />
        </label>
        <button v-if="frame" class="btn btn-sm btn-ghost btn-square -ml-3" aria-label="Fullscreen" title="Fullscreen"
          @click="viewport.requestFullscreen()"><Maximize :size="16" /></button>
      </div>
    </div>
    <div ref="viewport" class="studio-viewport" :class="enabled ? 'h-[clamp(420px,62vh,720px)]' : 'h-72'">
      <iframe v-if="frame" :src="frame" title="MuJoCo Studio live simulation" allow="fullscreen; cross-origin-isolated" allowfullscreen />
      <div v-else class="absolute inset-0 flex flex-col items-center justify-center p-8 text-center gap-3">
        <LoaderCircle v-if="rendering" :size="30" class="animate-spin opacity-60" />
        <Box v-else :size="30" class="opacity-50" />
        <p class="font-semibold">{{ running && !supported ? 'This job started with an older viewer'
          : rendering ? 'Waiting for the MuJoCo simulation…' : running ? 'Watch this run in MuJoCo Studio' : 'No live simulation' }}</p>
        <p v-if="running && !supported" class="text-sm opacity-70 max-w-lg">Training continues normally. Metrics and results remain available. The live rendering hook is available for new or resumed jobs started from this dashboard.</p>
        <p v-else-if="rendering" class="text-sm opacity-70 max-w-lg">Studio opens while the run steps its MuJoCo world. Gazebo tests and report generation have no MuJoCo view.</p>
        <p v-else-if="running" class="text-sm opacity-70 max-w-lg">Graphics run in your browser; the host needs no GPU or display. Viewing pauses when this panel scrolls away or the tab is hidden.</p>
        <p v-else class="text-sm opacity-70 max-w-lg">Start a training run or a MuJoCo test from the command centre to watch it here.</p>
        <button v-if="running && supported && !enabled" class="btn btn-primary btn-sm mt-2" @click="enabled = true">Open viewer</button>
      </div>
      <div v-if="frame" class="pointer-events-none absolute top-3 left-3 right-3 flex flex-wrap gap-2">
        <span class="overlay-chip"><span class="status status-success"></span> Live</span>
        <span v-if="caseLabel" class="overlay-chip">{{ caseLabel }}</span>
        <span v-if="pacing && testing" class="overlay-chip"><Timer :size="13" /> Real time</span>
      </div>
    </div>
    <p v-if="status?.error || error" class="px-5 py-3 text-error text-sm border-t border-base-300">{{ status?.error || error }}</p>
    <div v-if="enabled" class="px-5 py-3 border-t border-base-300 flex flex-wrap gap-x-5 gap-y-3 text-xs items-center">
      <label class="flex items-center gap-2">Camera
        <select v-model="mode" class="select select-sm w-auto" aria-label="Camera view"><option value="environment">Single environment</option><option value="overview">World overview</option></select></label>
      <label v-if="mode === 'environment'" class="flex items-center gap-2">Environment
        <input v-model.number="environment" type="number" min="0" :max="(status?.environments || 1) - 1" class="input input-sm w-20" /></label>
      <span v-if="mode === 'overview' && status?.shard_environments < status?.environments" class="muted">
        Overview shows the selected environment’s shard ({{ status.shard_environments }} environments).
      </span>
      <span class="muted lg:ml-auto">Drag to orbit · scroll to zoom · display only: DWARL owns physics and policy execution.</span>
    </div>
  </section>
</template>
