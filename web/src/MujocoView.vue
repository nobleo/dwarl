<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
const props = defineProps({ lane: String, running: Boolean, supported: Boolean, status: Object, active: Boolean })
const enabled = ref(false), visible = ref(false), host = ref(null), error = ref('')
const mode = ref('environment'), environment = ref(0), pageVisible = ref(!document.hidden)
const rendering = computed(() => enabled.value && visible.value && props.active && props.running && props.supported && pageVisible.value)
const frame = computed(() => rendering.value && props.status?.backend === 'studio' && props.status?.port
  ? `/studio/${props.lane}/?session=${props.status.session}` : '')
let timer, observer
async function update(lane = props.lane, forceOff = false) {
  try {
    const response = await fetch('/api/view', { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-DWARL-Token': sessionStorage.getItem('dwarl-token') || '' },
      body: JSON.stringify({ lane, enabled: !forceOff && rendering.value,
        camera: { mode: mode.value, environment: environment.value } }),
    })
    if (!response.ok) throw new Error((await response.json()).error)
    error.value = ''
  } catch (e) { error.value = e.message }
}
watch(() => props.lane, (_, old) => { update(old, true); enabled.value = false })
watch([rendering, mode, environment], () => update())
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
    <div class="flex flex-wrap items-center justify-between gap-4 p-5 border-b border-base-300">
      <div><p class="eyebrow">MuJoCo Studio</p><h2 class="font-semibold mt-1">Live simulator</h2></div>
      <label class="flex items-center gap-3 text-sm"><span>{{ rendering ? 'Browser graphics · live state up to 20 Hz' : enabled ? 'Suspended while hidden / idle' : 'Viewer off' }}</span>
        <input v-model="enabled" type="checkbox" class="toggle toggle-primary" :disabled="!running || !supported" aria-label="Enable MuJoCo rendering" /></label>
    </div>
    <div class="relative min-h-72 bg-base-200">
      <iframe v-if="frame" :src="frame" class="w-full h-[720px] border-0" title="MuJoCo Studio live simulation"
        allow="fullscreen; cross-origin-isolated" allowfullscreen />
      <div v-else class="h-72 flex flex-col items-center justify-center p-8 text-center gap-3">
        <p class="font-semibold">{{ running && !supported ? 'This job started with an older viewer' : running ? (enabled ? 'Starting MuJoCo Studio…' : 'Open the interactive MuJoCo Studio viewer') : 'Start a dashboard training or MuJoCo test command' }}</p>
        <p v-if="running && !supported" class="muted text-sm max-w-lg">Training continues normally. Metrics and results remain available. The live rendering hook is available for new or resumed jobs started from this dashboard.</p>
        <p v-else class="muted text-sm max-w-lg">Graphics run in your browser; the host needs no GPU or display. Viewing is off by default and stops when this view scrolls away, the tab is hidden, or the browser closes.</p>
      </div>
    </div>
    <p v-if="status?.error || error" class="p-4 text-error text-sm">{{ status?.error || error }}</p>
    <div v-if="enabled" class="p-4 border-t border-base-300 flex flex-wrap gap-4 text-xs items-center">
      <select v-model="mode" class="select select-sm" aria-label="Camera view"><option value="environment">Single environment</option><option value="overview">World overview</option></select>
      <label v-if="mode === 'environment'">Environment <input v-model.number="environment" type="number" min="0" :max="(status?.environments || 1) - 1" class="input input-sm w-20" /></label>
      <button class="btn btn-sm" :disabled="!frame" @click="host.querySelector('iframe')?.requestFullscreen()">Fullscreen</button>
      <span class="muted">Use Studio’s camera, visualization and inspector controls. DWARL owns physics and policy execution; simulation edits in Studio do not change the run.</span>
      <span v-if="mode === 'overview' && status?.shard_environments < status?.environments" class="muted">
        Overview shows the selected environment’s shard ({{ status.shard_environments }} environments).
      </span>
    </div>
  </section>
</template>
