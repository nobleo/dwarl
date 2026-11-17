<script setup>
import { computed, onUnmounted, ref, watch } from 'vue'
import MetricChart from './MetricChart.vue'
const props = defineProps({ run: String })
const series = ref([]), error = ref(''), filter = ref(''), group = ref('essentials'), smoothing = ref(.6)
let timer, generation = 0
const essentials = new Set(['Train/mean_reward', 'Train/mean_episode_length', 'Episode_Termination/success',
  'Episode_Termination/collision', 'Episode_Termination/timeout', 'Episode_Termination/stuck',
  'Metrics/path_completion', 'Validation/success_rate', 'Validation/collision_rate', 'Perf/total_fps',
  'Loss/value', 'Loss/surrogate', 'Policy/mean_std'])
const groups = computed(() => [...new Set(series.value.map(s => s.tag.split('/')[0]))])
const visible = computed(() => series.value.filter(s => {
  const match = group.value === 'all' || (group.value === 'essentials' ? essentials.has(s.tag) : s.tag.startsWith(group.value + '/'))
  return match && s.tag.toLowerCase().includes(filter.value.toLowerCase())
}))
async function poll(id) {
  if (!props.run || id !== generation) return
  try {
    const response = await fetch(`/api/metrics?run=${encodeURIComponent(props.run)}`, { headers: { 'X-DWARL-Token': sessionStorage.getItem('dwarl-token') || '' } })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error)
    if (id === generation) { series.value = data.series; error.value = '' }
  } catch (e) { if (id === generation) error.value = e.message }
  if (id === generation) timer = setTimeout(() => poll(id), 4000)
}
watch(() => props.run, () => { clearTimeout(timer); series.value = []; poll(++generation) }, { immediate: true })
onUnmounted(() => { generation++; clearTimeout(timer) })
</script>
<template>
  <section class="space-y-5">
    <div class="panel p-5 flex flex-wrap items-center justify-between gap-4">
      <div><p class="eyebrow">Training metrics</p><h2 class="font-semibold mt-1">Learning & navigation</h2>
        <p class="text-xs muted mt-2">Live TensorBoard event data · up to 1,500 samples per series · training outcomes are not held-out test results.</p></div>
      <div class="flex flex-wrap gap-3 items-center">
        <select v-model="group" class="select select-sm" aria-label="Metric group"><option value="essentials">Essential plots</option><option value="all">All metrics</option><option v-for="g in groups" :key="g">{{ g }}</option></select>
        <input v-model="filter" class="input input-sm" placeholder="Filter metrics" aria-label="Filter metrics" />
        <label class="text-xs">Smoothing {{ smoothing }}<input v-model.number="smoothing" type="range" min="0" max=".95" step=".05" class="range range-xs w-32 block mt-2" aria-label="Chart smoothing" /></label>
      </div>
    </div>
    <p v-if="error" class="alert alert-error">{{ error }}</p>
    <div class="grid md:grid-cols-2 xl:grid-cols-3 gap-4"><MetricChart v-for="s in visible" :key="s.tag" :series="s" :smoothing="smoothing" /></div>
    <p v-if="!visible.length" class="panel p-8 muted">{{ run ? 'No matching scalar data yet. Plots appear when training writes its first events.' : 'Select a run to see its metrics.' }}</p>
  </section>
</template>
