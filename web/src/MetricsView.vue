<script setup>
import { computed, onUnmounted, ref, watch } from 'vue'
import MetricChart from './MetricChart.vue'
import Draggable from 'vuedraggable'
import { GripVertical, PinOff } from 'lucide-vue-next'
const props = defineProps({ run: String, setup: Object })
const emit = defineEmits(['update'])
const series = ref([]), error = ref(''), filter = ref(''), library = ref(null), selected = ref('')
let timer, generation = 0
const essentials = new Set(['Train/mean_reward', 'Train/mean_episode_length', 'Episode_Termination/success',
  'Episode_Termination/collision', 'Episode_Termination/timeout', 'Episode_Termination/stuck',
  'Metrics/path_completion', 'Validation/success_rate', 'Validation/collision_rate', 'Perf/total_fps',
  'Loss/value', 'Loss/surrogate', 'Policy/mean_std'])
const byTag = computed(() => new Map(series.value.map(s => [s.tag, s])))
const pins = computed({
  get: () => props.setup.plots ?? series.value.map(s => s.tag).filter(t => essentials.has(t)),
  set: plots => emit('update', { plots })
})
const matches = computed(() => series.value.filter(s => s.tag.toLowerCase().includes(filter.value.toLowerCase())))
const peek = computed(() => matches.value.find(s => s.tag === selected.value) || matches.value[0])
function toggle(tag) { pins.value = pins.value.includes(tag) ? pins.value.filter(t => t !== tag) : [...pins.value, tag] }
function move(index, offset) {
  const next = [...pins.value], to = index + offset
  if (to < 0 || to >= next.length) return
  next.splice(to, 0, next.splice(index, 1)[0]); pins.value = next
}
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
      <div><p class="eyebrow">Training metrics</p><h2 class="text-2xl font-semibold mt-1">Chart wall</h2>
        <p class="text-xs muted mt-2">Live TensorBoard event data · up to 1,500 samples per series · training outcomes are not held-out test results.</p></div>
      <div class="flex flex-wrap gap-3 items-center">
        <button class="btn btn-outline" @click="library.showModal()">Metric library · {{ series.length }}</button>
        <label class="text-xs">Smoothing {{ setup.smoothing }}<input :value="setup.smoothing" @change="emit('update',{smoothing:Number($event.target.value)})" type="range" min="0" max=".95" step=".05" class="range range-xs w-32 block mt-2" aria-label="Chart smoothing" /></label>
      </div>
    </div>
    <p v-if="error" class="alert alert-error">{{ error }}</p>
    <Draggable v-model="pins" :item-key="tag=>tag" handle=".plot-grip" :animation="160" class="grid md:grid-cols-2 gap-4">
      <template #item="{element:tag,index}"><div>
        <MetricChart v-if="byTag.has(tag)" :series="byTag.get(tag)" :smoothing="setup.smoothing">
          <template #handle><button class="plot-grip cursor-grab touch-none" :aria-label="`Reorder ${tag}, use up or down arrow`" @keydown.up.prevent="move(index,-1)" @keydown.down.prevent="move(index,1)"><GripVertical :size="18"/></button></template>
          <template #actions><button :aria-label="`Unpin ${tag}`" @click="toggle(tag)"><PinOff :size="15"/></button></template>
        </MetricChart>
        <div v-else class="panel p-4"><div class="flex gap-2 items-center"><button class="plot-grip cursor-grab touch-none" :aria-label="`Reorder ${tag}, use up or down arrow`" @keydown.up.prevent="move(index,-1)" @keydown.down.prevent="move(index,1)"><GripVertical :size="18"/></button><b>{{ tag }}</b></div><p class="muted text-sm">Waiting for this metric in the selected run.</p><button class="btn btn-xs" @click="toggle(tag)">Unpin</button></div>
      </div></template>
    </Draggable>
    <p v-if="!pins.length" class="panel p-8 muted">{{ !run ? 'Select a run to see its metrics.' : series.length ? 'Open the metric library to pin a plot.' : 'Waiting for the first training events.' }}</p>
    <dialog ref="library" class="modal" @click.self="library.close()"><div class="modal-box max-w-4xl">
      <header class="flex justify-between items-center"><h2 class="text-xl font-semibold">Metric library</h2><button class="btn btn-sm" @click="library.close()">Close</button></header>
      <input v-model="filter" class="input w-full my-3" placeholder="Search all metrics" aria-label="Search all metrics">
      <div class="grid md:grid-cols-[1fr_2fr] gap-3"><div class="max-h-64 overflow-auto"><button v-for="s in matches" :key="s.tag" class="block text-left p-2 text-sm break-all" :class="{'font-bold':peek===s}" @click="selected=s.tag">{{ s.tag }}</button></div>
        <div v-if="peek"><MetricChart :series="peek" :smoothing="setup.smoothing"/><button class="btn btn-sm mt-2" @click="toggle(peek.tag)">{{ pins.includes(peek.tag)?'Unpin':'Pin to wall' }}</button></div>
        <p v-else class="muted">No matching metrics in this run.</p></div>
    </div></dialog>
  </section>
</template>
