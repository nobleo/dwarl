<script setup>
import { computed, ref, watch } from 'vue'
import MetricChart from '../MetricChart.vue'
const props = defineProps({ variant: String })
const tags = ['Train/mean_reward','Validation/success_rate','Loss/value','Policy/mean_std','Metrics/path_completion','Perf/total_fps','Episode_Termination/collision','Loss/surrogate']
const series = Object.fromEntries(tags.map((tag, i) => [tag, { tag, points: Array.from({length: 70}, (_, t) =>
  [t * 100, (i + 1) * (1 + Math.sin(t / 14 + i) * .3 + Math.cos(t * (i + 1)) * .04 + t / 100)]) }]))
const storage = 'dwarl-prototype-pins-v1'
const pins = ref(JSON.parse(localStorage.getItem(storage) || JSON.stringify(tags.slice(0,3))))
watch(pins, value => localStorage.setItem(storage, JSON.stringify(value)), {deep:true})
const library = ref(false), filter = ref(''), peek = ref(''), dragged = ref('')
const matches = computed(() => tags.filter(t => t.toLowerCase().includes(filter.value.toLowerCase())))
const showLibrary = computed(() => props.variant === 'B' || library.value)
function toggle(t) {
  if (pins.value.includes(t)) pins.value = pins.value.filter(p => p !== t)
  else pins.value.push(t)
}
function move(from, to) {
  const i = pins.value.indexOf(from)
  if (i >= 0 && to >= 0 && to < pins.value.length) pins.value.splice(to, 0, pins.value.splice(i,1)[0])
}
</script>
<template>
  <div class="panel panel-header"><div><p class="eyebrow">Training metrics</p><h1 class="text-2xl font-semibold">My plots</h1>
    <p class="muted text-sm">Drag a grip to reorder. Your {{ pins.length }} pinned plots are saved in this browser.</p></div>
    <button v-if="variant !== 'B'" class="btn btn-outline" @click="library = !library">{{ library ? 'Close library' : 'Browse all plots' }}</button></div>
  <div :class="variant === 'B' ? 'grid md:grid-cols-[270px_1fr] gap-4 items-start' : 'space-y-4'">
    <aside v-if="showLibrary" class="panel p-4 space-y-2"><b>Metric library</b>
      <input v-model="filter" aria-label="Search plots" placeholder="Search all metrics" class="input w-full">
      <div v-for="t in matches" :key="t" class="flex justify-between items-center text-xs gap-2">
        <button class="link text-left break-all" @click="peek = t">{{ t }}</button>
        <button class="btn btn-xs" @click="toggle(t)">{{ pins.includes(t) ? 'Unpin' : 'Pin' }}</button></div>
      <p v-if="!matches.length">No matching plots.</p></aside>
    <div class="space-y-4">
      <div v-if="peek" class="panel p-3"><div class="flex justify-between items-center"><b>Quick look · layout unchanged</b><button class="btn btn-xs" @click="peek = ''">Close</button></div><MetricChart :series="series[peek]" :smoothing=".6" /></div>
      <div :class="variant === 'C' ? 'grid md:grid-cols-[minmax(260px,1fr)_2fr] items-start gap-4' : ''">
        <div :class="variant === 'A' ? 'grid md:grid-cols-2 xl:grid-cols-3 gap-3' : 'space-y-3'">
          <article v-for="(t,i) in pins" :key="t" class="panel p-2" @dragover.prevent @drop.prevent="move(dragged,i); dragged=''">
            <div class="flex gap-2 items-center"><button class="btn btn-xs cursor-grab" draggable="true" :aria-label="`Drag ${t}`" @dragstart="dragged=t; $event.dataTransfer.setData('text/plain',t)" @dragend="dragged=''">⠿</button>
              <button v-if="variant === 'C'" class="link text-xs text-left flex-1" @click="peek=t">{{ t }}</button>
              <button class="btn btn-xs" :disabled="!i" :aria-label="`Move ${t} earlier`" @click="move(t,i-1)">↑</button>
              <button class="btn btn-xs" :disabled="i === pins.length-1" :aria-label="`Move ${t} later`" @click="move(t,i+1)">↓</button>
              <button class="btn btn-xs ml-auto" @click="toggle(t)">Unpin</button></div>
            <MetricChart v-if="variant !== 'C'" :series="series[t]" :smoothing=".6" /></article>
        </div>
        <MetricChart v-if="variant === 'C' && pins.length" :series="series[pins[0]]" :smoothing=".6" />
      </div>
      <p v-if="!pins.length" class="panel p-5">Your board is empty. Pin a metric from the library to start.</p>
    </div>
  </div>
</template>
