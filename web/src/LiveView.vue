<script setup>
import { computed } from 'vue'
const props = defineProps({ live: Object, running: Boolean })
const points = computed(() => {
  const path = props.live?.path || []
  return [...path, ...(props.live?.trail || []), ...(props.live?.pose ? [props.live.pose] : [])]
    .filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]))
})
const bounds = computed(() => {
  if (!points.value.length) return [-5, -5, 10, 10]
  const xs = points.value.map(p => p[0]), ys = points.value.map(p => p[1])
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys)
  const width = Math.max(4, xmax - xmin + 2), height = Math.max(4, ymax - ymin + 2)
  return [(xmin + xmax - width) / 2, -(ymin + ymax + height) / 2, width, height]
})
const path = computed(() => (props.live?.path || []).map(p => `${p[0]},${-p[1]}`).join(' '))
const trail = computed(() => (props.live?.trail || []).map(p => `${p[0]},${-p[1]}`).join(' '))
const scale = computed(() => Math.max(bounds.value[2], bounds.value[3]) / 35)
const pose = computed(() => props.live?.pose)
</script>

<template>
  <section class="panel overflow-hidden">
    <div class="flex items-center justify-between gap-3 p-5 border-b border-base-300">
      <div><p class="eyebrow">{{ live?.kind === 'replay' ? 'Recorded trace' : 'Live telemetry' }}</p><h2 class="font-semibold mt-1">{{ live?.label || 'Robot view' }}</h2></div>
      <span class="badge badge-soft" :class="running && live ? 'badge-success' : 'badge-neutral'">
        {{ live ? (running ? 'Sampled · 4 Hz max' : 'Last sample') : 'Waiting for a command' }}
      </span>
    </div>
    <div class="relative bg-base-200">
      <svg v-if="pose" :viewBox="bounds.join(' ')" class="w-full h-80 p-6" role="img" aria-label="Live robot pose and reference path in metres">
        <polyline :points="path" fill="none" stroke="var(--color-primary)" stroke-opacity=".4" :stroke-width="scale * .22" />
        <polyline v-if="trail" :points="trail" fill="none" stroke="var(--color-secondary)" :stroke-width="scale * .16" />
        <circle v-if="live.path?.length" :cx="live.path.at(-1)[0]" :cy="-live.path.at(-1)[1]" :r="scale * .65" fill="var(--color-secondary)" />
        <g :transform="`translate(${pose[0]},${-pose[1]}) rotate(${-pose[2] * 180 / Math.PI})`">
          <circle :r="scale" fill="var(--color-primary)" fill-opacity=".16" />
          <path :d="`M ${scale},0 L ${-scale * .6},${scale * .6} L ${-scale * .35},0 L ${-scale * .6},${-scale * .6} Z`" fill="var(--color-primary)" />
        </g>
      </svg>
      <div v-else class="h-80 flex flex-col items-center justify-center text-center p-8">
        <div class="w-16 h-16 rounded-full border border-dashed border-primary mb-4 grid place-items-center text-primary text-2xl">↗</div>
        <p class="font-medium">See the policy move</p>
        <p class="muted text-sm max-w-xs mt-2">Start training or testing with the live view enabled. Pose and reference path appear here.</p>
      </div>
    </div>
    <div class="p-4 border-t border-base-300 flex flex-wrap gap-x-6 gap-y-1 text-xs muted">
      <span>Reference path · metres · top-down</span>
      <span v-if="live">Simulation {{ Number(live.time_s).toFixed(1) }} s</span>
      <span v-if="live?.clearance_m != null">Clearance {{ live.clearance_m.toFixed(2) }} m</span>
      <span v-if="live?.updated_at">Received {{ new Date(live.updated_at * 1000).toLocaleTimeString() }}</span>
    </div>
  </section>
</template>
