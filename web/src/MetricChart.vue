<script setup>
import { computed, ref } from 'vue'
const props = defineProps({ series: Object, smoothing: Number })
const group = computed(() => props.series.tag.split('/')[0])
const title = computed(() => (props.series.tag.split('/').slice(1).join(' / ') || group.value).replaceAll('_', ' '))
const colors = { Train:'#7466ca', Validation:'#138c80', Loss:'#c17635', Policy:'#a36e9e', Perf:'#40888c' }
const hover = ref(null)
const raw = computed(() => props.series.points)
const points = computed(() => {
  let mean = raw.value[0]?.[1] || 0
  return raw.value.map(([step, value]) => { mean = mean * props.smoothing + value * (1 - props.smoothing); return [step, mean] })
})
const limits = computed(() => {
  const ys = raw.value.map(p => p[1])
  const low = Math.min(...ys), high = Math.max(...ys), pad = Math.max((high - low) * .08, Math.abs(low) * .01, .0001)
  return { x0: raw.value[0][0], x1: raw.value.at(-1)[0], y0: low - pad, y1: high + pad }
})
const x = value => 58 + (value - limits.value.x0) / Math.max(1, limits.value.x1 - limits.value.x0) * 420
const y = value => 160 - (value - limits.value.y0) / (limits.value.y1 - limits.value.y0) * 140
const line = values => values.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(' ')
const format = n => Math.abs(n) >= 10000 || (n !== 0 && Math.abs(n) < .001) ? n.toExponential(2) : Number(n.toPrecision(4)).toString()
function move(event) {
  const box = event.currentTarget.getBoundingClientRect()
  const target = limits.value.x0 + ((event.clientX - box.left) / box.width * 500 - 58) / 420 * (limits.value.x1 - limits.value.x0)
  hover.value = raw.value.reduce((a, b) => Math.abs(a[0] - target) < Math.abs(b[0] - target) ? a : b)
}
</script>
<template>
  <section class="panel p-5 min-w-0" :style="{'--trace-color':colors[group] || 'var(--color-primary)'}">
    <div class="flex gap-2 items-center">
      <slot name="handle"></slot>
      <div class="flex-1 min-w-0" :title="series.tag"><p class="eyebrow break-all">{{ group.replaceAll('_',' ') }}</p>
        <h3 class="font-semibold break-words first-letter:uppercase">{{ title }}</h3></div>
      <slot name="actions"></slot>
    </div>
    <p class="font-mono text-2xl tracking-tight mt-3">{{ format((hover || raw.at(-1))[1]) }}</p>
    <svg viewBox="0 0 500 190" class="w-full mt-3 text-base-content" role="img" :aria-label="`${series.tag} over training steps`" @mousemove="move" @mouseleave="hover = null">
      <g v-for="i in [0, 1, 2]" :key="i">
        <line x1="58" x2="478" :y1="20 + i * 70" :y2="20 + i * 70" stroke="currentColor" opacity=".1" />
        <text x="50" :y="24 + i * 70" text-anchor="end" fill="currentColor" opacity=".6" font-size="10">{{ format(limits.y1 - (limits.y1 - limits.y0) * i / 2) }}</text>
      </g>
      <path :d="line(points)+' L478,160 H58 Z'" fill="var(--trace-color)" opacity=".06" />
      <path :d="line(raw)" stroke="var(--trace-color)" stroke-opacity=".24" stroke-width="1" fill="none" />
      <path :d="line(points)" stroke="var(--trace-color)" stroke-width="2" fill="none" />
      <circle v-if="hover" :cx="x(hover[0])" :cy="y(hover[1])" r="4" fill="var(--color-secondary)" />
      <text x="58" y="180" fill="currentColor" font-size="10">{{ limits.x0 }}</text>
      <text x="478" y="180" text-anchor="end" fill="currentColor" font-size="10">{{ limits.x1 }}</text>
    </svg>
    <p class="text-xs muted">Step {{ (hover || raw.at(-1))[0] }} · raw {{ format((hover || raw.at(-1))[1]) }} · line {{ smoothing ? 'smoothed' : 'raw' }}</p>
  </section>
</template>
