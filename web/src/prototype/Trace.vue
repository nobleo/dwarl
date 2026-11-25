<script setup>
import { computed, ref, watch } from 'vue'
import { format } from './signals'
const props = defineProps({ signal:Object, mini:Boolean })
const index = ref(null)
watch(() => props.signal, () => {index.value=null})
const range = computed(() => {
  const low=Math.min(...props.signal.raw), high=Math.max(...props.signal.raw), pad=(high-low)*.12
  return [low-pad,high+pad]
})
const y = v => 93-(v-range.value[0])/(range.value[1]-range.value[0])*90
const path = vs => vs.map((v,i)=>`${i?'L':'M'}${i/120*400},${y(v).toFixed(2)}`).join(' ')
function hover(e) {
  const box=e.currentTarget.getBoundingClientRect()
  index.value=Math.max(0,Math.min(120,Math.round((e.clientX-box.left)/box.width*120)))
}
</script>
<template>
  <div class="trace" :class="{mini}" :style="{color:signal.color}">
    <div v-if="!mini" class="axis-y"><span>{{ format(range[1]) }}</span><span>{{ format((range[1]+range[0])/2) }}</span><span>{{ format(range[0]) }}</span></div>
    <div class="graph" @pointermove="hover" @pointerleave="index=null">
      <svg viewBox="0 0 400 96" preserveAspectRatio="none" role="img" :aria-label="`${signal.name} by training step`">
        <path v-if="!mini" d="M0,3H400 M0,48H400 M0,93H400" stroke="#dfdfe4" stroke-dasharray="3 4" fill="none"/>
        <path :d="path(signal.line)+' L400,96 H0 Z'" fill="currentColor" opacity=".07"/>
        <path v-if="!mini" :d="path(signal.raw)" stroke="currentColor" opacity=".26" fill="none" vector-effect="non-scaling-stroke"/>
        <path :d="path(signal.line)" stroke="currentColor" stroke-width="2.3" fill="none" vector-effect="non-scaling-stroke"/>
        <path v-if="index!==null && !mini" :d="`M${index/120*400},0v96`" stroke="currentColor" stroke-dasharray="3 3"/>
      </svg>
      <output v-if="index!==null && !mini">Step {{ (index*100).toLocaleString() }} · {{ format(signal.raw[index]) }} {{ signal.unit }}</output>
    </div>
    <div v-if="!mini" class="axis-x"><span>0</span><span>6k</span><span>12k steps</span></div>
  </div>
</template>
<style scoped>
.trace{display:grid;grid-template-columns:auto 1fr;gap:4px 9px}.graph{position:relative;min-width:0}.graph svg{width:100%;height:var(--trace-height,108px);display:block}.axis-x,.axis-y{font:10px ui-monospace,monospace;color:#74717b;display:flex;justify-content:space-between}.axis-y{flex-direction:column;text-align:right}.axis-x{grid-column:2}.mini{display:block}.mini svg{height:34px}.graph output{position:absolute;top:0;right:0;padding:3px 7px;background:#282435;color:white;font:11px ui-monospace,monospace;border-radius:4px;pointer-events:none}
</style>
