<script setup>
// Throwaway design review: compare three ways to curate a dashboard via ?prototype&variant=A/B/C.
import { computed, onMounted, onUnmounted, ref } from 'vue'
import Plots from './Plots.vue'
import Settings from './Settings.vue'
import logo from '../../shared/assets/dwarl-logo.png'

const params = new URLSearchParams(location.search)
const letters = ['A', 'B', 'C']
const variant = ref(letters.includes(params.get('variant')) ? params.get('variant') : 'A')
const page = ref(params.get('page') === 'settings' ? 'settings' : 'plots')
const names = computed(() => page.value === 'plots' ? ['Pinboard', 'Library rail', 'Focus queue'] : ['Essentials + sections', 'Setting finder', 'Override sheet'])
function navigate(next) {
  page.value = next
  params.set('page', next)
  history.replaceState(null, '', '?' + params)
}
function cycle(step) {
  variant.value = letters[(letters.indexOf(variant.value) + step + 3) % 3]
  params.set('variant', variant.value)
  history.replaceState(null, '', '?' + params)
}
function key(e) {
  if (e.target.closest('input,textarea,select,[contenteditable]')) return
  if (['ArrowLeft','ArrowRight'].includes(e.key)) cycle(e.key === 'ArrowLeft' ? -1 : 1)
}
onMounted(() => addEventListener('keydown', key))
onUnmounted(() => removeEventListener('keydown', key))
</script>
<template>
  <header class="app-header panel-header bg-base-100"><div class="flex items-center gap-4"><img :src="logo" alt="DWARL" width="90"><b>Workspace</b></div>
    <nav class="segmented" aria-label="Prototype features"><button :aria-selected="page==='plots'" @click="navigate('plots')">Metrics</button>
      <button :aria-selected="page==='settings'" @click="navigate('settings')">Workspace settings</button></nav>
    <span class="badge">Design preview · sample run</span></header>
  <main class="max-w-6xl mx-auto p-5 pb-28 space-y-5">
    <section v-show="page==='plots'" class="space-y-4"><Plots :variant="variant" /></section>
    <Settings v-show="page==='settings'" :variant="variant" />
  </main>
  <nav aria-label="Compare designs" class="fixed bottom-4 left-1/2 -translate-x-1/2 bg-neutral text-neutral-content rounded-full shadow-xl flex gap-4 items-center px-4 py-2 whitespace-nowrap">
    <button class="btn btn-circle btn-xs" aria-label="Previous design" @click="cycle(-1)">←</button>
    {{ variant }} · {{ names[letters.indexOf(variant)] }}
    <button class="btn btn-circle btn-xs" aria-label="Next design" @click="cycle(1)">→</button>
  </nav>
</template>
