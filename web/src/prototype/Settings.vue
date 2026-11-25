<script setup>
import { computed, ref } from 'vue'
// Snapshot of compose_config() with public defaults and no workspace overrides.
import defaults from './public-defaults.json'
function leaves(obj, prefix = '') {
  return Object.entries(obj).flatMap(([key, value]) => {
    const path = prefix + key
    return value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length
      ? leaves(value, path + '.') : [[path, value]]
  })
}
defineProps({ variant: String })
const fields = Object.fromEntries(leaves(defaults)), keys = Object.keys(fields)
const basics = ['robot','train.device','train.num_envs','train.max_iterations']
const edits = ref({}), query = ref(''), selected = ref('train.device'), adding = ref(false)
const matches = computed(() => keys.filter(k => k.includes(query.value.trim().toLowerCase())))
const section = k => k.startsWith('recipe.') ? k.split('.').slice(0,2).join('.') : k.split('.')[0]
const sections = [...new Set(keys.map(section))]
const open = ref('')
const display = v => typeof v === 'string' ? v : JSON.stringify(v)
const current = k => edits.value[k] ?? display(fields[k])
function reset(k) { delete edits.value[k] }
</script>
<template>
  <section class="space-y-4">
    <div class="panel panel-header"><div><p class="eyebrow">Command centre / configuration</p>
      <h1 class="text-2xl font-semibold">Workspace settings</h1>
      <p class="muted text-sm">{{ keys.length }} public defaults · draft only · no job will run</p></div>
      <span class="badge">{{ Object.keys(edits).length }} overrides</span></div>
    <div class="panel p-5 space-y-4">
      <template v-if="variant==='A'">
        <b>Essentials</b>
        <div class="grid md:grid-cols-2 gap-3"><label v-for="k in basics" :key="k" class="text-sm">{{ k }}
          <input class="input block w-full" :value="current(k)" @input="edits[k]=$event.target.value"></label></div>
        <details><summary class="cursor-pointer">Advanced · browse by section</summary>
          <select v-model="open" class="select my-3" aria-label="Settings section"><option value="">Choose a section</option><option v-for="s in sections" :key="s">{{ s }}</option></select>
          <div v-for="k in keys.filter(k=>section(k)===open)" :key="k"><label class="grid md:grid-cols-2 items-center gap-2 mb-2 text-sm">{{ k }}
            <input class="input w-full" :value="current(k)" @input="edits[k]=$event.target.value"></label></div>
        </details>
      </template>
      <template v-else-if="variant==='B'">
        <b>Find a setting</b>
        <div class="grid md:grid-cols-2 gap-5">
          <div><input v-model="query" placeholder="Search e.g. learning_rate" aria-label="Find a setting" class="input w-full">
            <p class="muted text-xs my-2">{{ matches.length }} matches</p>
            <select v-if="matches.length" v-model="selected" size="7" class="w-full font-mono text-xs" aria-label="Matching settings"><option v-for="k in matches" :key="k">{{ k }}</option></select></div>
          <div class="panel p-4"><p class="eyebrow">Inspector</p><label class="block my-2 break-all">{{ selected }}
            <input class="input w-full mt-2" :value="current(selected)" @input="edits[selected]=$event.target.value"></label>
            <small class="muted">Default: {{ display(fields[selected]) }}</small></div>
        </div>
      </template>
      <template v-else>
        <b>Override sheet</b><p class="muted text-sm">Start with the defaults. Add only what this experiment needs.</p>
        <button class="btn btn-outline" @click="adding=!adding">{{ adding ? 'Close picker' : '+ Add a setting' }}</button>
        <div v-if="adding" class="space-y-2"><input v-model="query" aria-label="Filter settings" class="input" placeholder="Filter by path">
          <select class="select max-w-full" aria-label="Add setting" value="" @change="edits[$event.target.value]=current($event.target.value); adding=false">
            <option disabled value="">Choose a setting</option><option v-for="k in matches" :key="k">{{ k }}</option></select></div>
        <label v-for="(_,k) in edits" :key="k" class="grid md:grid-cols-2 items-center gap-2 text-sm">{{ k }}
          <input class="input w-full" :value="current(k)" @input="edits[k]=$event.target.value"></label>
      </template>
    </div>
    <div class="panel p-5"><p class="eyebrow">Draft overrides</p>
      <p v-if="!Object.keys(edits).length" class="muted mt-2 text-sm">Using all configured defaults.</p>
      <div v-for="(value,k) in edits" :key="k" class="flex gap-3 items-center justify-between py-1">
        <code class="text-xs break-all">--set {{ k }}={{ value }}</code><button class="btn btn-xs" @click="reset(k)">Reset</button></div>
    </div>
  </section>
</template>
