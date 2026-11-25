<script setup>
import { computed, nextTick, ref } from 'vue'
import Draggable from 'vuedraggable'
import { GripVertical, X } from 'lucide-vue-next'

const props = defineProps({ rows: Array, fields: Object, gazebo: Boolean })
const emit = defineEmits(['update'])
const rows = computed({get: () => props.rows, set: value => emit('update', value)})
const search = ref(''), open = ref(false), finder = ref(null), error = ref('')
const labels = {'test.sim':'Simulator','test.suite':'Suite','test.show':'Show simulator windows',
  'test.obstacle_comparison':'Compare clean and obstructed routes','train.max_iterations':'Iteration budget'}
const name = path => labels[path] || path.split('.').at(-1).replaceAll('_',' ')
const normalize = text => text.toLowerCase().replace(/[._]/g,' ')
const matches = computed(() => Object.keys(props.fields).filter(path =>
  !rows.value.some(r => r.path === path) && normalize(path+' '+name(path)).includes(normalize(search.value.trim()))))
const value = row => Object.hasOwn(row,'value') ? row.value : props.fields[row.path]?.value
function change(path, patch) {
  error.value = ''
  rows.value = rows.value.map(r => r.path === path ? {path,...patch} : r)
}
function number(row, el) {
  if (!el.value || !el.checkValidity()) {
    error.value = `Enter a valid ${props.fields[row.path].kind} for ${row.path}.`
    el.value = value(row) ?? ''; return
  }
  change(row.path,{value:el.valueAsNumber})
}
function json(row, el) {
  try { change(row.path,{value:JSON.parse(el.value)}) }
  catch { error.value = `Invalid JSON for ${row.path}; previous value retained.`; el.value = JSON.stringify(value(row)) }
}
function move(i, n) {
  if (i+n < 0 || i+n >= rows.value.length) return
  const next = [...rows.value]; next.splice(i+n,0,next.splice(i,1)[0]); rows.value = next
}
async function find() { open.value=!open.value; search.value=''; await nextTick(); finder.value?.focus() }
</script>
<template>
  <section class="space-y-2"><h3 class="font-semibold text-sm">Settings overrides</h3>
    <p class="muted text-xs">Pin a field to keep it here. Only explicit values are passed to the CLI.</p>
    <Draggable v-model="rows" item-key="path" handle=".field-grip" :animation="160" class="space-y-2">
      <template #item="{element:r,index}"><div class="bg-base-200 p-3 rounded-lg space-y-1">
        <div class="flex gap-1 items-center"><button class="field-grip touch-none cursor-grab" :aria-label="`Reorder ${r.path}, use up or down arrow`" @keydown.up.prevent="move(index,-1)" @keydown.down.prevent="move(index,1)"><GripVertical :size="15"/></button>
          <label class="text-sm font-medium flex-1" :for="'field-'+r.path">{{ name(r.path) }}</label>
          <button :aria-label="`Remove ${r.path}`" @click="rows=rows.filter(item=>item.path!==r.path)"><X :size="14"/></button></div>
        <code class="block text-xs muted break-all">{{ r.path }}</code>
        <template v-if="fields[r.path]">
          <p v-if="r.path==='test.obstacle_comparison' && !gazebo" class="text-xs muted">Available with Gazebo.</p>
          <select v-else-if="fields[r.path].choices" :id="'field-'+r.path" class="select select-sm w-full" :value="value(r)" @change="change(r.path,{value:$event.target.value})"><option v-for="c in fields[r.path].choices" :key="c">{{ c }}</option></select>
          <input v-else-if="fields[r.path].kind==='bool'" :id="'field-'+r.path" class="checkbox checkbox-sm" type="checkbox" :checked="value(r)" @change="change(r.path,{value:$event.target.checked})">
          <template v-else-if="['int','float'].includes(fields[r.path].kind)">
            <input v-if="fields[r.path].minimum!=null && fields[r.path].maximum!=null" class="range range-xs" type="range" :aria-label="`${name(r.path)} slider`" :value="value(r)" :min="fields[r.path].minimum" :max="fields[r.path].maximum" step=".001" @change="number(r,$event.target)">
            <input :id="'field-'+r.path" class="input input-sm w-full" type="number" :step="fields[r.path].kind==='int'?1:'any'" :value="value(r)" :min="fields[r.path].minimum" :max="fields[r.path].maximum" @change="number(r,$event.target)">
          </template>
          <textarea v-else-if="fields[r.path].kind==='json'" :id="'field-'+r.path" class="textarea w-full text-xs font-mono" :value="JSON.stringify(value(r))" @change="json(r,$event.target)"></textarea>
          <input v-else :id="'field-'+r.path" class="input input-sm w-full" :value="value(r)??''" @change="change(r.path,{value:$event.target.value})">
          <div class="flex flex-wrap gap-2 items-center text-xs"><button v-if="Object.hasOwn(r,'value')" class="link" @click="change(r.path,{})">Reset to inherited</button>
            <span v-else class="muted">Inherited</span><button v-if="fields[r.path].nullable" class="link" @click="change(r.path,{value:null})">Set null</button></div>
        </template><p v-else class="text-xs muted">Field unavailable in this configuration.</p>
      </div></template>
    </Draggable>
    <button class="btn btn-sm btn-outline w-full" @click="find">{{ open?'Close picker':'+ Add setting' }}</button>
    <div v-if="open"><input ref="finder" v-model="search" class="input input-sm w-full" placeholder="Search all settings" aria-label="Search all settings">
      <div class="max-h-48 overflow-auto"><button v-for="path in matches" :key="path" class="block break-all p-2 text-left text-xs hover:bg-base-200" @click="rows=[...rows,{path}]; open=false">{{ path }}</button><p v-if="!matches.length" class="text-sm muted p-2">No matching settings.</p></div></div>
    <p v-if="error" role="alert" class="text-error text-xs">{{ error }}</p>
  </section>
</template>
