<script setup>
import {computed, nextTick, ref, watch} from 'vue'
const preset=ref('navigation'), fields=ref({}), edits=ref({}), added=ref([])
const search=ref(''), picker=ref(false), searchBox=ref(null), result=ref(null), error=ref(''), busy=ref(false)
const basics=['train.num_envs','train.max_iterations','train.device']
const visible=computed(()=>[...basics,...added.value].filter(k=>fields.value[k]))
const results=computed(()=>Object.keys(fields.value).filter(k=>!visible.value.includes(k) &&
  k.replace(/[._]/g,' ').includes(search.value.trim().toLowerCase().replace(/[._]/g,' '))))
const label=k=>k.split('.').at(-1).replaceAll('_',' ')
const show=v=>typeof v==='string'?v:JSON.stringify(v)
const inherited=k=>fields.value[k].value
const value=k=>Object.hasOwn(edits.value,k)?edits.value[k]:inherited(k)
const text=k=>Object.hasOwn(edits.value,k)?edits.value[k]:JSON.stringify(inherited(k))
function add(k){added.value.push(k);picker.value=false}
function reset(k){delete edits.value[k]}
async function openPicker(){picker.value=!picker.value; search.value='';await nextTick();searchBox.value?.focus()}
async function post(payload) {
  const r=await fetch('/prototype-train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
  const data=await r.json()
  if(data.error)throw Error(data.error)
  return data
}
let version=0
watch(preset,async p=>{
  const id=++version;fields.value={};result.value=null;error.value='';busy.value=false
  try {const data=await post({preset:p});if(id===version)fields.value=data.fields}
  catch(e){if(id===version)error.value=e.message}
},{immediate:true})
watch(edits,()=>{version++;busy.value=false;result.value=null;error.value=''}, {deep:true,flush:'sync'})
async function preview(){
  const id=++version;error.value='';result.value=null;busy.value=true
  try{
    const overrides={}
    for(const [k,v] of Object.entries(edits.value)){
      if(fields.value[k].kind==='json' && v!==null){
        try{overrides[k]=JSON.parse(v)}catch{throw Error(`Enter valid JSON for ${k}.`)}
      } else overrides[k]=v
    }
    const data=await post({preset:preset.value,edits:overrides})
    if(id===version)result.value=data.resolved
  }catch(e){if(id===version)error.value=e.message}
  finally{if(id===version)busy.value=false}
}
</script>
<template>
  <section class="max-w-3xl mx-auto space-y-4">
    <div class="panel panel-header"><div><p class="eyebrow">Workspace / Train</p>
      <h1 class="text-2xl font-semibold">Configure a training run</h1>
      <p class="muted text-sm mt-1">Start from a recipe. Override only what this run needs.</p></div>
      <label class="text-xs">Inherit from recipe<select v-model="preset" class="select block mt-1" aria-label="Recipe"><option>navigation</option><option>smoke</option></select></label></div>
    <form class="panel p-5 space-y-4" @submit.prevent="preview">
      <div v-for="k in visible" :key="k" class="border-b border-base-300 pb-4">
        <div class="flex justify-between items-center gap-3"><label :for="k" class="font-semibold capitalize">{{ label(k) }}</label>
          <button v-if="Object.hasOwn(edits,k)" type="button" class="btn btn-xs" @click="reset(k)">Reset to inherited</button>
          <span v-else class="text-xs muted">Inherited</span></div>
        <code class="block text-xs muted mb-2">{{ k }} · {{ fields[k].kind }} · recipe: {{ show(inherited(k)) }}</code>
        <input v-if="fields[k].kind==='bool'" :id="k" type="checkbox" class="checkbox" :checked="value(k)" @change="edits[k]=$event.target.checked">
        <template v-else-if="['int','float'].includes(fields[k].kind)">
          <input v-if="fields[k].maximum!=null && fields[k].minimum!=null" class="range range-sm max-w-xs align-middle mr-3" type="range" :aria-label="`${label(k)} slider`"
            :min="fields[k].minimum" :max="fields[k].maximum" step=".001" :value="value(k)" @input="edits[k]=$event.target.valueAsNumber">
          <input :id="k" type="number" class="input w-36" required :step="fields[k].kind==='int'?1:'any'" :min="fields[k].minimum" :max="fields[k].maximum" :value="value(k)" @input="edits[k]=$event.target.value===''?'':$event.target.valueAsNumber">
        </template>
        <textarea v-else-if="fields[k].kind==='json'" :id="k" class="textarea block w-full font-mono" :value="text(k)" @input="edits[k]=$event.target.value"></textarea>
        <input v-else :id="k" class="input w-full" :value="value(k)??''" @input="edits[k]=$event.target.value">
        <button v-if="fields[k].nullable" type="button" class="btn btn-xs mt-1" @click="edits[k]=null">Set to null</button>
      </div>
      <button type="button" class="btn btn-outline" @click="openPicker">{{ picker?'Close search':'+ Add setting…' }}</button>
      <div v-if="picker" class="space-y-2"><input ref="searchBox" v-model="search" class="input w-full" aria-label="Find a setting" placeholder="Search by name, e.g. early stop">
        <div class="max-h-48 overflow-auto"><button v-for="k in results" :key="k" type="button" class="block text-left w-full p-2 text-sm hover:bg-base-200" @click="add(k)">{{ k }}</button>
          <span v-if="!results.length" class="muted">No matching settings.</span></div></div>
      <div class="flex flex-wrap justify-between items-center gap-2"><span class="text-sm muted">{{ Object.keys(edits).length }} explicit overrides</span>
        <button :disabled="busy || !visible.length" class="btn btn-primary">{{ busy?'Checking…':'Preview & check inputs' }}</button></div>
    </form>
    <p v-if="error" role="alert" class="alert alert-error whitespace-pre-wrap">{{ error }}</p>
    <div v-if="result" class="panel p-5"><b>Configuration valid</b><p class="text-sm muted">Resolved with the Python composer. No job started.</p>
      <details class="mt-2"><summary>View resolved configuration</summary><pre class="terminal console rounded-lg p-3 mt-2 max-h-80 overflow-auto">{{ JSON.stringify(result,null,2) }}</pre></details></div>
  </section>
</template>
