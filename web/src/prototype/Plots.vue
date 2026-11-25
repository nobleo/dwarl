<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import Draggable from 'vuedraggable'
import { Activity, BookOpen, Check, ChevronRight, GripVertical, Pin, Plus, Search, X } from 'lucide-vue-next'
import Trace from './Trace.vue'
import { signals, format } from './signals'
const props=defineProps({variant:String})
const tags=Object.keys(signals), key='dwarl-prototype-pins-v1'
const pins=ref(JSON.parse(localStorage.getItem(key)||JSON.stringify(tags.slice(0,4))).filter(t=>signals[t]))
watch(pins,v=>localStorage.setItem(key,JSON.stringify(v)),{deep:true})
const selected=ref(''), active=computed(()=>signals[pins.value.includes(selected.value)?selected.value:pins.value[0]])
const dialog=ref(null), search=ref(''), peek=ref(''), input=ref(null)
const matches=computed(()=>tags.filter(t=>(t+signals[t].name).toLowerCase().includes(search.value.toLowerCase())))
const review=computed(()=>signals[matches.value.includes(peek.value)?peek.value:matches.value[0]])
function browse(t) {
  peek.value=t || tags.find(t=>!pins.value.includes(t)) || tags[0]
  search.value=''; dialog.value.showModal(); nextTick(()=>input.value.focus())
}
function toggle(t){if(pins.value.includes(t)) pins.value=pins.value.filter(v=>v!==t);else pins.value.push(t)}
function move(t,to){const from=pins.value.indexOf(t);if(from>=0 && to>=0 && to<pins.value.length)pins.value.splice(to,0,pins.value.splice(from,1)[0])}
const last=s=>format(s.raw.at(-1))
</script>
<template>
  <div class="studio">
    <header class="mast">
      <div><div class="kicker"><Activity :size="15"/> RUN ANALYSIS <span>/</span> SAMPLE 001</div>
        <h1>Training signals</h1><p>Navigation policy <span>·</span> 12,000 steps <span>·</span> Sample traces</p></div>
      <button class="library-button" @click="browse()"><BookOpen :size="16"/> Metric library <span>{{ tags.length }}</span></button>
    </header>
    <div class="board-head"><h2>{{ {A:'My plots',B:'Signal ledger',C:'Inspection desk'}[variant] }}</h2><span>{{ pins.length }} pinned · saved locally</span></div>
    <div v-if="!pins.length" class="empty">No pinned signals yet. <button class="link" @click="browse()">Open metric library</button></div>
    <div v-else class="layout" :class="'mode-'+variant">
      <Draggable v-model="pins" :item-key="id => id" handle=".grip" ghost-class="grabbed" :animation="160" class="collection">
        <template #item="{ element:t, index:i }">
        <article :data-slot="t" class="signal" :class="{chosen:variant==='C' && active.tag===t}" :style="{'--ink':signals[t].color}">
          <div class="signal-head">
            <button class="grip" :aria-label="`Reorder ${signals[t].name}, use up or down arrow`" @keydown.up.prevent="move(t,i-1)" @keydown.down.prevent="move(t,i+1)"><GripVertical :size="18"/></button>
            <div class="identity"><small>{{ t.split('/')[0].replace('_',' ') }}</small>
              <button @click="variant==='C'?selected=t:browse(t)">{{ signals[t].name }}</button></div>
            <button v-if="variant==='A'" class="icon remove" :aria-label="`Unpin ${signals[t].name}`" @click="toggle(t)"><Pin :size="14"/></button>
          </div>
          <div class="reading">{{ last(signals[t]) }} <small>{{ signals[t].unit }}</small></div>
          <Trace v-if="variant!=='C'" :signal="signals[t]" :mini="variant==='B'"/>
          <button v-if="variant==='B'" class="icon" :aria-label="`Inspect ${signals[t].name}`" @click="browse(t)"><ChevronRight :size="18"/></button>
        </article>
        </template>
      </Draggable>
      <section v-if="variant==='C' && active" class="detail">
        <small class="kicker">SIGNAL INSPECTOR</small><h2>{{ active.name }}</h2><code>{{ active.tag }}</code>
        <div class="reading">{{ last(active) }} <small>{{ active.unit }}</small></div><Trace :signal="active"/>
        <button class="link text-sm mt-4" @click="browse(active.tag)">Inspect in library</button>
      </section>
    </div>
    <footer><span><GripVertical :size="14"/> Drag the grip to reorder</span><span>{{ variant==='B'?'Select a name for axes and detail.':variant==='C'?'Select a name to inspect its trace.':'Faint line: samples · solid line: smoothed' }}</span></footer>
    <dialog ref="dialog" aria-label="Metric library" @click.self="dialog.close()">
      <div class="library">
        <header><div><small class="kicker">EXPLORE SIGNALS</small><h2>Metric library</h2></div><button class="icon" aria-label="Close library" @click="dialog.close()"><X :size="19"/></button></header>
        <label class="search"><Search :size="16"/><input ref="input" v-model="search" placeholder="Find a metric or group…" aria-label="Find a metric"></label>
        <div v-if="matches.length" class="library-body">
          <div class="results"><button v-for="t in matches" :key="t" :aria-pressed="review.tag===t" @click="peek=t">
            <i :style="{background:signals[t].color}"/>{{ signals[t].name }}<Pin v-if="pins.includes(t)" :size="12" aria-label="Pinned"/></button></div>
          <div class="preview"><small class="kicker">QUICK LOOK</small><h3>{{ review.name }}</h3><code>{{ review.tag }}</code>
            <div class="reading">{{ last(review) }} <small>{{ review.unit }}</small></div><Trace :signal="review"/>
            <button class="pin-action" @click="toggle(review.tag)"><component :is="pins.includes(review.tag)?Check:Plus" :size="16"/>{{ pins.includes(review.tag)?'Pinned · remove':'Pin to my board' }}</button>
          </div>
        </div><p v-else class="empty">No signals match “{{ search }}”.</p>
      </div>
    </dialog>
  </div>
</template>
<style scoped>
.studio{color:#262334;--border:#dfdde5;--quiet:#686474}.mast{background:#2d293f;color:#fff;border-radius:18px;padding:30px 34px;display:flex;justify-content:space-between;align-items:center;gap:20px}.kicker,.identity small{font:600 10px/1.4 ui-monospace,monospace;letter-spacing:.12em;display:flex;gap:9px;align-items:center}.mast .kicker{color:#d4c5fc}.mast h1{font-size:clamp(32px,4vw,52px);font-weight:620;letter-spacing:-.055em;line-height:1.15;margin:9px 0}.mast p{color:#ddd9e9;font-size:13px}.mast p span{padding:0 4px}.library-button,.pin-action{display:inline-flex;align-items:center;gap:9px;padding:10px 14px;background:white;color:#332c43;border-radius:9px;font-size:13px;font-weight:600}.library-button span{background:#ece8f3;padding:0 5px;border-radius:4px}.board-head,.studio footer,.library header{display:flex;align-items:center;justify-content:space-between;gap:12px}.board-head{margin:25px 0 12px}.board-head h2,.detail h2,.library h2,.preview h3{font-weight:620;letter-spacing:-.035em}.board-head h2{font-size:22px}.board-head span,.studio footer{font-size:12px;color:var(--quiet)}.collection{display:grid;gap:12px}.mode-A .collection{grid-template-columns:repeat(2,minmax(0,1fr))}.signal,.detail,.empty,.preview{background:white;border:1px solid var(--border);border-radius:12px}.signal{padding:17px;min-width:0;position:relative}.signal-head,.studio footer span{display:flex;align-items:center;gap:8px}.identity small{color:var(--ink);text-transform:uppercase;font-size:9px}.identity button{font-weight:620;text-align:left}.grip,.icon{padding:5px;border-radius:6px;color:#7a7483;display:grid;place-items:center}.grip{cursor:grab;touch-action:none}.grip:hover,.icon:hover{background:#f1eef5}.remove{margin-left:auto}.reading{font:550 32px/1.25 ui-monospace,monospace;letter-spacing:-.07em;margin:10px 0}.reading small{font:500 12px system-ui;color:var(--quiet);letter-spacing:0}.studio footer{margin-top:15px;flex-wrap:wrap}.target{outline:3px solid #8877d9;outline-offset:-2px;background:#f7f5ff}.grabbed{opacity:.55}.sort-move{transition:transform .24s ease}.mode-B .collection{gap:0;background:white;border:1px solid var(--border);border-radius:12px;overflow:hidden}.mode-B .signal{display:grid;grid-template-columns:minmax(0,2fr) 1fr 2fr auto;align-items:center;gap:16px;border:0;border-bottom:1px solid var(--border);border-radius:0}.mode-B .signal:last-child{border:0}.mode-B .reading,.mode-C .signal .reading{font-size:20px}.mode-C{display:grid;grid-template-columns:minmax(240px,1fr) 2fr;gap:12px;align-items:start}.mode-C .signal{padding:10px;display:flex;align-items:center;justify-content:space-between}.chosen{box-shadow:inset 4px 0 var(--ink)}.detail,.preview{padding:24px}.detail{--trace-height:190px}.detail h2,.library h2,.preview h3{font-size:24px}.studio code{font-size:11px;color:var(--quiet);overflow-wrap:anywhere}.detail .reading{margin:24px 0}.empty{padding:30px}.studio dialog{padding:0;border:0;margin:auto;width:min(760px,94vw);max-height:88dvh;overflow:auto;background:white;color:#262334;border-radius:16px;box-shadow:0 24px 90px #231d3240}.studio dialog::backdrop{background:#211a3080}.library{padding:22px}.library header,.search{margin-bottom:15px}.search{display:flex;align-items:center;gap:8px;background:#f5f3f7;padding:9px 12px;border-radius:8px}.search input{width:100%;background:none}.library-body{display:grid;grid-template-columns:1fr 1.65fr;gap:15px}.results button{display:flex;align-items:center;gap:8px;text-align:left;width:100%;padding:9px;border-radius:7px;font-size:13px}.results i{width:8px;height:8px;border-radius:50%}.results button[aria-pressed=true]{background:#edeaf6}.results svg{margin-left:auto}.pin-action{background:#342d49;color:white;margin-top:20px}.preview{min-width:0}.preview .kicker,.detail .kicker{color:#706589}
@media(max-width:680px){.mast{align-items:flex-start;flex-direction:column;padding:22px}.mode-A .collection,.mode-C,.library-body{grid-template-columns:1fr}.mode-B .signal{gap:5px;padding:10px;grid-template-columns:1.8fr 1fr 1fr auto}.mode-B .reading{font-size:15px}.results{max-height:145px;overflow:auto}.preview{padding:15px}}
@media(prefers-reduced-motion:reduce){.sort-move{transition:none}}
</style>
