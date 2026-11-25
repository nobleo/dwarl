<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { nodes, nav, loop, navSmall, loopSmall } from './layout'

const root = ref(null), dialog = ref(null), heading = ref(null), back = ref(null)
const inside = ref(false), narrow = ref(false), topic = ref('')
const layout = computed(() => inside.value ? (narrow.value ? loopSmall : loop) : (narrow.value ? navSmall : nav))
const related = {
  environment: ['routes', 'reward'], learning: ['reward', 'validation'],
  actor: ['bundle'], validation: ['bundle'], bundle: ['controller'],
}
const section = id => document.querySelector(`.theory-notes h2[id="${id}"]`)
const title = id => section(id)?.textContent
const paragraphs = computed(() => {
  const html = []
  for (let p = section(topic.value)?.nextElementSibling; p?.matches('p'); p = p.nextElementSibling) html.push(p.outerHTML)
  return html.join('')
})
function boxStyle([x, y, w, h]) {
  const [width, height] = layout.value.size
  return {left: `${100*x/width}%`, top: `${100*y/height}%`, width: `${100*w/width}%`, height: `${100*h/height}%`}
}
async function inspect(id) {
  topic.value = id
  await nextTick()
  if (!dialog.value.open) dialog.value.showModal()
  heading.value.focus()
  dialog.value.scrollTop = 0
}
async function changeLevel(value) {
  inside.value = value
  await nextTick()
  ;(value ? back.value : root.value.querySelector('[data-node="controller"]')).focus()
}
let observer
onMounted(() => {
  observer = new ResizeObserver(([entry]) => { narrow.value = entry.contentRect.width < 620 })
  observer.observe(root.value)
})
onUnmounted(() => observer?.disconnect())
</script>

<template>
  <section ref="root" class="theory" :class="{ compact: narrow }" aria-label="Interactive system architecture">
    <header>
      <div class="theory-location">
        <button v-if="inside" ref="back" type="button" @click="changeLevel(false)">← Nav2 integration</button>
        <span v-else>System architecture</span>
        <span v-if="inside"> / DWARL</span>
      </div>
      <h2>{{ inside ? 'Inputs, actions and training loop' : 'Nav2 integration' }}</h2>
    </header>
    <div class="theory-stage" :style="{ aspectRatio: layout.size.join(' / ') }">
      <svg :viewBox="`0 0 ${layout.size.join(' ')}`" aria-hidden="true" class="theory-wires">
        <defs><marker id="theory-tip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M2 1 L8 5 L2 9" fill="none" stroke="currentColor" stroke-width="1.6" />
        </marker></defs>
        <path v-for="d in layout.wires" :key="d" :d="d" class="wire" />
        <path v-for="d in layout.dashed" :key="d" :d="d" class="wire training-wire" />
        <text v-for="[x,y,text] in layout.labels" :key="text" :x="x" :y="y" text-anchor="middle">{{ text }}</text>
      </svg>
      <button v-for="(box, id) in layout.boxes" :key="id" type="button" class="theory-node"
        :data-node="id" :style="boxStyle(box)" :aria-haspopup="id === 'controller' ? undefined : 'dialog'"
        @click="id === 'controller' ? changeLevel(true) : inspect(id)">
        <strong>{{ nodes[id][0] }}</strong>
        <span v-for="line in nodes[id].slice(1)" :key="line">{{ line }}</span>
      </button>
    </div>
    <footer v-if="inside">
      <b>One tick:</b> solid arrows carry observations and commands. Dashed arrows train the actor.
      <button @click="inspect('bundle')">What gets deployed? →</button>
    </footer>
    <footer v-else><b>Follow one decision:</b> path and sensor feedback meet at DWARL; the collision monitor gates its command.</footer>
    <dialog ref="dialog" class="theory-dialog" aria-labelledby="theory-heading">
      <div class="theory-location">{{ inside ? 'DWARL / Inside the controller' : 'System architecture / Nav2' }}</div>
      <h2 id="theory-heading" ref="heading" tabindex="-1">{{ title(topic) }}</h2>
      <!-- Trusted Markdown from this same built documentation page. -->
      <div v-html="paragraphs"></div>
      <div class="theory-actions">
        <button @click="dialog.close()">Back to diagram</button>
        <button v-for="id in related[topic]" :key="id" @click="inspect(id)">{{ title(id) }} →</button>
      </div>
    </dialog>
  </section>
</template>
