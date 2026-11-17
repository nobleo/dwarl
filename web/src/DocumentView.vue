<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import DOMPurify from 'dompurify'
const props = defineProps({ src: String, docs: Boolean })
const page = ref(''), content = ref(''), links = ref([]), query = ref(''), index = ref([]), error = ref('')
const article = ref(null), loading = ref(false)
let sequence = 0
const results = computed(() => {
  const words = query.value.toLowerCase().trim().split(/\s+/).filter(Boolean)
  return words.length ? index.value.filter(row => words.every(word => `${row.title} ${row.text}`.toLowerCase().includes(word))).slice(0, 30) : []
})
async function load(url) {
  const request = ++sequence
  loading.value = true
  error.value = ''
  try {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`Unable to load this page (${response.status})`)
    const doc = new DOMParser().parseFromString(await response.text(), 'text/html')
    const body = doc.querySelector(props.docs ? 'article.md-content__inner' : 'main')
    if (!body) throw new Error('Document content not found. Rebuild with pixi run web-build.')
    const base = response.url
    body.querySelectorAll('[href], [src]').forEach(el => {
      for (const attr of ['href', 'src']) {
        if (el.hasAttribute(attr)) el.setAttribute(attr, new URL(el.getAttribute(attr), base).href)
      }
    })
    body.querySelectorAll('img').forEach(el => el.setAttribute('loading', 'lazy'))
    if (request !== sequence) return
    if (props.docs && !links.value.length) {
      const seen = new Set()
      links.value = [...doc.querySelectorAll('.md-nav--primary a.md-nav__link[href]')].flatMap(a => {
        const href = new URL(a.getAttribute('href'), base)
        if (href.hash || seen.has(href.pathname) || !href.pathname.startsWith('/docs/')) return []
        seen.add(href.pathname)
        return [{ title: a.textContent.trim(), href: href.href }]
      })
    }
    content.value = DOMPurify.sanitize(body.innerHTML)
    page.value = new URL(url, location.href).href
    await nextTick()
    const hash = new URL(page.value).hash
    if (hash) article.value?.querySelector(`[id="${CSS.escape(decodeURIComponent(hash.slice(1)))}"]`)?.scrollIntoView()
    else if (props.docs) article.value?.scrollIntoView({ block: 'start' })
  } catch (e) { if (request === sequence) error.value = e.message }
  finally { if (request === sequence) loading.value = false }
}
function navigate(event) {
  const anchor = event.target.closest('a')
  if (!anchor) return
  const url = new URL(anchor.href)
  if (url.origin === location.origin && (url.pathname.startsWith('/docs/') || url.pathname.endsWith('.html'))) {
    event.preventDefault()
    query.value = ''
    load(url.href)
  } else {
    anchor.target = '_blank'
    anchor.rel = 'noopener'
  }
}
watch(() => props.src, url => { if (url) load(url) }, { immediate: true })
onMounted(async () => {
  if (props.docs) {
    try { index.value = (await (await fetch('/docs/search/search_index.json')).json()).docs }
    catch (e) { error.value = `Search unavailable: ${e.message}` }
  }
})
</script>
<template>
  <div :class="docs ? 'grid lg:grid-cols-[240px_1fr] min-h-[70vh]' : ''">
    <aside v-if="docs" class="p-4 border-r border-base-300 bg-base-200 lg:sticky lg:top-4 self-start max-h-[85vh] overflow-auto">
      <input v-model="query" type="search" class="input input-sm w-full mb-4" placeholder="Search guides & reference" aria-label="Search documentation" />
      <nav aria-label="Documentation pages" class="flex flex-col gap-1 text-sm">
        <template v-if="query.trim()">
          <p class="text-xs muted mb-2">{{ results.length }} matching sections</p>
          <button v-for="(item, i) in results" :key="i" class="text-left px-3 py-2 hover:bg-base-300 rounded-lg" @click="load(`/docs/${item.location}`); query = ''">{{ item.title }}</button>
        </template>
        <button v-for="item in query.trim() ? [] : links" :key="item.href" class="text-left px-3 py-2 rounded-lg" :class="page.split('#')[0] === item.href ? 'bg-primary text-primary-content' : 'hover:bg-base-300'" @click="load(item.href)">{{ item.title }}</button>
      </nav>
    </aside>
    <div class="min-w-0">
      <p v-if="loading" class="px-6 py-3 text-sm muted">Loading…</p>
      <p v-if="error" class="alert alert-error m-5">{{ error }}</p>
      <article ref="article" class="md-typeset dwarl-content p-5 sm:p-8" @click="navigate" v-html="content"></article>
    </div>
  </div>
</template>
