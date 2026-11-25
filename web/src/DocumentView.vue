<script setup>
import { nextTick, ref, watch } from 'vue'
import DOMPurify from 'dompurify'
const props = defineProps({ src: String })
const content = ref(''), error = ref(''), article = ref(null), loading = ref(false)
let sequence = 0
async function load(url) {
  const request = ++sequence
  loading.value = true
  error.value = ''
  try {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`Unable to load this page (${response.status})`)
    const doc = new DOMParser().parseFromString(await response.text(), 'text/html')
    const body = doc.querySelector('main')
    if (!body) throw new Error('Report content not found. Regenerate it with dwarl report.')
    const base = response.url
    body.querySelectorAll('[href], [src]').forEach(el => {
      for (const attr of ['href', 'src']) {
        if (el.hasAttribute(attr)) el.setAttribute(attr, new URL(el.getAttribute(attr), base).href)
      }
    })
    body.querySelectorAll('img').forEach(el => el.setAttribute('loading', 'lazy'))
    if (request !== sequence) return
    content.value = DOMPurify.sanitize(body.innerHTML)
    await nextTick()
    const hash = new URL(url, location.href).hash
    if (hash) article.value?.querySelector(`[id="${CSS.escape(decodeURIComponent(hash.slice(1)))}"]`)?.scrollIntoView()
  } catch (e) { if (request === sequence) error.value = e.message }
  finally { if (request === sequence) loading.value = false }
}
function navigate(event) {
  const anchor = event.target.closest('a')
  if (!anchor) return
  const url = new URL(anchor.href)
  if (url.origin === location.origin && url.pathname.startsWith('/reports/') && url.pathname.endsWith('.html')) {
    event.preventDefault()
    load(url.href)
  } else {
    anchor.target = '_blank'
    anchor.rel = 'noopener'
  }
}
watch(() => props.src, url => { if (url) load(url) }, { immediate: true })
</script>
<template>
  <div class="min-w-0 border-b border-base-300">
    <p v-if="loading" class="px-6 py-3 text-sm muted">Loading…</p>
    <p v-if="error" class="alert alert-error m-5">{{ error }}</p>
    <article ref="article" class="md-typeset dwarl-content p-5 sm:p-8" @click="navigate" v-html="content"></article>
  </div>
</template>
