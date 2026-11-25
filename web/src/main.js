import { createApp } from 'vue'
import App from './App.vue'
import './style.css'

if (import.meta.env.DEV && new URLSearchParams(location.search).has('prototype')) {
  import('./prototype/Dashboard.vue').then(({ default: prototype }) => createApp(prototype).mount('#app'))
} else {
  createApp(App).mount('#app')
}
