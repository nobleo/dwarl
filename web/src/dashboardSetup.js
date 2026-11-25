import { ref } from 'vue'

export function dashboardSetup(api) {
  const setup = ref(null), message = ref(''), pending = ref(0)
  let queue = Promise.resolve()
  // Preserve edit order when a second drag or change happens during a write.
  function run(action) {
    pending.value++
    queue = queue.then(action).catch(e => { message.value = e.message }).finally(() => pending.value--)
    return queue
  }
  function load() {
    return run(async () => { setup.value = await api('setup'); message.value = 'Loaded from web/dashboard-setup.json (or defaults).' })
  }
  function save() {
    const snapshot = JSON.parse(JSON.stringify(setup.value))
    return run(async () => { await api('setup', snapshot); message.value = 'Saved to web/dashboard-setup.json.' })
  }
  function update(patch) {
    setup.value = { ...setup.value, ...patch }
    return save()
  }
  function importFile(file) {
    if (!file) return
    return run(async () => {
      if (file.size > 65536) throw Error('Setup exceeds 64 KiB.')
      setup.value = await api('setup', JSON.parse(await file.text()))
      message.value = 'Imported and saved workspace setup.'
    })
  }
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(setup.value, null, 2)+'\n'], {type:'application/json'}))
    const link = document.createElement('a')
    link.href = url; link.download = 'dashboard-setup.json'; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return { setup, message, pending, load, save, update, importFile, download }
}
