<script setup>
import { computed, ref, watch } from 'vue'
const props = defineProps({ evaluations: { type: Array, default: () => [] } })
const workload = ref('')
const groups = computed(() => [...new Set(props.evaluations.filter(e => e.summary?.workload).map(e => e.summary.workload))])
watch(groups, values => { if (!values.includes(workload.value)) workload.value = values[0] || '' }, { immediate: true })
const rows = computed(() => props.evaluations.filter(e => e.summary?.workload === workload.value)
  .flatMap(e => Object.entries(e.summary.simulators).map(([backend, stats]) => ({ ...stats, backend, ...e.summary, name: e.name }))))
const number = value => value == null ? '—' : value.toFixed(2)
const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`
</script>
<template>
  <section class="panel overflow-hidden">
    <div class="p-5 border-b border-base-300">
      <p class="eyebrow">Checkpoint comparison</p>
      <div class="flex flex-wrap justify-between items-center gap-3 mt-1">
        <h2 class="font-semibold">Compare the same workload</h2>
        <select v-model="workload" class="select select-sm max-w-full" aria-label="Comparison workload">
          <option v-if="!groups.length" value="">No completed workloads</option>
          <option v-for="id in groups" :key="id" :value="id">Workload {{ id.slice(0, 12) }}</option>
        </select>
      </div>
      <p class="text-xs muted mt-3">Grouped by exact case-set identity. Compare the same backend and configuration; incomplete results are labelled. Time is averaged over successful cases only.</p>
    </div>
    <div class="overflow-auto">
      <table v-if="rows.length" class="table table-sm">
        <thead><tr><th>Iteration / suite</th><th>Backend</th><th>Complete</th><th>Success</th><th>Collision</th><th>Time · sim s</th><th>Mean min clearance · m</th></tr></thead>
        <tbody><tr v-for="row in rows" :key="`${row.name}-${row.backend}`">
          <td>{{ row.iteration }} · {{ row.suite }}<span class="block text-xs muted">{{ row.name }}</span></td>
          <td>{{ row.backend }}</td><td>{{ row.completed }}/{{ row.expected }} · {{ row.status }}</td>
          <td>{{ percent(row.success_rate) }}</td><td>{{ percent(row.collision_rate) }}</td>
          <td>{{ number(row.successful_mean_time_s) }}</td><td>{{ number(row.mean_min_clearance_m) }}</td>
        </tr></tbody>
      </table>
      <p v-else class="p-5 text-sm muted">Evaluate checkpoints on the same suite to compare their recorded outcomes here.</p>
    </div>
  </section>
</template>
