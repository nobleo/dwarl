<script setup>
import { computed, ref, watch } from 'vue'
import LiveView from './LiveView.vue'
const props = defineProps({ data: Object })
const index = ref(0)
const samples = computed(() => props.data?.trace?.samples || [])
watch(() => props.data, () => { index.value = 0 })
const sample = computed(() => samples.value[index.value])
const live = computed(() => sample.value ? {
  kind: 'replay',
  pose: [sample.value.x_m, sample.value.y_m, sample.value.yaw_rad],
  path: props.data.episode.reference_path,
  label: props.data.episode.case_name,
  time_s: sample.value.time_s, clearance_m: sample.value.clearance_m,
  trail: samples.value.slice(0, index.value + 1).map(s => [s.x_m, s.y_m]),
} : null)
</script>
<template>
  <div v-if="data" class="p-5 bg-base-200 space-y-4 border-t border-base-300">
    <div class="flex flex-wrap gap-4 text-sm">
      <strong>{{ data.episode.case_name }} · {{ data.episode.outcome }}</strong>
      <span>{{ data.episode.termination_reason || data.episode.error || 'No termination reason recorded' }}</span>
    </div>
    <template v-if="samples.length">
      <LiveView :live="live" :running="false" />
      <label class="text-xs block">Recorded sample {{ index + 1 }} / {{ samples.length }}
        <input v-model.number="index" type="range" min="0" :max="samples.length - 1" class="range range-primary range-sm mt-3 w-full" aria-label="Replay sample" />
      </label>
      <div class="text-xs flex flex-wrap gap-5">
        <span>Command {{ sample.command_v_m_s?.toFixed(2) ?? '—' }} m/s · {{ sample.command_w_rad_s?.toFixed(2) ?? '—' }} rad/s</span>
        <span>Measured {{ sample.measured_v_m_s?.toFixed(2) ?? '—' }} m/s · {{ sample.measured_w_rad_s?.toFixed(2) ?? '—' }} rad/s</span>
        <span>Goal distance {{ sample.distance_to_goal_m?.toFixed(2) ?? '—' }} m</span>
      </div>
    </template>
    <p v-else class="text-sm muted">No recorded trace is available for this case.</p>
  </div>
</template>
