// Deterministic sample traces for the design review, not recorded training results.
const definitions = [
  ['Train/mean_reward', 'Mean reward', 'reward', '#6960c9', t => -12 + 48 * (1-Math.exp(-3*t)), 6],
  ['Validation/success_rate', 'Validation success', '%', '#008374', t => 18 + 72 / (1+Math.exp(-7*(t-.38))), 4],
  ['Loss/value', 'Value loss', '', '#c16a23', t => .15 + 2.8*Math.exp(-4*t), .22],
  ['Policy/mean_std', 'Policy spread', 'σ', '#976197', t => .9-.48*t, .04],
  ['Metrics/path_completion', 'Path completion', '%', '#407bae', t => 42+49*t, 4],
  ['Perf/total_fps', 'Throughput', 'fps', '#367f84', t => 710+35*Math.sin(6*t), 40],
  ['Episode_Termination/collision', 'Collision rate', '%', '#bf5468', t => 4+26*Math.exp(-3*t), 3],
  ['Loss/surrogate', 'Surrogate loss', '', '#837036', t => -.12*Math.exp(-4*t), .012],
]
export const signals = Object.fromEntries(definitions.map(([tag,name,unit,color,trend,noise],i) => {
  let seed = 1234+i, ema
  const raw = Array.from({length:121}, (_,j) => {
    seed = (1664525*seed + 1013904223) >>> 0
    return trend(j/120) + noise*((seed/2**32-.5)*2 + .35*Math.sin(j/5))
  })
  const line = raw.map(v => ema = ema == null ? v : ema*.74+v*.26)
  return [tag,{tag,name,unit,color,raw,line}]
}))
export const format = n => Math.abs(n)>=100 ? n.toFixed(0) : Math.abs(n)>=1 ? n.toFixed(1) : n.toFixed(3)
