// Fixed positions preserve the two readings of the ROSCon slides. Each tuple is
// [left, top, width, height] in the same coordinate system as the SVG wires.
export const nodes = {
  planner: ['Global planner', 'Map + localisation'],
  path: ['Global path', 'Reference route'],
  sensors: ['Lidar + state', 'Robot observations'],
  controller: ['DWARL', 'Local controller', 'Explore inside ↗'],
  guard: ['Collision monitor', 'Command guard'],
  robot: ['Robot', 'Drive + feedback'],
  recovery: ['Blocked?   →   Recovery / replan   →   Updated path'],
  learning: ['PPO · asymmetric actor–critic', 'Update weights · training only'],
  observation: ['Observation', '180 rays × 3 scans', '4 path lookaheads', 'Measured v, ω + last action'],
  actor: ['Actor', '1D CNN → 3-layer MLP', 'Scan + path + motion'],
  window: ['Dynamic window', 'Limit velocity change', 'by acceleration and speed'],
  environment: ['Environment', 'Robot dynamics · lidar · obstacles', 'Footprint collisions · stop zones'],
}

export const nav = {
  size: [960, 320],
  boxes: {
    planner: [254, 16, 200, 70], path: [506, 16, 200, 70],
    sensors: [8, 139, 200, 95], controller: [254, 139, 200, 95],
    guard: [506, 139, 200, 95], robot: [752, 139, 200, 95],
    recovery: [176, 265, 608, 42],
  },
  wires: ['M458 51 H499', 'M606 87 V104 H354 V134', 'M212 186 H248', 'M458 186 H500', 'M710 186 H746'],
}
export const loop = {
  size: [960, 386],
  boxes: {
    learning: [337, 22, 265, 60], observation: [8, 121, 241, 126],
    actor: [337, 121, 265, 126], window: [695, 121, 245, 126],
    environment: [322, 292, 300, 84],
  },
  wires: ['M250 184 H331', 'M603 184 H689', 'M818 248 V325 H628', 'M321 337 H120 V253'],
  dashed: ['M469 83 V115', 'M623 353 H949 V52 H608'],
  labels: [[646, 169, 'Δv, Δω'], [860, 290, 'command'], [208, 325, 'next observation'], [777, 39, 'rollouts + reward + critic state']],
}
// At phone widths use two columns, rather than shrinking the slide lettering.
export const navSmall = {
  size: [360, 392],
  boxes: {
    planner: [4, 8, 155, 70], path: [200, 8, 155, 70],
    sensors: [4, 119, 155, 90], controller: [200, 119, 155, 90],
    guard: [200, 252, 155, 70], robot: [4, 252, 155, 70],
    recovery: [4, 347, 351, 40],
  },
  wires: ['M160 43 H194', 'M278 79 V113', 'M160 164 H194', 'M278 210 V246', 'M199 287 H165'],
}
export const loopSmall = {
  size: [360, 502],
  boxes: {
    learning: [10, 12, 251, 66], observation: [4, 126, 158, 119],
    actor: [198, 126, 157, 119], window: [198, 304, 157, 92],
    environment: [4, 304, 158, 162],
  },
  wires: ['M163 182 H192', 'M275 246 V298', 'M197 350 H168', 'M85 303 V251'],
  dashed: ['M230 79 V120', 'M80 467 V486 H358 V45 H267'],
  labels: [[295, 278, 'Δv, Δω'], [84, 279, 'next observation']],
}
