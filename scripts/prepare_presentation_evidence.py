#!/usr/bin/env python3
"""Prepare audited held-out comparison evidence for the ROSCon presentation.

Manifest: {"final": "/path/to/test", "rounds": [{"name": "label", "path": "..."}]}.
Run after experiments on an idle host: this also benchmarks recorded ONNX inputs.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import yaml
from audit_comparison import audit

from dwarl.testing.compute import distribution
from dwarl.testing.model_benchmark import execute as benchmark


def prepare(manifest_path, deck):
    manifest = json.loads(manifest_path.read_text())
    final = Path(manifest['final']).resolve()
    checked = audit(final)
    if not checked['evidence_consistent']:
        raise ValueError(checked['errors'])
    summary = json.loads((final / 'summary.json').read_text())
    if summary['suite'] != 'test':
        raise ValueError('Presentation final evidence must come from held-out test cases')
    episodes = json.loads((final / 'episodes.json').read_text())
    labels = {'dwarl': 'gazebo', 'mppi': 'gazebo-mppi', 'dwb': 'gazebo-dwb'}
    if set(episodes) != set(labels.values()):
        raise ValueError('Expected all three Gazebo controllers')
    cases = json.loads((final / 'cases.json').read_text())['cases']
    bundle = Path(summary['bundle'])
    deploy = yaml.safe_load((bundle / 'deploy.yaml').read_text())
    budget_ms = float(deploy['control']['dt']) * 1000
    destination = deck / 'evidence'
    destination.mkdir(parents=True, exist_ok=True)
    (deck / 'assets').mkdir(exist_ok=True)
    # Unique evidence directory names prevent mixing previous final runs.
    if (destination / 'final').exists():
        raise FileExistsError('Archive the previous evidence/final before preparing another study')
    shutil.copytree(final, destination / 'final')
    shutil.copy2(manifest_path, destination / 'study.json')
    data = {'study': manifest, 'final_summary': summary, 'cases': len(cases),
            'map_types': sorted({c['name'] for c in cases}), 'controllers': {}, 'tuning': {},
            'control_budget_ms': budget_ms, 'audit': checked,
            'measurement_notes': ['Latency pools received calls, including failed controller calls.',
                                  'CPU 100% is one logical core; controller server includes costmap and ROS.',
                                  'RSS is sampled; no physical-robot performance was measured.']}
    for controller, label in labels.items():
        rows = episodes[label]
        timings, cpu, rss = [], [], []
        for row in rows:
            if not row.get('compute_file'):
                raise ValueError(f'Missing compute evidence: {label}/{row["case_id"]}')
            raw = json.loads((final / row['compute_file']).read_text())
            timings.extend(raw['timings'])
            groups = {}
            for sample in raw['process_samples']:
                if sample['group'] != 'controller_server':
                    continue
                group = groups.setdefault(sample['wall_time_s'], {'cpu': [], 'rss': 0})
                if sample['cpu_percent'] is not None:
                    group['cpu'].append(sample['cpu_percent'])
                group['rss'] += sample['rss_bytes']
            cpu.extend(sum(v['cpu']) for v in groups.values() if v['cpu'])
            rss.extend(v['rss'] / 1048576 for v in groups.values())
        calls = [r['controller_ms'] for r in timings if 'controller_ms' in r]
        data['controllers'][controller] = {
            'successes': sum(r['success'] for r in rows), 'collisions': sum(r['collision'] for r in rows),
            'timeouts': sum(r['timeout'] for r in rows),
            'outside_goal': sum(r['termination_reason'] == 'nav2_succeeded_outside_goal_tolerance' for r in rows),
            'controller_ms': distribution(calls),
            'inference_ms': distribution([r['inference_ms'] for r in timings if 'inference_ms' in r]),
            'observation_ms': distribution([r['observation_ms'] for r in timings if 'observation_ms' in r]),
            'cpu_percent': distribution(cpu), 'rss_mib': distribution(rss),
            'received_controller_calls': len(calls), 'over_budget': sum(v > budget_ms for v in calls),
        }
        row = rows[0]  # Illustration selected by frozen case order, not outcome.
        folder = (final / row['episode_file']).parent
        for kind in ('trajectory', 'motion', 'compute'):
            shutil.copy2(folder / f'{kind}.png', deck / 'assets' / f'final_{controller}_{kind}.png')
        data['controllers'][controller]['illustration'] = {
            'name': cases[0]['name'], 'case_id': row['case_id'],
            'outcome': row['outcome'], 'time_s': row['completion_time']}
    for entry in manifest.get('rounds', []):
        path = Path(entry['path'])
        report = json.loads((path / 'summary.json').read_text())
        if report['suite'] != 'validation':
            raise ValueError('Tuning records must be validation evidence')
        for controller, label in labels.items():
            rows = json.loads((path / 'episodes.json').read_text()).get(label)
            if rows is None:
                continue
            data['tuning'].setdefault(controller, []).append({
                'name': entry['name'], 'path': str(path), 'cases': len(rows), 'workload': report['workload'],
                'successes': sum(r.get('success', False) for r in rows),
                'collisions': sum(r.get('collision', False) for r in rows),
                'audit': audit(path)})
    observations = final / 'gazebo' / 'dwarl' / cases[0]['id'] / 'observations.npz'
    data['standalone'] = benchmark(bundle, observations, destination / 'model-benchmark.json')
    model = onnx.load(bundle / 'policy.onnx')
    data['model'] = {'parameter_count': sum(int(np.prod(t.dims)) for t in model.graph.initializer
                                           if t.name.endswith(('.weight', '.bias'))),
                     'bytes': (bundle / 'policy.onnx').stat().st_size}
    (deck / 'results.json').write_text(json.dumps(data, indent=2))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('deck_directory', type=Path)
    args = parser.parse_args()
    prepare(args.manifest, args.deck_directory)
