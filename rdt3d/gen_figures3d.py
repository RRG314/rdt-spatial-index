"""
Figure generation for RDT3D evaluation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def load_result(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run validate3d.py, benchmark3d.py, and stress3d.py first.")
    with path.open() as f:
        return json.load(f)


# Load data
bench_data = load_result("benchmark3d.json")
stress_data = load_result("stress3d.json")
valid_data = load_result("validation3d.json")

fig_dir = BASE_DIR / "figures"
fig_dir.mkdir(exist_ok=True)

# Color scheme
colors = {
    'RDT3D-Python': '#1f77b4',
    'RDT3D-Vectorized': '#1f77b4',
    'RDT3D-C': '#1f77b4',
    'scipy-KDTree': '#ff7f0e',
    'R-tree': '#2ca02c',
    'UniformGrid': '#d62728',
    'Octree': '#9467bd',
}

# Figure 1: Build time scaling
print("Generating figure 1: build time scaling...")
fig, ax = plt.subplots(figsize=(10, 6))

for method in ['RDT3D-Python', 'RDT3D-Vectorized', 'scipy-KDTree', 'UniformGrid']:
    data = [b for b in bench_data['benchmarks'] if b['method'] == method]
    if not data:
        continue
    sorted_data = sorted(data, key=lambda x: x['scale'])
    scales_sorted = [b['scale'] for b in sorted_data]
    build_times = [b['build_mean_ms'] for b in sorted_data]
    ax.loglog(scales_sorted, build_times, marker='o', label=method, color=colors[method], linewidth=2)

ax.set_xlabel('Dataset size (points)', fontsize=11)
ax.set_ylabel('Build time (ms)', fontsize=11)
ax.set_title('RDT3D vs Baselines: Build Time Scaling', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'fig01_build_time_scaling.png', dpi=150)
plt.close()

# Figure 2: Query time scaling
print("Generating figure 2: query time scaling...")
fig, ax = plt.subplots(figsize=(10, 6))

for method in ['RDT3D-Python', 'RDT3D-Vectorized', 'scipy-KDTree', 'UniformGrid']:
    data = [b for b in bench_data['benchmarks'] if b['method'] == method]
    if not data:
        continue
    sorted_data = sorted(data, key=lambda x: x['scale'])
    scales_sorted = [b['scale'] for b in sorted_data]
    query_times = [b['query_mean_ms'] for b in sorted_data]
    ax.loglog(scales_sorted, query_times, marker='s', label=method, color=colors[method], linewidth=2)

ax.set_xlabel('Dataset size (points)', fontsize=11)
ax.set_ylabel('Query time (ms)', fontsize=11)
ax.set_title('RDT3D vs Baselines: Query Time Scaling', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'fig02_query_time_scaling.png', dpi=150)
plt.close()

# Figure 3: Build time by distribution (N=50K)
print("Generating figure 3: build time by distribution...")
fig, ax = plt.subplots(figsize=(12, 6))

scale_50k = [b for b in bench_data['benchmarks'] if b['scale'] == 50000]
dists = sorted(set(b['distribution'] for b in scale_50k))
methods = ['RDT3D-Vectorized', 'scipy-KDTree', 'UniformGrid']

x = np.arange(len(dists))
width = 0.25

for i, method in enumerate(methods):
    data = [b for b in scale_50k if b['method'] == method]
    times = [next((b['build_mean_ms'] for b in data if b['distribution'] == d), 0) for d in dists]
    ax.bar(x + i*width, times, width, label=method, color=colors[method])

ax.set_xlabel('Distribution', fontsize=11)
ax.set_ylabel('Build time (ms)', fontsize=11)
ax.set_title('Build Time by Distribution (N=50K)', fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(dists, rotation=45)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(fig_dir / 'fig03_build_vs_dist.png', dpi=150)
plt.close()

# Figure 4: Query time by distribution (N=50K)
print("Generating figure 4: query time by distribution...")
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(dists))
for i, method in enumerate(methods):
    data = [b for b in scale_50k if b['method'] == method]
    times = [next((b['query_mean_ms'] for b in data if b['distribution'] == d), 0) for d in dists]
    ax.bar(x + i*width, times, width, label=method, color=colors[method])

ax.set_xlabel('Distribution', fontsize=11)
ax.set_ylabel('Query time (ms)', fontsize=11)
ax.set_title('Query Time by Distribution (N=50K)', fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(dists, rotation=45)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(fig_dir / 'fig04_query_vs_dist.png', dpi=150)
plt.close()

# Figure 5: Stress test results
print("Generating figure 5: stress test comparison...")
fig, ax = plt.subplots(figsize=(14, 6))

stress_tests = stress_data['tests']
test_names = [t['name'] for t in stress_tests]
rdt3d_times = [t['methods']['RDT3D-Vectorized']['query_ms'] for t in stress_tests]
kdtree_times = [t['methods']['scipy-KDTree']['query_ms'] for t in stress_tests]

x = np.arange(len(test_names))
width = 0.35

ax.bar(x - width/2, rdt3d_times, width, label='RDT3D-Vectorized', color=colors['RDT3D-Vectorized'])
ax.bar(x + width/2, kdtree_times, width, label='scipy-KDTree', color=colors['scipy-KDTree'])

ax.set_xlabel('Stress Test', fontsize=11)
ax.set_ylabel('Query time (ms, log scale)', fontsize=11)
ax.set_title('Stress Test Results (N=50K, Q=100)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(test_names, rotation=45, ha='right')
ax.set_yscale('log')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(fig_dir / 'fig05_stress_comparison.png', dpi=150)
plt.close()

# Figure 6: Correctness summary
print("Generating figure 6: correctness summary...")
fig, ax = plt.subplots(figsize=(12, 6))

validation_tests = valid_data['test_cases']
index_types = ['RDT3D-Python', 'RDT3D-Vectorized', 'scipy-KDTree']
pass_counts = {idx: 0 for idx in index_types}
total_counts = {idx: 0 for idx in index_types}

for test in validation_tests:
    for idx_result in test['indices']:
        idx_name = idx_result['index']
        if idx_name in pass_counts:
            if idx_result['passed']:
                pass_counts[idx_name] += 1
            total_counts[idx_name] += 1

pass_rates = [100 * pass_counts[idx] / total_counts[idx] for idx in index_types]

colors_list = [colors[idx] for idx in index_types]
bars = ax.bar(index_types, pass_rates, color=colors_list)

# Add percentage labels on bars
for bar, rate in zip(bars, pass_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.0f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Correctness Pass Rate (%)', fontsize=11)
ax.set_title('Validation: Correctness Results', fontsize=12, fontweight='bold')
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(fig_dir / 'fig06_correctness_summary.png', dpi=150)
plt.close()

print(f"\nAll figures generated in {fig_dir}")
print("  - fig01_build_time_scaling.png")
print("  - fig02_query_time_scaling.png")
print("  - fig03_build_vs_dist.png")
print("  - fig04_query_vs_dist.png")
print("  - fig05_stress_comparison.png")
print("  - fig06_correctness_summary.png")
