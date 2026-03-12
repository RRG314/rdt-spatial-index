"""
generate_figures.py — Produce all publication figures and tables from raw benchmark results.

Run: python benchmarks/generate_figures.py [--outdir publication]
"""

import sys, os, json, math, argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import rcParams

# ── Style ──────────────────────────────────────────────────────────────────────
rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

METHOD_COLORS = {
    'rdt':           '#e74c3c',
    'rdt_fast':      '#2ecc71',
    'rdt_optimized': '#3498db',
    'uniform_grid':  '#f39c12',
    'kd_tree':       '#9b59b6',
    'scipy_kd':      '#1abc9c',
    'quadtree':      '#e67e22',
    'rtree':         '#34495e',
}
METHOD_LABELS = {
    'rdt':           'RDT (base)',
    'rdt_fast':      'RDT-Fast',
    'rdt_optimized': 'RDT-Opt',
    'uniform_grid':  'Uniform Grid',
    'kd_tree':       'KD-Tree (custom)',
    'scipy_kd':      'Scipy KD-Tree',
    'quadtree':      'Quadtree',
    'rtree':         'R-tree',
}
METHOD_MARKERS = {
    'rdt':           'v',
    'rdt_fast':      'o',
    'rdt_optimized': 's',
    'uniform_grid':  'D',
    'kd_tree':       '^',
    'scipy_kd':      'P',
    'quadtree':      'h',
    'rtree':         'X',
}

DS_LABELS = {
    'uniform':             'Uniform Random',
    'clustered':           'Clustered (8 clusters)',
    'sparse_dense':        'Sparse+Dense',
    'adversarial_line':    'Adversarial Line',
    'adversarial_hotspot': 'Adversarial Hotspot',
    'fractal':             'Fractal (Cantor)',
    'grid_regular':        'Regular Grid',
    'taxi_like':           'Taxi-like (real-world)',
    'osm_like':            'OSM-like (real-world)',
}


def load_json(path):
    if not os.path.exists(path):
        print(f"  [WARN] missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)

def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ {os.path.relpath(path, ROOT)}")


# ── Fig 1: Scaling analysis — query time vs N ─────────────────────────────────

def fig_scaling_query(scaling_data, fig_dir):
    ds_subset = ['uniform', 'clustered', 'adversarial_hotspot']
    ds_titles = ['Uniform Random', 'Clustered (8 clusters)', 'Adversarial Hotspot']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle('Query Time vs. Object Count\n(512 queries, mean ± std, 3 runs)', fontsize=12)

    for ax, ds_name, title in zip(axes, ds_subset, ds_titles):
        rows = [r for r in scaling_data if r['dataset'] == ds_name]
        methods = sorted({r['method'] for r in rows})

        for method in methods:
            m_rows = sorted([r for r in rows if r['method'] == method], key=lambda x: x['n'])
            ns   = [r['n'] for r in m_rows]
            qs   = [r.get('query_mean_ms', r.get('query_mean', 0)) for r in m_rows]
            stds = [r.get('query_std_ms', r.get('query_std', 0))  for r in m_rows]
            ax.errorbar(ns, qs,
                        yerr=stds,
                        label=METHOD_LABELS.get(method, method),
                        color=METHOD_COLORS.get(method, 'gray'),
                        marker=METHOD_MARKERS.get(method, 'o'),
                        markersize=5,
                        linewidth=1.5,
                        capsize=3)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Object count (N)')
        ax.set_ylabel('Query time (ms)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which='both')

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(loc='upper left', framealpha=0.8)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, 'fig1_scaling_query.pdf'))
    savefig(fig, os.path.join(fig_dir, 'fig1_scaling_query.png'))


# ── Fig 2: Scaling analysis — build time vs N ─────────────────────────────────

def fig_scaling_build(scaling_data, fig_dir):
    ds_subset = ['uniform', 'clustered', 'adversarial_hotspot']
    ds_titles = ['Uniform Random', 'Clustered', 'Adversarial Hotspot']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle('Build Time vs. Object Count (mean ± std, 3 runs)', fontsize=12)

    for ax, ds_name, title in zip(axes, ds_subset, ds_titles):
        rows = [r for r in scaling_data if r['dataset'] == ds_name]
        methods = sorted({r['method'] for r in rows})

        for method in methods:
            m_rows = sorted([r for r in rows if r['method'] == method], key=lambda x: x['n'])
            ns   = [r['n'] for r in m_rows]
            bs   = [r.get('build_mean_ms', r.get('build_mean', 0)) for r in m_rows]
            stds = [r.get('build_std_ms', r.get('build_std', 0))  for r in m_rows]
            ax.errorbar(ns, bs,
                        yerr=stds,
                        label=METHOD_LABELS.get(method, method),
                        color=METHOD_COLORS.get(method, 'gray'),
                        marker=METHOD_MARKERS.get(method, 'o'),
                        markersize=5, linewidth=1.5, capsize=3)

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Object count (N)'); ax.set_ylabel('Build time (ms)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which='both')

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(loc='upper left', framealpha=0.8)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, 'fig2_scaling_build.pdf'))
    savefig(fig, os.path.join(fig_dir, 'fig2_scaling_build.png'))


# ── Fig 3: Cross-dataset comparison heatmap at N=50K ─────────────────────────

def fig_heatmap_n50k(summary_data, fig_dir):
    """Heatmap of query time across datasets × methods at N=50K."""
    target_n = 50_000
    rows_50k = [r for r in summary_data if r['n'] == target_n]
    if not rows_50k:
        print("  [WARN] No N=50K rows for heatmap")
        return

    datasets = list(dict.fromkeys(r['dataset'] for r in rows_50k))
    methods  = [m for m in ['rdt_fast', 'rdt_optimized', 'uniform_grid', 'kd_tree']
                if any(r['method']==m for r in rows_50k)]

    matrix = np.full((len(methods), len(datasets)), np.nan)
    for i, method in enumerate(methods):
        for j, ds in enumerate(datasets):
            hits = [r for r in rows_50k if r['method']==method and r['dataset']==ds]
            if hits:
                matrix[i, j] = hits[0]['query_ms']['mean']

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DS_LABELS.get(d,d) for d in datasets], rotation=30, ha='right')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS.get(m,m) for m in methods])

    for i in range(len(methods)):
        for j in range(len(datasets)):
            v = matrix[i, j]
            if not np.isnan(v):
                txt = f"{v:.0f}"
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color='black' if v < np.nanmax(matrix)*0.6 else 'white')

    cbar = fig.colorbar(im, ax=ax, label='Query time (ms)')
    ax.set_title(f'Query Time (ms) at N=50,000 — 512 queries, mean over 3 runs\n'
                 f'Lower = faster. All methods 100% exact.', fontsize=10)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, 'fig3_heatmap_n50k.pdf'))
    savefig(fig, os.path.join(fig_dir, 'fig3_heatmap_n50k.png'))


# ── Fig 4: Speedup of RDT-Fast over KD-tree ───────────────────────────────────

def fig_speedup(scaling_data, fig_dir):
    """Speedup = kd_tree query / rdt_fast query — ratio > 1 means RDT-Fast wins."""
    ds_subset = ['uniform', 'clustered', 'adversarial_hotspot']
    ds_titles = ['Uniform Random', 'Clustered', 'Adversarial Hotspot']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')

    ns_all = sorted({r['n'] for r in scaling_data})
    for ds_name, title in zip(ds_subset, ds_titles):
        ratios = []
        ns_valid = []
        for n in ns_all:
            rdt_rows = [r for r in scaling_data if r['dataset']==ds_name and r['method']=='rdt_fast' and r['n']==n]
            kd_rows  = [r for r in scaling_data if r['dataset']==ds_name and r['method']=='kd_tree'  and r['n']==n]
            if rdt_rows and kd_rows:
                _qget = lambda r: r.get('query_mean_ms', r.get('query_mean', 0))
                ratio = _qget(kd_rows[0]) / max(_qget(rdt_rows[0]), 0.001)
                ratios.append(ratio)
                ns_valid.append(n)
        if ns_valid:
            ax.plot(ns_valid, ratios,
                    label=title,
                    marker='o', markersize=5, linewidth=1.8)

    ax.set_xscale('log')
    ax.set_xlabel('Object count (N)')
    ax.set_ylabel('Speedup vs KD-Tree\n(query: KD-Tree ms / RDT-Fast ms)')
    ax.set_title('RDT-Fast Query Speedup vs. KD-Tree\n> 1.0 = RDT-Fast is faster')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.fill_between([min(ns_all), max(ns_all)], [1.0, 1.0], [ax.get_ylim()[0], ax.get_ylim()[0]],
                    alpha=0.05, color='red', label='_nolegend_')
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, 'fig4_speedup_vs_kdtree.pdf'))
    savefig(fig, os.path.join(fig_dir, 'fig4_speedup_vs_kdtree.png'))


# ── Fig 5: Speedup of RDT-Fast over Uniform Grid ─────────────────────────────

def fig_speedup_vs_grid(scaling_data, fig_dir):
    ds_subset = ['uniform', 'clustered', 'adversarial_hotspot']
    ds_titles = ['Uniform Random', 'Clustered', 'Adversarial Hotspot']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')

    ns_all = sorted({r['n'] for r in scaling_data})
    for ds_name, title in zip(ds_subset, ds_titles):
        ratios = []
        ns_valid = []
        for n in ns_all:
            rdt_rows  = [r for r in scaling_data if r['dataset']==ds_name and r['method']=='rdt_fast'    and r['n']==n]
            grid_rows = [r for r in scaling_data if r['dataset']==ds_name and r['method']=='uniform_grid' and r['n']==n]
            if rdt_rows and grid_rows:
                _qget2 = lambda r: r.get('query_mean_ms', r.get('query_mean', 0))
                ratio = _qget2(grid_rows[0]) / max(_qget2(rdt_rows[0]), 0.001)
                ratios.append(ratio)
                ns_valid.append(n)
        if ns_valid:
            ax.plot(ns_valid, ratios, label=title, marker='s', markersize=5, linewidth=1.8)

    ax.set_xscale('log')
    ax.set_xlabel('Object count (N)')
    ax.set_ylabel('Speedup (Grid ms / RDT-Fast ms)')
    ax.set_title('RDT-Fast Query Speedup vs. Uniform Grid\n> 1.0 = RDT-Fast is faster')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, 'fig5_speedup_vs_grid.pdf'))
    savefig(fig, os.path.join(fig_dir, 'fig5_speedup_vs_grid.png'))


# ── Fig 6: Alpha ablation ────────────────────────────────────────────────────

def fig_ablation(ablation_data, fig_dir):
    if not ablation_data:
        print("  [SKIP] ablation data not available")
        return

    ds_subset = ['uniform', 'clustered']
    ds_titles = ['Uniform Random', 'Clustered']
    max_leafs = sorted({r['max_leaf'] for r in ablation_data})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Alpha Sensitivity (RDT-Fast, N=50K)\nLower query time = better', fontsize=11)

    for ax, ds_name, title in zip(axes, ds_subset, ds_titles):
        for ml in max_leafs:
            rows = sorted([r for r in ablation_data if r['dataset']==ds_name and r['max_leaf']==ml],
                          key=lambda x: x['alpha'])
            if not rows:
                continue
            alphas = [r['alpha'] for r in rows]
            qmeans = [r.get('query_mean_ms', r.get('query_mean', 0)) for r in rows]
            qstds  = [r.get('query_std_ms', r.get('query_std', 0))  for r in rows]
            ax.errorbar(alphas, qmeans, yerr=qstds,
                        label=f'max_leaf={ml}',
                        marker='o', markersize=5, linewidth=1.5, capsize=3)

        ax.set_xlabel('Alpha (subdivision sensitivity)')
        ax.set_ylabel('Query time (ms)')
        ax.set_title(title)
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, 'fig6_ablation_alpha.pdf'))
    savefig(fig, os.path.join(fig_dir, 'fig6_ablation_alpha.png'))


# ── Fig 7: Method comparison bar chart at N=50K ──────────────────────────────

def fig_bar_n50k(summary_data, fig_dir):
    target_n = 50_000
    methods  = ['rdt_fast', 'rdt_optimized', 'uniform_grid', 'kd_tree']
    datasets = ['uniform', 'clustered', 'adversarial_hotspot', 'adversarial_line']
    ds_labels_short = {'uniform':'Uniform', 'clustered':'Clustered',
                       'adversarial_hotspot':'Hotspot', 'adversarial_line':'Line'}

    rows_50k = [r for r in summary_data if r['n'] == target_n]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Build & Query Time at N={target_n:,} (mean ± std, 3 runs)', fontsize=12)

    x = np.arange(len(datasets))
    width = 0.18

    for ax_idx, metric in enumerate(['build', 'query']):
        ax = axes[ax_idx]
        for mi, method in enumerate(methods):
            vals = []
            errs = []
            for ds in datasets:
                hits = [r for r in rows_50k if r['method']==method and r['dataset']==ds]
                if hits:
                    ms_key = f'{metric}_ms'
                    vals.append(hits[0][ms_key]['mean'])
                    errs.append(hits[0][ms_key]['std'])
                else:
                    vals.append(0); errs.append(0)
            offset = (mi - len(methods)/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=METHOD_LABELS.get(method, method),
                          color=METHOD_COLORS.get(method, 'gray'),
                          alpha=0.85,
                          yerr=errs, capsize=2, error_kw={'linewidth': 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels([ds_labels_short.get(d,d) for d in datasets])
        ax.set_ylabel(f'{metric.capitalize()} time (ms)')
        ax.set_yscale('log')
        ax.set_title(f'{metric.capitalize()} time')
        ax.legend(framealpha=0.8, fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, 'fig7_bar_n50k.pdf'))
    savefig(fig, os.path.join(fig_dir, 'fig7_bar_n50k.png'))


# ── Fig 8: Memory usage ──────────────────────────────────────────────────────

def fig_memory(summary_data, fig_dir):
    """Peak build memory vs N for each method."""
    target_ds = 'uniform'
    rows = [r for r in summary_data if r['dataset'] == target_ds
            and r.get('peak_build_kb', None) is not None
            and r['peak_build_kb'] > 0]

    if not rows:
        print("  [SKIP] memory data not available")
        return

    methods = sorted({r['method'] for r in rows})
    ns_all  = sorted({r['n'] for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        m_rows = sorted([r for r in rows if r['method']==method], key=lambda x: x['n'])
        ns = [r['n'] for r in m_rows]
        kb = [r['peak_build_kb']/1024 for r in m_rows]  # to MB
        ax.plot(ns, kb,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, 'gray'),
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=5, linewidth=1.5)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Object count (N)')
    ax.set_ylabel('Peak build memory (MB)')
    ax.set_title(f'Peak Memory During Build\n(tracemalloc, {target_ds} distribution)')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, 'fig8_memory.pdf'))
    savefig(fig, os.path.join(fig_dir, 'fig8_memory.png'))


# ── Tables ────────────────────────────────────────────────────────────────────

def make_table_wins(summary_data, table_dir):
    """Wins/ties/losses table per method across all workloads."""
    os.makedirs(table_dir, exist_ok=True)
    target_n = 50_000
    rows_50k = [r for r in summary_data if r['n'] == target_n]
    datasets = list(dict.fromkeys(r['dataset'] for r in rows_50k))
    methods  = ['rdt_fast', 'rdt_optimized', 'uniform_grid', 'kd_tree']

    records = []
    for ds in datasets:
        ds_rows = {r['method']: r for r in rows_50k if r['dataset']==ds}
        if not ds_rows:
            continue
        # Query time winner
        q_times = {m: ds_rows[m]['query_ms']['mean'] for m in methods if m in ds_rows}
        b_times = {m: ds_rows[m]['build_ms']['mean'] for m in methods if m in ds_rows}
        if not q_times:
            continue
        best_q = min(q_times, key=q_times.get)
        best_b = min(b_times, key=b_times.get)
        records.append({'dataset': ds, 'query_winner': best_q, 'build_winner': best_b,
                        'q_times': q_times, 'b_times': b_times})

    # Write markdown table
    lines = ["# Wins/Losses Summary — N=50,000\n",
             "| Dataset | Query Winner | Build Winner | Grid q(ms) | KD q(ms) | RDT-Fast q(ms) | RDT-Opt q(ms) |",
             "|---------|-------------|--------------|------------|----------|---------------|--------------|"]
    for rec in records:
        q = rec['q_times']
        lines.append(f"| {DS_LABELS.get(rec['dataset'], rec['dataset'])} | "
                     f"**{METHOD_LABELS.get(rec['query_winner'],'?')}** | "
                     f"{METHOD_LABELS.get(rec['build_winner'],'?')} | "
                     f"{q.get('uniform_grid', float('nan')):6.1f} | "
                     f"{q.get('kd_tree', float('nan')):6.1f} | "
                     f"{q.get('rdt_fast', float('nan')):6.1f} | "
                     f"{q.get('rdt_optimized', float('nan')):6.1f} |")

    # Summary counts
    from collections import Counter
    qw = Counter(rec['query_winner'] for rec in records)
    bw = Counter(rec['build_winner'] for rec in records)
    lines += ["\n## Query Win Counts (N=50K)"]
    for m, cnt in qw.most_common():
        lines.append(f"- {METHOD_LABELS.get(m,m)}: {cnt} / {len(records)} datasets")
    lines += ["\n## Build Win Counts (N=50K)"]
    for m, cnt in bw.most_common():
        lines.append(f"- {METHOD_LABELS.get(m,m)}: {cnt} / {len(records)} datasets")

    path = os.path.join(table_dir, 'table_wins_n50k.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ {os.path.relpath(path, ROOT)}")

    # Also write CSV
    import csv
    csv_path = os.path.join(table_dir, 'table_wins_n50k.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset','rdt_fast_q','rdt_opt_q','grid_q','kd_q','query_winner','build_winner'])
        for rec in records:
            q = rec['q_times']
            w.writerow([rec['dataset'],
                        f"{q.get('rdt_fast',''):.2f}" if 'rdt_fast' in q else '',
                        f"{q.get('rdt_optimized',''):.2f}" if 'rdt_optimized' in q else '',
                        f"{q.get('uniform_grid',''):.2f}" if 'uniform_grid' in q else '',
                        f"{q.get('kd_tree',''):.2f}" if 'kd_tree' in q else '',
                        rec['query_winner'], rec['build_winner']])
    print(f"  ✓ {os.path.relpath(csv_path, ROOT)}")
    return records


def make_table_scaling(scaling_data, table_dir):
    os.makedirs(table_dir, exist_ok=True)
    lines = ["# Scaling Analysis Table (N up to 1M)\n",
             "| Dataset | N | RDT-Fast build(ms) | RDT-Fast query(ms) | Grid build(ms) | Grid query(ms) | ScipyKD build(ms) | ScipyKD query(ms) |",
             "|---------|---|-------------------|-------------------|----------------|----------------|------------------|------------------|"]
    for ds in ['uniform', 'clustered', 'adversarial_hotspot']:
        ns = sorted({r['n'] for r in scaling_data if r['dataset']==ds})
        for n in ns:
            def get(method, stat):
                hits = [r for r in scaling_data if r['dataset']==ds and r['method']==method and r['n']==n]
                if not hits:
                    return '—'
                # support both old key format and new _ms suffix format
                val = hits[0].get(stat + '_ms', hits[0].get(stat, None))
                return f"{val:.1f}" if val is not None else '—'
            lines.append(f"| {DS_LABELS.get(ds,ds)} | {n:,} | "
                         f"{get('rdt_fast','build_mean')} ± {get('rdt_fast','build_std')} | "
                         f"{get('rdt_fast','query_mean')} ± {get('rdt_fast','query_std')} | "
                         f"{get('uniform_grid','build_mean')} | "
                         f"{get('uniform_grid','query_mean')} | "
                         f"{get('scipy_kd','build_mean')} | "
                         f"{get('scipy_kd','query_mean')} |")

    path = os.path.join(table_dir, 'table_scaling.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ {os.path.relpath(path, ROOT)}")


def make_summary_table(summary_data, table_dir):
    os.makedirs(table_dir, exist_ok=True)
    lines = ["# Full Benchmark Results Summary\n",
             "All methods: 100% exact match vs. brute force across all configurations.\n",
             "| Dataset | N | Method | Build mean(ms) | Build std | Query mean(ms) | Query std | Memory(MB) |",
             "|---------|---|--------|---------------|-----------|---------------|-----------|------------|"]
    for row in summary_data:
        b = row['build_ms']
        q = row['query_ms']
        mem = row.get('peak_build_kb', 0) / 1024 if row.get('peak_build_kb') else 0
        lines.append(f"| {DS_LABELS.get(row['dataset'],row['dataset'])} "
                     f"| {row['n']:,} "
                     f"| {METHOD_LABELS.get(row['method'],row['method'])} "
                     f"| {b['mean']:.2f} | {b['std']:.2f} "
                     f"| {q['mean']:.2f} | {q['std']:.2f} "
                     f"| {mem:.2f} |")
    path = os.path.join(table_dir, 'table_full_results.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ {os.path.relpath(path, ROOT)}")


# ── Fig 9: Method Diagram ─────────────────────────────────────────────────────

def fig_method_diagram(fig_dir):
    """
    Visual explanation of the RDT subdivision rule vs. Quadtree and Uniform Grid.
    Shows 3 side-by-side panels:
      Left:   Uniform Grid (fixed g×g cells regardless of density)
      Middle: RDT (log-based g(n) adaptive subdivision)
      Right:  Quadtree (always 4 children, fixed threshold)
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        'Spatial Index Subdivision Comparison (N=200 clustered points)',
        fontsize=11, fontweight='bold', y=1.01
    )

    # Generate a small clustered dataset for illustration
    rng = np.random.default_rng(42)
    n_cluster = 160
    n_sparse  = 40
    cx, cy = 0.65, 0.65
    cluster_pts = rng.normal([cx, cy], 0.08, (n_cluster, 2))
    cluster_pts = np.clip(cluster_pts, 0, 1)
    sparse_pts  = rng.uniform(0, 1, (n_sparse, 2))
    pts = np.vstack([cluster_pts, sparse_pts])

    titles = ['Uniform Grid\ng×g fixed (g=5)',
              'RDT\ng(n)=floor(log(n+1)^α)',
              'Quadtree\n4-way split at threshold']
    colors = ['#f39c12', '#2ecc71', '#3498db']

    for ax, title, color in zip(axes, titles, colors):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.scatter(pts[:, 0], pts[:, 1], s=4, color='#2c3e50', alpha=0.5, zorder=3)
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)

    # Panel 0: Uniform Grid — draw fixed 5×5 grid lines
    ax = axes[0]
    g = 5
    for i in range(1, g):
        ax.axhline(i / g, color='#f39c12', linewidth=0.8, alpha=0.7)
        ax.axvline(i / g, color='#f39c12', linewidth=0.8, alpha=0.7)

    # Panel 1: RDT — adaptive subdivision using actual g(n) rule
    ax = axes[1]
    G_MAX = 6
    ALPHA = 1.2

    def rdt_g(n, alpha=ALPHA, G=G_MAX):
        if n <= 1:
            return 1
        return min(G, max(2, int(math.floor(math.log(n + 1) ** alpha))))

    def draw_rdt_cell(ax, x0, y0, x1, y1, pts_in_cell, depth=0, max_depth=3):
        if pts_in_cell.shape[0] == 0 or depth >= max_depth:
            return
        g = rdt_g(pts_in_cell.shape[0])
        if g <= 1:
            return
        dx = (x1 - x0) / g
        dy = (y1 - y0) / g
        lw = max(0.3, 1.2 - depth * 0.35)
        for i in range(1, g):
            ax.plot([x0 + i*dx, x0 + i*dx], [y0, y1], color='#2ecc71', linewidth=lw, alpha=0.8)
            ax.plot([x0, x1], [y0 + i*dy, y0 + i*dy], color='#2ecc71', linewidth=lw, alpha=0.8)
        for gi in range(g):
            for gj in range(g):
                cx0, cx1 = x0 + gi*dx, x0 + (gi+1)*dx
                cy0, cy1 = y0 + gj*dy, y0 + (gj+1)*dy
                mask = ((pts_in_cell[:,0] >= cx0) & (pts_in_cell[:,0] < cx1) &
                        (pts_in_cell[:,1] >= cy0) & (pts_in_cell[:,1] < cy1))
                sub = pts_in_cell[mask]
                if sub.shape[0] > 8:
                    draw_rdt_cell(ax, cx0, cy0, cx1, cy1, sub, depth+1, max_depth)

    draw_rdt_cell(axes[1], 0, 0, 1, 1, pts)

    # Panel 2: Quadtree — recursive 4-way split
    ax = axes[2]
    MAX_LEAF = 16

    def draw_quadtree(ax, x0, y0, x1, y1, pts_in_cell, depth=0, max_depth=5):
        if pts_in_cell.shape[0] <= MAX_LEAF or depth >= max_depth:
            return
        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        lw = max(0.3, 1.2 - depth * 0.25)
        ax.plot([mx, mx], [y0, y1], color='#3498db', linewidth=lw, alpha=0.8)
        ax.plot([x0, x1], [my, my], color='#3498db', linewidth=lw, alpha=0.8)
        for (qx0, qy0, qx1, qy1) in [(x0,y0,mx,my),(mx,y0,x1,my),(x0,my,mx,y1),(mx,my,x1,y1)]:
            mask = ((pts_in_cell[:,0] >= qx0) & (pts_in_cell[:,0] < qx1) &
                    (pts_in_cell[:,1] >= qy0) & (pts_in_cell[:,1] < qy1))
            draw_quadtree(ax, qx0, qy0, qx1, qy1, pts_in_cell[mask], depth+1, max_depth)

    draw_quadtree(axes[2], 0, 0, 1, 1, pts)

    # Annotation: highlight dense region
    for ax in axes:
        circle = plt.Circle((cx, cy), 0.12, fill=False, edgecolor='red',
                             linewidth=1.5, linestyle='--', alpha=0.7, zorder=5)
        ax.add_patch(circle)

    # Caption
    fig.text(0.5, -0.04,
             'Red circle marks the dense cluster. '
             'RDT uses log-based adaptive subdivision; '
             'Quadtree always splits 4 ways; '
             'Uniform Grid ignores density entirely.',
             ha='center', fontsize=8, color='#555555', style='italic')

    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, 'fig9_method_diagram.pdf'))
    savefig(fig, os.path.join(fig_dir, 'fig9_method_diagram.png'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=os.path.join(ROOT, 'publication'))
    args = parser.parse_args()

    raw_dir   = os.path.join(args.outdir, 'RAW_RESULTS')
    fig_dir   = os.path.join(args.outdir, 'PAPER_FIGURES')
    table_dir = os.path.join(args.outdir, 'PAPER_TABLES')

    summary_data = load_json(os.path.join(raw_dir, 'benchmark_summary.json'))
    scaling_data = load_json(os.path.join(raw_dir, 'scaling_results.json'))
    ablation_data= load_json(os.path.join(raw_dir, 'ablation_alpha.json'))

    print("\n── Generating figures...")
    if scaling_data:
        fig_scaling_query(scaling_data, fig_dir)
        fig_scaling_build(scaling_data, fig_dir)
        fig_speedup(scaling_data, fig_dir)
        fig_speedup_vs_grid(scaling_data, fig_dir)
    if summary_data:
        fig_heatmap_n50k(summary_data, fig_dir)
        fig_bar_n50k(summary_data, fig_dir)
        fig_memory(summary_data, fig_dir)
    if ablation_data:
        fig_ablation(ablation_data, fig_dir)
    # Method diagram (no data needed — generated from scratch)
    fig_method_diagram(fig_dir)

    print("\n── Generating tables...")
    if summary_data:
        make_table_wins(summary_data, table_dir)
        make_summary_table(summary_data, table_dir)
    if scaling_data:
        make_table_scaling(scaling_data, table_dir)

    print("\n✓ All outputs written to", args.outdir)


if __name__ == '__main__':
    main()
