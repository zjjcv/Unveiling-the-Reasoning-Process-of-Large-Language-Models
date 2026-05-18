"""
Energy Distribution Plot: three proxy types (al, ml, al+ml).

Energy per layer: E(l) = (||x||₂ / ||h_l||₂) × (1 − cos(x, h_l))
  where x ∈ {al, ml, al+ml}

Inputs (all per question × layer):
  - gsm8k_{al,ml,al_plus_ml}.csv   L2 norms (3_proxy_collection.py)
  - gsm8k_hl.csv                   hidden state L2 (computed from ratio_al)
  - cos_{al,ml,al_plus_ml}.csv     cosine with h_l (layer_proxy_collection.py)
  - {al,ml,al_plus_ml}_syn_red_pairwise.csv  (compute_al_syn_red_pairwise_mp.py)

Outputs:
  - energy/energy_{al,ml,al_plus_ml}.csv
  - energy/energy_distribution.png

Usage:
    python utils/plot/energy_distribution_plot.py qwen3_8b_base
    python utils/plot/energy_distribution_plot.py gemma3_4b_base
    python utils/plot/energy_distribution_plot.py gemma3_12b_base
    python utils/plot/energy_distribution_plot.py gemma3_4b_it
    python utils/plot/energy_distribution_plot.py               # Gemma3-12B-IT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Global style ───────────────────────────────────────────────────────────
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Noto Serif', 'DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'font.size': 22,
    'font.weight': 'bold',
    'axes.labelsize': 28,
    'axes.labelweight': 'bold',
    'axes.titlesize': 30,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# ── Colour palette (Nature-style) ─────────────────────────────────────────
_COLOR_AL = '#E64B35'
_COLOR_ML = '#4DBBD5'
_COLOR_AL_ML = '#00A087'

# ── Configuration ──────────────────────────────────────────────────────────
import sys

if len(sys.argv) > 1 and sys.argv[1] == 'qwen3_8b_base':
    L2_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/L2_Norm"
    RS_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/residual_stream"
    ENERGY_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/energy"
    FIG_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/Figure"
    NUM_LAYERS = 36
    MODEL_LABEL = "Qwen3-8B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    L2_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/L2_Norm"
    RS_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/residual_stream"
    ENERGY_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/energy"
    FIG_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure"
    NUM_LAYERS = 34
    MODEL_LABEL = "Gemma3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b_base':
    L2_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/data/L2_Norm"
    RS_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/data/residual_stream"
    ENERGY_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/data/energy"
    FIG_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/Figure"
    NUM_LAYERS = 48
    MODEL_LABEL = "Gemma3-12B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_it':
    L2_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/L2_Norm"
    RS_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/residual_stream"
    ENERGY_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/energy"
    FIG_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/Figure"
    NUM_LAYERS = 34
    MODEL_LABEL = "Gemma-3-4B-Instruct"
else:
    L2_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/L2_Norm"
    RS_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/residual_stream"
    ENERGY_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Instruct/data/energy"
    FIG_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure"
    NUM_LAYERS = 48
    MODEL_LABEL = "Gemma3-12B-IT"


# ── Data loading ───────────────────────────────────────────────────────────

def load_layer_l2_mean(csv_path, effective_lengths_path):
    """Layer-level L2 CSV → per-layer mean across steps and questions."""
    df = pd.read_csv(csv_path)
    step_cols = [c for c in df.columns if c.startswith('step_')]

    if os.path.exists(effective_lengths_path):
        eff_df = pd.read_csv(effective_lengths_path)
        eff_dict = dict(zip(eff_df['question_id'], eff_df['effective_length']))
    else:
        eff_dict = {qid: len(step_cols) for qid in df['question_id'].unique()}

    records = []
    for _, row in df.iterrows():
        qid = int(row['question_id'])
        layer = int(row['layer'])
        eff_len = min(eff_dict.get(qid, len(step_cols)), len(step_cols))
        mean_l2 = row[step_cols[:eff_len]].values.astype(float).mean()
        records.append({'question_id': qid, 'layer': layer, 'mean_l2': mean_l2})

    return pd.DataFrame(records).groupby('layer')['mean_l2'].mean().reset_index()


def load_al_layer_l2(csv_path, effective_lengths_path):
    """Per-head L2 CSV → aggregate to layer-level ||al|| = sqrt(sum_h ||a_h||²)."""
    df = pd.read_csv(csv_path)
    step_cols = [c for c in df.columns if c.startswith('step_')]

    if os.path.exists(effective_lengths_path):
        eff_df = pd.read_csv(effective_lengths_path)
        eff_dict = dict(zip(eff_df['question_id'], eff_df['effective_length']))
    else:
        eff_dict = {qid: len(step_cols) for qid in df['question_id'].unique()}

    records = []
    for _, row in df.iterrows():
        qid = int(row['question_id'])
        layer = int(row['layer'])
        eff_len = min(eff_dict.get(qid, len(step_cols)), len(step_cols))
        mean_l2 = row[step_cols[:eff_len]].values.astype(float).mean()
        records.append({'question_id': qid, 'layer': layer, 'head': int(row['head']), 'mean_l2': mean_l2})

    per_q = pd.DataFrame(records)
    # For each (question, layer): ||al|| = sqrt(sum_h ||a_h||²)
    per_q['l2_sq'] = per_q['mean_l2'] ** 2
    layer_sq = per_q.groupby(['question_id', 'layer'])['l2_sq'].sum().reset_index()
    layer_sq['al_l2'] = np.sqrt(layer_sq['l2_sq'])
    # Average across questions
    return layer_sq.groupby('layer')['al_l2'].mean().reset_index().rename(columns={'al_l2': 'mean_l2'})


def load_cosine(csv_path):
    """cos(x, h_l) scalar CSV → per-layer mean."""
    df = pd.read_csv(csv_path)
    avg = df.groupby('layer')['value'].mean().reset_index()
    return dict(zip(avg['layer'].astype(int), avg['value']))


def load_hl_mean(csv_path, effective_lengths_path):
    """gsm8k_hl.csv → per-layer mean ||h_l||."""
    return load_layer_l2_mean(csv_path, effective_lengths_path)


def load_syn_red_layer(pairwise_path, num_layers, is_per_head=False):
    """Pairwise CSV → per-layer (Syn - Red)."""
    df = pd.read_csv(pairwise_path)

    if is_per_head:
        # Per-head pairwise: aggregate heads to layer level first
        df_avg = df.groupby(['layer_1', 'head_1', 'layer_2', 'head_2']).agg(
            syn=('syn', 'mean'), red=('red', 'mean')).reset_index()
        # Then aggregate to layer level
        layer_syn, layer_red, layer_cnt = {}, {}, {}
        for _, row in df_avg.iterrows():
            for l in [int(row['layer_1']), int(row['layer_2'])]:
                layer_syn[l] = layer_syn.get(l, 0.0) + row['syn']
                layer_red[l] = layer_red.get(l, 0.0) + row['red']
                layer_cnt[l] = layer_cnt.get(l, 0) + 1
    else:
        df_avg = df.groupby(['layer_1', 'layer_2']).agg(
            syn=('syn', 'mean'), red=('red', 'mean')).reset_index()
        layer_syn, layer_red, layer_cnt = {}, {}, {}
        for _, row in df_avg.iterrows():
            for l in [int(row['layer_1']), int(row['layer_2'])]:
                layer_syn[l] = layer_syn.get(l, 0.0) + row['syn']
                layer_red[l] = layer_red.get(l, 0.0) + row['red']
                layer_cnt[l] = layer_cnt.get(l, 0) + 1

    syn_red = np.full(num_layers, np.nan)
    for l, c in layer_cnt.items():
        if l < num_layers and c > 0:
            syn_red[l] = layer_syn[l] / c - layer_red[l] / c
    return syn_red


def compute_energy(l2_vals, hl_vals, cos_vals, num_layers):
    """E(l) = (||x|| / ||h_l||) × (1 - cos(x, h_l))."""
    l2_dict = dict(zip(l2_vals['layer'].astype(int), l2_vals['mean_l2']))
    hl_dict = dict(zip(hl_vals['layer'].astype(int), hl_vals['mean_l2']))

    energy = np.full(num_layers, np.nan)
    for l in range(num_layers):
        num = l2_dict.get(l)
        den = hl_dict.get(l)
        cos = cos_vals.get(l)
        if num is not None and den is not None and den > 1e-10 and cos is not None:
            energy[l] = (num / den) * (1.0 - cos)
    return energy


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_energy_bars(ax, energies, labels, colors, num_layers):
    """Left: overlapping bar chart for all three proxy types."""
    x = np.arange(num_layers)
    # Draw largest first so smaller overlaps on top
    order = sorted(range(3), key=lambda i: -np.nansum(energies[i]))
    drawn = set()
    for i in order:
        lbl = labels[i] if labels[i] not in drawn else None
        ax.bar(x, energies[i], width=0.88, bottom=0, color=colors[i],
               label=lbl, edgecolor='white', linewidth=0.4, alpha=0.92)
        if lbl:
            drawn.add(labels[i])

    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Energy')
    ax.set_xlim(-0.6, num_layers - 0.4)

    step = 2 if num_layers <= 40 else 4
    ax.set_xticks(range(0, num_layers, step))

    ax.grid(True, axis='y', linestyle='-', alpha=0.10, linewidth=0.5, color='#999999')
    ax.tick_params(axis='both', length=4, width=0.6, colors='#333333', direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    leg = ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
                    framealpha=0.95, loc='upper center', ncol=3,
                    bbox_to_anchor=(0.5, 0.97), borderaxespad=0,
                    handletextpad=0.5, columnspacing=1.8, borderpad=0.6,
                    handlelength=1.8, handleheight=1.0)
    leg.get_frame().set_linewidth(0.6)
    for t in leg.get_texts():
        t.set_fontweight('bold')


def plot_scatter(ax, energy_ranks, syn_red_ranks, labels, colors):
    """Right: scatter for all three proxy types with Pearson r in legend."""
    for i in range(3):
        e = energy_ranks[i]
        s = syn_red_ranks[i]
        mask = ~np.isnan(e) & ~np.isnan(s)
        x, y = s[mask], e[mask]

        r, _ = stats.pearsonr(x, y)
        print(f"  {labels[i]}: Pearson r = {r:.4f}, n = {len(x)}")

        ax.scatter(x, y, c=colors[i], s=48, alpha=0.70, edgecolors='none',
                   label=f'{labels[i]}  $r$={r:.3f}', zorder=3)

        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), color=colors[i], linewidth=1.8,
                linestyle='--', alpha=0.6, zorder=2)

    ax.set_xlabel(r'(Syn $-$ Red) Rank')
    ax.set_ylabel(r'Energy Rank')
    ax.grid(True, linestyle='-', alpha=0.10, linewidth=0.5, color='#999999')
    ax.tick_params(axis='both', length=4, width=0.6, colors='#333333', direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    leg = ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
                    framealpha=0.95, loc='upper left', borderpad=0.6,
                    handlelength=1.5, handleheight=1.0)
    leg.get_frame().set_linewidth(0.6)
    for t in leg.get_texts():
        t.set_fontweight('bold')


# ── Main ───────────────────────────────────────────────────────────────────

PROXY_CONFIGS = [
    {
        'key': 'al_plus_ml',
        'l2_file': 'gsm8k_al_plus_ml.csv',
        'cos_file': 'cos_al_plus_ml.csv',
        'pairwise_file': 'al_plus_ml_syn_red_pairwise.csv',
        'label': r'$\mathrm{AL+ML}$',
        'color': _COLOR_AL_ML,
        'per_head_l2': False,
        'per_head_pairwise': False,
    },
    {
        'key': 'ml',
        'l2_file': 'gsm8k_ml.csv',
        'cos_file': 'cos_ml.csv',
        'pairwise_file': 'ml_syn_red_pairwise.csv',
        'label': r'$\mathrm{ML}$',
        'color': _COLOR_ML,
        'per_head_l2': False,
        'per_head_pairwise': False,
    },
    {
        'key': 'al',
        'l2_file': 'gsm8k_al.csv',
        'cos_file': 'cos_al.csv',
        'pairwise_file': 'al_syn_red_pairwise.csv',
        'label': r'$\mathrm{AL}$',
        'color': _COLOR_AL,
        'per_head_l2': True,
        'per_head_pairwise': True,
    },
]


def main():

    print("=" * 60)
    print(f"Energy Distribution Plot ({MODEL_LABEL})")
    print("=" * 60)

    eff_csv = os.path.join(L2_DIR, "gsm8k_effective_lengths.csv")
    hl_csv = os.path.join(L2_DIR, "gsm8k_hl.csv")

    # Load ||h_l|| (shared denominator)
    print("\nLoading ||h_l||...")
    hl = load_hl_mean(hl_csv, eff_csv)
    print(f"  {len(hl)} layers")

    energies, energy_ranks, syn_red_ranks = [], [], []

    for cfg in PROXY_CONFIGS:
        key = cfg['key']
        print(f"\n{'─'*60}")
        print(f"Proxy: {key}")
        print(f"{'─'*60}")

        l2_path = os.path.join(L2_DIR, cfg['l2_file'])
        cos_path = os.path.join(RS_DIR, cfg['cos_file'])
        pw_path = os.path.join(L2_DIR, cfg['pairwise_file'])

        for f in [l2_path, cos_path, pw_path]:
            if not os.path.exists(f):
                print(f"  Error: {f} not found")
                energies.append(np.full(NUM_LAYERS, np.nan))
                energy_ranks.append(np.full(NUM_LAYERS, np.nan))
                syn_red_ranks.append(np.full(NUM_LAYERS, np.nan))
                break
        else:
            # L2 norms
            if cfg['per_head_l2']:
                l2 = load_al_layer_l2(l2_path, eff_csv)
            else:
                l2 = load_layer_l2_mean(l2_path, eff_csv)

            # Cosine
            cos = load_cosine(cos_path)

            # Energy
            e = compute_energy(l2, hl, cos, NUM_LAYERS)
            valid = e[~np.isnan(e)]
            print(f"  Energy: [{valid.min():.4f}, {valid.max():.4f}]")

            # Syn-Red diff
            sr = load_syn_red_layer(pw_path, NUM_LAYERS,
                                     is_per_head=cfg['per_head_pairwise'])

            # Rank
            e_rank = pd.Series(e).rank(method='dense').values
            sr_rank = pd.Series(sr).rank(method='dense').values
            e_rank[np.isnan(e)] = np.nan
            sr_rank[np.isnan(sr)] = np.nan

            energies.append(e)
            energy_ranks.append(e_rank)
            syn_red_ranks.append(sr_rank)

            # Save intermediate CSV
            os.makedirs(ENERGY_DIR, exist_ok=True)
            records = []
            l2_dict = dict(zip(l2['layer'].astype(int), l2['mean_l2']))
            hl_dict = dict(zip(hl['layer'].astype(int), hl['mean_l2']))
            for l in range(NUM_LAYERS):
                records.append({
                    'layer': l,
                    'x_l2': l2_dict.get(l, np.nan),
                    'hl_l2': hl_dict.get(l, np.nan),
                    'cos': cos.get(l, np.nan),
                    'energy': e[l],
                    'energy_rank': e_rank[l],
                    'syn_red_diff': sr[l],
                    'syn_red_rank': sr_rank[l],
                })
            csv_path = os.path.join(ENERGY_DIR, f"energy_{key}.csv")
            pd.DataFrame(records).to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

    # Plot
    print(f"\n{'='*60}")
    print("Plotting...")
    print(f"{'='*60}")

    labels = [c['label'] for c in PROXY_CONFIGS]
    colors = [c['color'] for c in PROXY_CONFIGS]

    fig_w = max(16, NUM_LAYERS * 0.52 + 10)
    fig_h = 8.0

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={'width_ratios': [NUM_LAYERS * 0.52, 10], 'wspace': 0.18},
        constrained_layout=True)
    fig.subplots_adjust(top=0.88)

    plot_energy_bars(ax1, energies, labels, colors, NUM_LAYERS)
    plot_scatter(ax2, energy_ranks, syn_red_ranks, labels, colors)

    os.makedirs(FIG_DIR, exist_ok=True)
    output_path = os.path.join(FIG_DIR, "energy_distribution.png")
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    print(f"\n{'='*60}")
    print("All Done!")
    print(f"{'='*60}")
    print(f"\nIntermediate data: {ENERGY_DIR}/")
    print(f"Figure: {FIG_DIR}/energy_distribution.png")


if __name__ == "__main__":
    main()
