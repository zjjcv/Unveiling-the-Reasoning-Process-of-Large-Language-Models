"""
Plot layer-wise cosine similarities and ratios from scalar CSV data.

Reads pre-computed scalar CSVs from layer_proxy_collection.py and generates
publication-quality stacked bar plots.

Input:  {cos_al,cos_ml,cos_al_plus_ml,ratio_al,ratio_ml,ratio_al_plus_ml}.csv
        (columns: question_id, layer, value)
Output: residual_stream_ratios.png, cosine_similarities.png

Usage:
    python src/residual_stream_plot.py                       # Gemma3-12B-IT
    python src/residual_stream_plot.py gemma3_12b_base       # Gemma3-12B-Base
    python src/residual_stream_plot.py gemma3_4b_base        # Gemma3-4B-Base
    python src/residual_stream_plot.py gemma3_4b_it          # Gemma-3-4B-Instruct
    python src/residual_stream_plot.py qwen3_8b_base         # Qwen3-8B-Base
    python src/residual_stream_plot.py qwen3_4b_base         # Qwen3-4B-Base
    python src/residual_stream_plot.py qwen3_14b_base        # Qwen3-14B-Base
    python src/residual_stream_plot.py llama3_8b             # Llama-3.1-8B
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Noto Serif', 'DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'font.size': 22,
    'font.weight': 'bold',
    'axes.labelsize': 22,
    'axes.labelweight': 'bold',
    'axes.titlesize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# ── Colour palette — RdBu_r consistent ───────────────────────────────────
# ── Colour palette — Nature-style muted, RdBu_r family ────────────────────
_COLOR_AL = '#E64B35'        # Nature red
_COLOR_ML = '#4DBBD5'        # Nature cyan
_COLOR_AL_ML = '#00A087'     # Nature teal

# Configuration
import sys

if len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b_base':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/data/residual_stream"
    PLOT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/Figure"
    NUM_LAYERS = 48
    MODEL_LABEL = "Gemma3-12B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/residual_stream"
    PLOT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure"
    NUM_LAYERS = 34
    MODEL_LABEL = "Gemma3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_it':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/residual_stream"
    PLOT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/Figure"
    NUM_LAYERS = 34
    MODEL_LABEL = "Gemma-3-4B-Instruct"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_8b_base':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/residual_stream"
    PLOT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/Figure"
    NUM_LAYERS = 36
    MODEL_LABEL = "Qwen3-8B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/residual_stream"
    PLOT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/Figure"
    NUM_LAYERS = 36
    MODEL_LABEL = "Qwen3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/residual_stream"
    PLOT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/Figure"
    NUM_LAYERS = 40
    MODEL_LABEL = "Qwen3-14B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/residual_stream"
    PLOT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/Figure"
    NUM_LAYERS = 32
    MODEL_LABEL = "Llama-3.1-8B"
else:
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/residual_stream"
    PLOT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure"
    NUM_LAYERS = 48
    MODEL_LABEL = "Gemma3-12B-IT"


def load_scalar_csv(filepath):
    """Load scalar CSV and return per-layer mean and std."""
    df = pd.read_csv(filepath)
    # Auto-detect value column: 'value', 'cosine', or 'ratio'
    value_col = None
    for col in ['value', 'cosine', 'ratio']:
        if col in df.columns:
            value_col = col
            break
    if value_col is None:
        # Fall back to last numeric column
        value_col = df.select_dtypes(include=[np.number]).columns[-1]
    stats = df.groupby('layer')[value_col].agg(['mean', 'std', 'count']).reset_index()
    stats.columns = ['layer', 'mean_value', 'std_value', 'count']
    return stats


def _overlap_bar(ax, x, vals_list, colors, labels):
    """Draw overlapping bars — largest first, smallest on top."""
    arr = np.column_stack(vals_list)
    n = len(x)
    drawn_labels = set()

    for i in range(n):
        row = arr[i]
        pos_indices = [j for j in range(len(row)) if row[j] >= 0]
        neg_indices = [j for j in range(len(row)) if row[j] < 0]

        pos_indices.sort(key=lambda j: row[j], reverse=True)
        neg_indices.sort(key=lambda j: row[j])

        for j in pos_indices:
            lbl = labels[j] if labels[j] not in drawn_labels else None
            ax.bar(x[i], row[j], width=0.88, bottom=0,
                   color=colors[j], label=lbl,
                   edgecolor='white', linewidth=0.4, alpha=0.92)
            if lbl:
                drawn_labels.add(labels[j])

        for j in neg_indices:
            lbl = labels[j] if labels[j] not in drawn_labels else None
            ax.bar(x[i], row[j], width=0.88, bottom=0,
                   color=colors[j], label=lbl,
                   edgecolor='white', linewidth=0.4, alpha=0.92)
            if lbl:
                drawn_labels.add(labels[j])


def _setup_axes(ax, n_layers, xlabel, legend_labels, legend_colors):
    """Common axis styling."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    ax.set_xlim(-0.8, n_layers - 0.2)

    step = 2 if n_layers <= 40 else 4
    ax.set_xticks(range(0, n_layers, step))

    ax.grid(True, axis='y', linestyle='-', alpha=0.10, linewidth=0.5, color='#999999')
    ax.tick_params(axis='both', length=4, width=0.6, colors='#333333', direction='out')

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')


def _add_legend(fig, legend_labels, legend_colors):
    """Add figure-level legend above the plot, outside axes."""
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec='none', alpha=0.92)
                 for c in legend_colors],
        labels=legend_labels,
        frameon=True, fancybox=False, edgecolor='#cccccc',
        framealpha=0.95, loc='upper center', ncol=3,
        bbox_to_anchor=(0.5, 0.97),
        borderaxespad=0,
        handletextpad=0.5, columnspacing=1.8, borderpad=0.6,
        handlelength=1.8, handleheight=1.0,
    ).get_frame().set_linewidth(0.6)


def plot_ratios(al_stats, ml_stats, al_plus_ml_stats):
    """Plot layer-wise residual stream ratios."""
    print("\nPlotting residual stream ratios...")

    n_layers = len(al_stats)
    fig_width = max(16, n_layers * 0.52)
    fig_height = 9.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.05, right=0.98)

    x = np.arange(n_layers)
    vals = [al_plus_ml_stats['mean_value'].values,
            al_stats['mean_value'].values,
            ml_stats['mean_value'].values]
    colors = [_COLOR_AL_ML, _COLOR_AL, _COLOR_ML]
    labels = [r'$||\mathrm{AL+ML}||^2\, /\, ||h_l||^2$',
              r'$||\mathrm{AL}||^2\, /\, ||h_l||^2$',
              r'$||\mathrm{ML}||^2\, /\, ||h_l||^2$']

    _overlap_bar(ax, x, vals, colors, labels)
    _setup_axes(ax, n_layers, 'Layer', labels, colors)
    _add_legend(fig, labels, colors)

    output_path = os.path.join(PLOT_OUTPUT_DIR, "residual_stream_ratios.png")
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_cosine(al_stats, ml_stats, al_plus_ml_stats):
    """Plot layer-wise cosine similarities."""
    print("\nPlotting cosine similarities...")

    n_layers = len(al_stats)
    fig_width = max(16, n_layers * 0.52)
    fig_height = 9.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.05, right=0.98)

    al_vals = al_stats['mean_value'].values.copy()
    ml_vals = ml_stats['mean_value'].values.copy()
    al_ml_vals = al_plus_ml_stats['mean_value'].values.copy()

    x = np.arange(n_layers)
    vals = [al_ml_vals, al_vals, ml_vals]
    colors = [_COLOR_AL_ML, _COLOR_AL, _COLOR_ML]
    labels = [r'$\cos(\mathrm{AL+ML},\ h_l)$',
              r'$\cos(\mathrm{AL},\ h_l)$',
              r'$\cos(\mathrm{ML},\ h_l)$']

    _overlap_bar(ax, x, vals, colors, labels)
    _setup_axes(ax, n_layers, 'Layer', labels, colors)
    _add_legend(fig, labels, colors)

    output_path = os.path.join(PLOT_OUTPUT_DIR, "cosine_similarities.png")
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print(f"Residual Stream Plotting ({MODEL_LABEL})")
    print("=" * 60)

    metrics = {}
    for name in ['cos_al', 'cos_ml', 'cos_al_plus_ml',
                 'ratio_al', 'ratio_ml', 'ratio_al_plus_ml']:
        filepath = os.path.join(INPUT_DIR, f"{name}.csv")
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found!")
            print("Please run layer_proxy_collection.py first.")
            return
        metrics[name] = load_scalar_csv(filepath)
        print(f"  Loaded {name}: {len(metrics[name])} layers")

    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"cos(AL, h_l)     - mean: {metrics['cos_al']['mean_value'].mean():.6f}")
    print(f"cos(ML, h_l)     - mean: {metrics['cos_ml']['mean_value'].mean():.6f}")
    print(f"cos(AL+ML, h_l)  - mean: {metrics['cos_al_plus_ml']['mean_value'].mean():.6f}")

    plot_ratios(metrics['ratio_al'], metrics['ratio_ml'], metrics['ratio_al_plus_ml'])
    plot_cosine(metrics['cos_al'], metrics['cos_ml'], metrics['cos_al_plus_ml'])

    print(f"\n{'='*60}")
    print("All Done!")
    print(f"{'='*60}")
    print(f"Output directory: {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
