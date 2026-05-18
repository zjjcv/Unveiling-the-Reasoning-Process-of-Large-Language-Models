"""
Plot layer relative change heatmaps from 4 experiment CSVs (saved as 4 separate figures).

Figures:
  (a) Full Ablation         → layer_relative_change.png
  (b) Future Prediction     → future_prediction.png
  (c) Circuit Loc (All)     → circuit_localization_all.png
  (d) Circuit Loc (Future)  → circuit_localization_future.png

Y-axis: Ablated layer (s), X-axis: Affected layer (l)
Color: Metric magnitude (upper-right triangle only, l > s)

Usage:
    python utils/plot/layer_relative_change_plot.py
    python utils/plot/layer_relative_change_plot.py gemma3_4b_base
    python utils/plot/layer_relative_change_plot.py gemma3_12b
    python utils/plot/layer_relative_change_plot.py qwen3_8b_base
    python utils/plot/layer_relative_change_plot.py qwen3_14b_base
    python utils/plot/layer_relative_change_plot.py llama3_8b
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global style (consistent with synergy_core_syn_ratio_rank.py) ──────
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Noto Serif', 'DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'font.size': 18,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
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

# ── Paths (selected via CLI) ────────────────────────────────────────────
import sys

if len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/layer_relative_change"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure"
    NUM_LAYERS = 34   # Gemma3-4B-Base
    MODEL_LABEL = "Gemma3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/layer_relative_change"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure"
    NUM_LAYERS = 48   # Gemma3-12B-Instruct
    MODEL_LABEL = "Gemma3-12B-Instruct"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_8b_base':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/layer_relative_change"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/Figure"
    NUM_LAYERS = 36
    MODEL_LABEL = "Qwen3-8B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/layer_relative_change"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/Figure"
    NUM_LAYERS = 36
    MODEL_LABEL = "Qwen3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/layer_relative_change"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/Figure"
    NUM_LAYERS = 40
    MODEL_LABEL = "Qwen3-14B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/layer_relative_change"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/Figure"
    NUM_LAYERS = 32
    MODEL_LABEL = "Llama-3.1-8B"
else:
    INPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/layer_relative_change"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/Figure"
    NUM_LAYERS = 36   # Gemma-3-4B-Instruct
    MODEL_LABEL = "Gemma-3-4B-Instruct"

# Experiment configuration
EXPERIMENTS = [
    {
        'filename': 'layer_relative_change.csv',
        'output': 'layer_relative_change.png',
        'title': 'Full Ablation',
    },
    {
        'filename': 'future_prediction.csv',
        'output': 'future_prediction.png',
        'title': 'Future Prediction',
    },
    {
        'filename': 'circuit_localization_all.csv',
        'output': 'circuit_localization_all.png',
        'title': 'Circuit Localization (All)',
    },
    {
        'filename': 'circuit_localization_future.csv',
        'output': 'circuit_localization_future.png',
        'title': 'Circuit Localization (Future)',
    },
]


def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    """Load CSV and aggregate metric_value by (s, l) pairs using max (paper convention)."""
    df = pd.read_csv(csv_path)
    # Handle legacy column name 'relative_change'
    if 'metric_value' not in df.columns and 'relative_change' in df.columns:
        df.rename(columns={'relative_change': 'metric_value'}, inplace=True)
    df_filtered = df[df['ablated_layer_s'] > 0].copy()
    aggregated = df_filtered.groupby(['ablated_layer_s', 'affected_layer_l'])['metric_value'].agg([
        'mean', 'count'
    ]).reset_index()
    aggregated.columns = ['ablated_layer_s', 'affected_layer_l', 'mean_metric', 'count']
    return aggregated


def create_heatmap_matrix(df: pd.DataFrame, num_layers: int) -> np.ndarray:
    """Create upper-triangle heatmap matrix: matrix[s, l] = mean metric."""
    matrix = np.full((num_layers, num_layers), np.nan)
    for _, row in df.iterrows():
        s = int(row['ablated_layer_s'])
        l = int(row['affected_layer_l'])
        if 0 <= s < num_layers and 0 <= l < num_layers and l > s:
            matrix[s, l] = row['mean_metric']
    return matrix


def plot_single_heatmap(matrix: np.ndarray, title: str, output_path: str):
    """Plot and save a single upper-triangle heatmap."""
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0',
                   '#FFF9C4', '#FDDCB3', '#F4A582', '#D6604D', '#B2182B']
    cmap = LinearSegmentedColormap.from_list('blue_yellow_red', colors_list, N=100)

    n = matrix.shape[0]

    # Upper triangle only
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    matrix_display = np.where(mask, matrix, np.nan)

    fig_w = max(12, n * 0.42)
    fig_h = fig_w * 0.88
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    im = sns.heatmap(
        matrix_display,
        cmap=cmap,
        ax=ax,
        cbar_kws={'label': 'Relative Change',
                  'shrink': 0.78, 'aspect': 28},
        vmin=0,
        vmax=np.nanmax(matrix_display) if np.any(~np.isnan(matrix_display)) else 1,
        linewidths=0,
        square=True,
        xticklabels=range(n),
        yticklabels=range(n),
    )

    # Style colorbar
    cbar = im.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_linewidth(0)
    cbar.outline.set_edgecolor('none')

    ax.set_title(title, fontsize=24, fontweight='bold')
    ax.set_xlabel('Affected Layer (l)')
    ax.set_ylabel('Ablated Layer (s)')
    ax.invert_yaxis()

    # Thin ticks
    step = 2 if n <= 40 else 4
    ax.set_xticks(ax.get_xticks()[::step])
    ax.set_xticklabels(range(0, n, step))
    ax.set_yticks(ax.get_yticks()[::step])
    ax.set_yticklabels(range(0, n, step))
    ax.tick_params(axis='both', length=5)

    # Diagonal reference line
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5],
            color='white', linestyle='--', linewidth=1.5, alpha=0.6)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print(f"Layer Relative Change Plots ({MODEL_LABEL})")
    print("=" * 60)
    print(f"\nInput: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for exp in EXPERIMENTS:
        csv_path = os.path.join(INPUT_DIR, exp['filename'])
        output_path = os.path.join(OUTPUT_DIR, exp['output'])

        if not os.path.exists(csv_path):
            print(f"\n  Warning: {csv_path} not found, skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"  {exp['title']}: {exp['filename']}")

        agg_df = load_and_aggregate(csv_path)
        matrix = create_heatmap_matrix(agg_df, NUM_LAYERS)
        valid = matrix[~np.isnan(matrix)]

        print(f"    Records: {len(agg_df)}")
        print(f"    Range: [{valid.min():.6f}, {valid.max():.6f}]")
        print(f"    Mean: {valid.mean():.6f}")

        plot_single_heatmap(matrix, exp['title'], output_path)

    print(f"\n{'='*60}")
    print("All Done!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Generated files:")
    for exp in EXPERIMENTS:
        print(f"  - {exp['output']}")


if __name__ == "__main__":
    main()
