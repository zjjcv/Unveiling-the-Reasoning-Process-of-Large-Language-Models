"""
Plot Integrated Gradients heatmap (Qwen3-4B-Base).

This script reads IG data and generates a heatmap showing the attribution
of each input token to each layer's representation.

Input: /data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/IG/IG.csv

Output: /data/zjj/Synergistic_Core/results/Qwen3_4B_Base/Figure/IG_heatmap.png

Usage:
    python utils/plot/IG_plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global style (consistent with paper figures) ───────────────────────
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

# Configuration (selected via CLI)
import sys

if len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/IG/IG.csv"
    OUTPUT_PLOT = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure/IG_heatmap.png"
    NUM_LAYERS = 34   # Gemma3-4B-Base
    MODEL_LABEL = "Gemma3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b':
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/IG/IG.csv"
    OUTPUT_PLOT = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure/IG_heatmap.png"
    NUM_LAYERS = 40   # Gemma-3-12B-Instruct
    MODEL_LABEL = "Gemma-3-12B-Instruct"
else:
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/IG/IG.csv"
    OUTPUT_PLOT = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/Figure/IG_heatmap.png"
    NUM_LAYERS = 36   # Gemma-3-4B-Instruct
    MODEL_LABEL = "Gemma-3-4B-Instruct"


def load_ig_data(input_path: str) -> pd.DataFrame:
    """Load IG data from CSV file.

    Args:
        input_path: Path to IG CSV file

    Returns:
        DataFrame with IG data
    """
    print(f"Loading IG data from {input_path}...")

    df = pd.read_csv(input_path)

    print(f"  Total records: {len(df):,}")
    print(f"  Layers: {df['layer'].nunique()}")
    print(f"  Token positions: {df['token_position'].nunique()}")

    return df


def prepare_ig_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare IG data for heatmap visualization.

    Args:
        df: Raw IG DataFrame

    Returns:
        DataFrame with layer, token_position, token, ig_value
    """
    print("\nPreparing IG data...")

    print(f"  Records: {len(df):,}")
    print(f"  IG range: [{df['ig_value'].min():.6f}, {df['ig_value'].max():.6f}]")
    print(f"  Mean IG: {df['ig_value'].mean():.6f}")

    return df


def create_heatmap_matrix(df: pd.DataFrame, num_layers: int) -> tuple:
    """Create heatmap matrix from IG data.

    Args:
        df: Prepared IG DataFrame
        num_layers: Number of layers

    Returns:
        Tuple of (matrix, token_labels)
    """
    print("\nCreating heatmap matrix...")

    # Pivot to create layer x token matrix
    matrix = df.pivot(index='layer', columns='token_position', values='ig_value')

    # Get unique tokens for labels
    token_labels = df.groupby('token_position')['token'].first().values

    # Fill missing values with 0
    matrix = matrix.fillna(0)

    # Ensure we have all layers
    for layer in range(num_layers):
        if layer not in matrix.index:
            matrix.loc[layer] = 0

    # Sort by layer
    matrix = matrix.sort_index()

    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Matrix value range: [{matrix.values.min():.6f}, {matrix.values.max():.6f}]")

    return matrix.values, token_labels


def plot_ig_heatmap(matrix: np.ndarray, token_labels: np.ndarray, output_path: str):
    """Plot IG heatmap with layer-level aggregation panel on the right."""
    print("\nPlotting IG heatmap with layer aggregation...")

    num_tokens = len(token_labels)
    num_layers_display = matrix.shape[0]

    # Use absolute values for visualization
    matrix_abs = np.abs(matrix)

    # Normalize for better visualization
    matrix_norm = (matrix_abs - matrix_abs.min()) / (matrix_abs.max() - matrix_abs.min() + 1e-10)

    # Per-layer mean IG (original absolute values)
    layer_mean = np.mean(matrix_abs, axis=1)

    # Custom colormap: Blue -> light yellow -> Red
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FFF9C4', '#FDDCB3', '#F4A582', '#D6604D', '#B2182B']
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('blue_yellow_red', colors, N=n_bins)

    # Two panels: heatmap + layer bar chart
    fig_width = max(14, num_tokens * 0.45)
    fig_height = max(8, num_layers_display * 0.35)

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(fig_width + 3.5, fig_height),
        gridspec_kw={'width_ratios': [num_tokens, 2.2], 'wspace': 0.05},
        constrained_layout=True
    )

    # ── Left: heatmap ──
    sns.heatmap(
        matrix_norm,
        cmap=cmap_custom,
        ax=ax_heat,
        cbar_kws={'label': 'Normalized Attribution Magnitude',
                  'shrink': 0.82, 'aspect': 30},
        vmin=0,
        vmax=1,
        linewidths=0,
        xticklabels=range(num_tokens),
        yticklabels=range(num_layers_display),
    )

    # Style colorbar
    im = ax_heat.collections[0]
    cbar = im.colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_linewidth(0)
    cbar.outline.set_edgecolor('none')

    # Layer boundary lines on heatmap
    key_boundaries = [6, 12, 18, 24, 30]
    for y in key_boundaries:
        if y < num_layers_display:
            ax_heat.axhline(y=y, color='#333333', linewidth=1.2, alpha=0.7, linestyle='--')

    ax_heat.set_xlabel('Token Position', fontsize=24)
    ax_heat.set_ylabel('Layer Index', fontsize=24)

    if num_layers_display >= 30:
        yticks = ax_heat.get_yticks()
        yticks_filtered = [t for t in yticks if 0 <= t < num_layers_display and int(t) % 2 == 0]
        ax_heat.set_yticks(yticks_filtered)
        ax_heat.set_yticklabels([int(t) for t in yticks_filtered])

    ax_heat.tick_params(axis='both', length=5)

    # ── Right: per-layer mean IG bar chart ──
    layers = np.arange(num_layers_display)
    bar_colors = []
    for l in layers:
        if l < 6:
            bar_colors.append('#92C5DE')
        elif l < 12:
            bar_colors.append('#4393C3')
        elif l < 18:
            bar_colors.append('#D1E5F0')
        elif l < 24:
            bar_colors.append('#FDDCB3')
        elif l < 30:
            bar_colors.append('#F4A582')
        else:
            bar_colors.append('#D6604D')

    ax_bar.barh(layers, layer_mean, color=bar_colors, edgecolor='none', height=0.85)

    for y in key_boundaries:
        if y < num_layers_display:
            ax_bar.axhline(y=y, color='#333333', linewidth=1.2, alpha=0.7, linestyle='--')

    ax_bar.set_xlabel('Mean |IG|', fontsize=24)
    ax_bar.set_ylim(num_layers_display - 0.5, -0.5)
    ax_bar.set_yticks([])
    ax_bar.grid(True, axis='x', linestyle='-', alpha=0.12, linewidth=0.6, color='#555555')

    for spine in ['top', 'right']:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines['left'].set_linewidth(1.3)
    ax_bar.spines['bottom'].set_linewidth(1.3)
    ax_bar.tick_params(axis='both', length=5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Integrated Gradients Heatmap Plotting")
    print("=" * 60)
    print(f"\nInput: {INPUT_CSV}")
    print(f"Output: {OUTPUT_PLOT}")

    # Check input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"\nError: Input file not found: {INPUT_CSV}")
        print("Please run src/IG_collection.py first to generate the data.")
        return

    # Load data
    print("\n" + "=" * 60)
    print("Loading IG data...")
    print("=" * 60)

    ig_df = load_ig_data(INPUT_CSV)

    # Prepare data
    prepared_df = prepare_ig_data(ig_df)

    # Create heatmap matrix
    matrix, token_labels = create_heatmap_matrix(prepared_df, NUM_LAYERS)

    # Plot heatmap
    plot_ig_heatmap(matrix, token_labels, OUTPUT_PLOT)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
