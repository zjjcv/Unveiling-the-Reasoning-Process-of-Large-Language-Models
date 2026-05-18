import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

# ── Global style: publication-quality, serif font ──────────────────────
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

# ── Colour palette — RdBu_r consistent ──────────────────────────────────
_COLOR_BLUE = '#2166ac'
_COLOR_RED = '#b2182b'
_CMAP = 'RdBu_r'


def compute_head_stats_from_pairwise(pairwise_path: str):
    """Compute average syn and red for each head/layer from pairwise data.

    Args:
        pairwise_path: Path to pairwise CSV file

    Returns:
        DataFrame with columns [layer, head, syn, red]
        For layer-level data (ML, AL+ML), head column will be 0 for all rows
    """
    import pandas as pd
    print(f"Loading pairwise data from {pairwise_path}...")
    df = pd.read_csv(pairwise_path)

    print(f"  Total pairs: {len(df):,}")
    print(f"  Questions: {df['question_id'].nunique()}")

    # Check if this is head-level or layer-level data
    if 'head_1' in df.columns:
        head_stats = df.groupby(['layer_1', 'head_1']).agg({
            'syn': 'mean',
            'red': 'mean'
        }).reset_index()
        head_stats.columns = ['Layer', 'Head', 'Syn', 'Red']
        print(f"  Unique heads: {len(head_stats)}")
        print(f"  Layers: {head_stats['Layer'].nunique()}")
    else:
        layer_stats = df.groupby(['layer_1']).agg({
            'syn': 'mean',
            'red': 'mean'
        }).reset_index()
        layer_stats.columns = ['Layer', 'Syn', 'Red']
        layer_stats['Head'] = 0
        layer_stats = layer_stats[['Layer', 'Head', 'Syn', 'Red']]
        print(f"  Unique layers: {len(layer_stats)}")

    return head_stats if 'head_1' in df.columns else layer_stats


def plot_syn_ratio_rank_gsm8k(csv_path, output_dir=None, metric='syn_ratio_rank', level_name=''):
    """Plot Syn/(Syn+Red) Rank or (Syn-Red) Rank for pairwise data.

    Args:
        csv_path: Path to pairwise CSV file
        output_dir: Directory to save plots
        metric: 'syn_ratio_rank' or 'syn_red_rank'
        level_name: Name of the difficulty level
    """
    # 1. Load data - check if it's pairwise or already aggregated
    df_test = pd.read_csv(csv_path, nrows=10)

    is_pairwise = 'question_id' in df_test.columns and 'layer_1' in df_test.columns

    if is_pairwise:
        df = compute_head_stats_from_pairwise(csv_path)
    else:
        df = pd.read_csv(csv_path)
        if 'syn' in df.columns:
            df['Syn'] = df['syn']
        if 'red' in df.columns:
            df['Red'] = df['red']
        if 'layer' in df.columns:
            df['Layer'] = df['layer']
        if 'head' in df.columns:
            df['Head'] = df['head']

    # 2. Compute syn_ratio = Syn / (Syn + Red)
    df['syn_ratio'] = df['Syn'] / (df['Syn'] + df['Red'])

    # 3. Compute rank of syn_ratio
    df['syn_ratio_rank'] = df['syn_ratio'].rank(method='dense')

    # 4. Compute (Syn-Red) and its rank
    df['syn_red_diff'] = df['Syn'] - df['Red']
    df['syn_red_rank'] = df['syn_red_diff'].rank(method='dense')

    # Select metric
    if metric == 'syn_red_rank':
        metric_col = 'syn_red_rank'
        metric_label = '(Syn-Red) Rank'
    else:
        metric_col = 'syn_ratio_rank'
        metric_label = 'Syn/(Syn+Red) Rank'

    # Head level plot: heatmap + layer profile
    heatmap_data = df.pivot(index='Head', columns='Layer', values=metric_col)

    n_layers = heatmap_data.shape[1]
    n_heads = heatmap_data.shape[0]

    # Adaptive figure size for large grids (Qwen3-8B: 36 layers × 32 heads)
    heatmap_width = max(14, n_layers * 0.48)
    heatmap_height = max(6, n_heads * 0.22)
    fig_width = heatmap_width + 12
    fig_height = max(heatmap_height, 5.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height),
                                    gridspec_kw={'width_ratios': [heatmap_width, 12],
                                                 'wspace': 0.12},
                                    constrained_layout=True)

    # ── Heatmap ────────────────────────────────────────────────────────
    cmap = plt.get_cmap(_CMAP)
    vmin, vmax = heatmap_data.min().min(), heatmap_data.max().max()

    im = ax1.imshow(heatmap_data.values, cmap=cmap, aspect='auto',
                    vmin=vmin, vmax=vmax, interpolation='nearest', origin='upper')

    # Ticks: show actual Layer/Head values
    xtick_positions = np.arange(n_layers)
    ytick_positions = np.arange(n_heads)
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(heatmap_data.columns)
    ax1.set_yticks(ytick_positions)
    ax1.set_yticklabels(heatmap_data.index)

    # Thin ticks for large grids
    if n_layers > 20:
        ax1.set_xticks(xtick_positions[::2])
        ax1.set_xticklabels(heatmap_data.columns[::2])
    if n_heads > 16:
        ax1.set_yticks(ytick_positions[::2])
        ax1.set_yticklabels(heatmap_data.index[::2])

    # Colorbar — slim, clean, no label
    cbar = fig.colorbar(im, ax=ax1, shrink=0.75, aspect=22, pad=0.02)
    cbar.ax.tick_params(labelsize=22, length=4, width=0.8)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#cccccc')

    # Frame: consistent dark border with right plot
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('')
    ax1.tick_params(axis='both', length=3, width=0.6, colors='#333333', direction='out')

    # Highlight synergistic core region: top-25% layers by average rank
    layer_means = heatmap_data.mean(axis=0)
    threshold = layer_means.quantile(0.75)
    core_layers = layer_means[layer_means >= threshold].index.tolist()
    core_start = min(core_layers) if core_layers else None
    core_end = max(core_layers) if core_layers else None
    if core_layers:
        rect = mpl.patches.Rectangle(
            (core_start - 0.5, -0.5), core_end - core_start + 1, n_heads,
            linewidth=1.8, edgecolor='#333333', facecolor='none',
            linestyle='--', zorder=10, alpha=0.7)
        ax1.add_patch(rect)


    # ── Layer profile (smoothed) ───────────────────────────────────────
    layer_avg = df.groupby('Layer')[metric_col].mean().reset_index()

    x_norm = (layer_avg['Layer'] - layer_avg['Layer'].min()) / (layer_avg['Layer'].max() - layer_avg['Layer'].min())
    y_min, y_max = layer_avg[metric_col].min(), layer_avg[metric_col].max()
    y_values = (layer_avg[metric_col].values - y_min) / (y_max - y_min)

    window_size = max(4, len(x_norm) // 5)
    y_ma = np.convolve(y_values, np.ones(window_size) / window_size, mode='same')

    win_len = min(15, len(y_ma) // 2 * 2 + 1)
    if win_len % 2 == 0:
        win_len += 1
    y_smooth = savgol_filter(y_ma, window_length=win_len, polyorder=3)

    x_smooth = np.linspace(x_norm.min(), x_norm.max(), 300)
    spline = make_interp_spline(x_norm, y_smooth, k=3)
    y_final = spline(x_smooth)

    # Diverging fill: above 0.5 → red, below 0.5 → blue, intensity by distance
    cmap_profile = plt.get_cmap(_CMAP)
    ax2.axhline(0.5, color='#555555', linestyle='-', linewidth=0.6, alpha=0.4, zorder=0)
    ax2.fill_between(x_smooth, 0.5, y_final, where=y_final >= 0.5,
                     color=_COLOR_RED, alpha=0.18, linewidth=0)
    ax2.fill_between(x_smooth, 0.5, y_final, where=y_final < 0.5,
                     color=_COLOR_BLUE, alpha=0.18, linewidth=0)

    # Curve line: color gradient via colored segments
    for i in range(len(x_smooth) - 1):
        val = y_final[i]
        color = cmap_profile(val)
        ax2.plot(x_smooth[i:i+2], y_final[i:i+2], color=color, lw=3.2,
                 solid_capstyle='round', zorder=3)

    # Scatter: each dot colored by its value via the same RdBu_r colormap
    scatter_colors = [cmap_profile(v) for v in y_values]
    ax2.scatter(x_norm, y_values, s=52, c=scatter_colors, edgecolors='#333333',
                linewidths=0.6, zorder=5, alpha=0.9)

    ax2.set_xlabel('Normalized Layer')
    ax2.set_ylabel('')
    ax2.set_xlim(0, 1)
    y_pad = 0.05
    ax2.set_ylim(-y_pad, 1 + y_pad)
    ax2.grid(False)
    ax2.tick_params(axis='both', length=3, width=0.6, colors='#333333', direction='out')
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    # Highlight core region on profile too (same layers as heatmap)
    if core_layers:
        x_core_start = (core_start - layer_avg['Layer'].min()) / (layer_avg['Layer'].max() - layer_avg['Layer'].min())
        x_core_end = (core_end - layer_avg['Layer'].min()) / (layer_avg['Layer'].max() - layer_avg['Layer'].min())
        ax2.axvspan(x_core_start, x_core_end, alpha=0.06, color='#333333', zorder=0)

    metric_suffix = 'syn_red_diff' if metric == 'syn_red_rank' else 'syn_ratio_rank'
    if level_name:
        level_suffix = level_name.lower().replace(' ', '_')
        output_path = os.path.join(output_dir, f"{level_suffix}_{metric_suffix}_heatmap.png")
    else:
        output_path = os.path.join(output_dir, f"qwen3_gsm8k_al_{metric_suffix}_profile.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ======================================================================
# CLI entry point
# ======================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'gemma3':
        # Gemma3-4B-Instruct GSM8K heatmap generator
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/L2_Norm"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/Figure"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Gemma3-4B-Instruct GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"Gemma3-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - gemma3_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - gemma3_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b_base':
        # ── Gemma3-12B-Base GSM8K heatmap generator ─────────────────────
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/data/L2_Norm"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/Figure"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Gemma3-12B-Base GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"Gemma3-12B-Base-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - gemma3_12b_base_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - gemma3_12b_base_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b':
        # Gemma3-12B-IT GSM8K heatmap generator
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-IT/data/L2_Norm"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-IT/Figure"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Gemma3-12B-IT GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"Gemma3-12B-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - gemma3_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - gemma3_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'math':
        # MATH data configuration
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/MATH"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/MATH/plots/heatmaps"
        LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
        LEVEL_SUFFIXES = ["Level_1", "Level_2", "Level_3", "Level_4", "Level_5"]

        print("=" * 60)
        print("MATH Heatmap Generator")
        print("=" * 60)

        for level, level_suffix in zip(LEVELS, LEVEL_SUFFIXES):
            print(f"\n{'='*60}")
            print(f"Processing {level}")
            print(f"{'='*60}")

            level_dir = os.path.join(INPUT_BASE_DIR, level_suffix)
            input_path = os.path.join(level_dir, f"math_{level_suffix.lower()}_al_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=level
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=level
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (10 heatmaps):")
        for level_suffix in LEVEL_SUFFIXES:
            level_suffix_lower = level_suffix.lower()
            print(f"  - math_{level_suffix_lower}_syn_red_diff_heatmap.png")
            print(f"  - math_{level_suffix_lower}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'mmlu':
        # MMLU data configuration
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/mmlu/pairwise"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/mmlu/plots/heatmaps"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("MMLU Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"MMLU {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - mmlu_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - mmlu_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_8b_base_gsm8k':
        # ── Qwen3-8B-Base GSM8K all proxy types heatmap generator ────
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/L2_Norm"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/Figure"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Qwen3-8B-Base GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"Qwen3-8B-Base-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - qwen3_8b_base_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - qwen3_8b_base_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_gsm8k':
        # ── Qwen3-8B GSM8K publication-quality heatmaps ───────────────
        INPUT_FILE = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/L2_Norm/al_syn_red_pairwise.csv"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Plots/Qwen3_8_Base"

        print("=" * 60)
        print("Qwen3-8B-GSM8K Publication Heatmap Generator")
        print("=" * 60)
        print(f"\nInput:  {INPUT_FILE}")
        print(f"Output: {OUTPUT_DIR}")

        if not os.path.exists(INPUT_FILE):
            print(f"\nError: Input file not found: {INPUT_FILE}")
            exit(1)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("\nPlotting (Syn-Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_red_rank'
        )

        print("\nPlotting Syn/(Syn+Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_ratio_rank'
        )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print(f"  - qwen3_gsm8k_al_syn_red_diff_profile.png")
        print(f"  - qwen3_gsm8k_al_syn_ratio_rank_profile.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'arc':
        # ── Qwen3-8B ARC (Easy + Challenge) publication-quality heatmaps ─
        ARC_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/ai2arc/pairwise"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Plots/Qwen3_8_Base"

        ARC_DATASETS = [
            ("ARC-Easy", "easy"),
            ("ARC-Challenge", "challenge")
        ]

        print("=" * 60)
        print("Qwen3-8B ARC Publication Heatmap Generator")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for dataset_name, dataset_dir in ARC_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")

            input_file = os.path.join(ARC_BASE_DIR, dataset_dir, "al_syn_red_pairwise.csv")

            if not os.path.exists(input_file):
                print(f"Warning: Input file not found: {input_file}")
                print(f"   Skipping...")
                continue

            print(f"Input: {input_file}")

            # Plot (Syn-Red) Rank
            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=dataset_name
            )

            # Plot Syn/(Syn+Red) Rank
            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=dataset_name
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (4 heatmaps):")
        print(f"  - arc_easy_syn_red_diff_heatmap.png")
        print(f"  - arc_easy_syn_ratio_rank_heatmap.png")
        print(f"  - arc_challenge_syn_red_diff_heatmap.png")
        print(f"  - arc_challenge_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'arc_qwen3_4b_base':
        # ── Qwen3-4B-Base ARC (Easy + Challenge) publication-quality heatmaps ─
        ARC_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/Figure"

        ARC_DATASETS = [
            ("ARC-Easy", "easy"),
            ("ARC-Challenge", "challenge")
        ]

        print("=" * 60)
        print("Qwen3-4B-Base ARC Publication Heatmap Generator")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for dataset_name, dataset_dir in ARC_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")

            input_file = os.path.join(ARC_BASE_DIR, dataset_dir, "al_syn_red_pairwise.csv")

            if not os.path.exists(input_file):
                print(f"Warning: Input file not found: {input_file}")
                print(f"   Skipping...")
                continue

            print(f"Input: {input_file}")

            # Plot (Syn-Red) Rank
            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=f"Qwen3-4B-{dataset_name}"
            )

            # Plot Syn/(Syn+Red) Rank
            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=f"Qwen3-4B-{dataset_name}"
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (4 heatmaps):")
        print(f"  - qwen3_4b_base_arc_easy_syn_red_diff_heatmap.png")
        print(f"  - qwen3_4b_base_arc_easy_syn_ratio_rank_heatmap.png")
        print(f"  - qwen3_4b_base_arc_challenge_syn_red_diff_heatmap.png")
        print(f"  - qwen3_4b_base_arc_challenge_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_al':
        # ── Gemma3-4B-Instruct GSM8K AL pairwise data ─────────────────
        INPUT_FILE = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/L2_Norm/al_syn_red_pairwise.csv"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/Figure"

        print("=" * 60)
        print("Gemma3-4B-Instruct GSM8K AL Heatmap Generator")
        print("=" * 60)
        print(f"\nInput:  {INPUT_FILE}")
        print(f"Output: {OUTPUT_DIR}")

        if not os.path.exists(INPUT_FILE):
            print(f"\nError: Input file not found: {INPUT_FILE}")
            exit(1)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("\nPlotting (Syn-Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_red_rank',
            level_name='Gemma3-GSM8K-AL'
        )

        print("\nPlotting Syn/(Syn+Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_ratio_rank',
            level_name='Gemma3-GSM8K-AL'
        )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print(f"  - gemma3_gsm8k_al_syn_red_diff_heatmap.png")
        print(f"  - gemma3_gsm8k_al_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'arc_gemma3_4b_base':
        # ── Gemma3-4B-Base ARC (Easy + Challenge) publication-quality heatmaps ─
        ARC_BASE_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure"

        ARC_DATASETS = [
            ("ARC-Easy", "easy"),
            ("ARC-Challenge", "challenge")
        ]

        print("=" * 60)
        print("Gemma3-4B-Base ARC Publication Heatmap Generator")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for dataset_name, dataset_dir in ARC_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")

            input_file = os.path.join(ARC_BASE_DIR, dataset_dir, "al_syn_red_pairwise.csv")

            if not os.path.exists(input_file):
                print(f"Warning: Input file not found: {input_file}")
                print(f"   Skipping...")
                continue

            print(f"Input: {input_file}")

            # Plot (Syn-Red) Rank
            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=f"Gemma3-4B-Base-{dataset_name}"
            )

            # Plot Syn/(Syn+Red) Rank
            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=f"Gemma3-4B-Base-{dataset_name}"
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (4 heatmaps):")
        print(f"  - gemma3_4b_base_arc_easy_syn_red_diff_heatmap.png")
        print(f"  - gemma3_4b_base_arc_easy_syn_ratio_rank_heatmap.png")
        print(f"  - gemma3_4b_base_arc_challenge_syn_red_diff_heatmap.png")
        print(f"  - gemma3_4b_base_arc_challenge_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
        # ── Gemma3-4B-Base GSM8K heatmap generator ─────────────────────
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/L2_Norm"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Gemma3-4B-Base GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"Gemma3-4B-Base-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - gemma3_4b_base_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - gemma3_4b_base_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
        # ── Qwen3-4B-Base GSM8K heatmap generator ─────────────────────
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/L2_Norm"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/Figure"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Qwen3-4B-Base GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"Qwen3-4B-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - qwen3_4b_base_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - qwen3_4b_base_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'arc_gemma3_4b':
        # ── Gemma-3-4B-Instruct ARC (Easy + Challenge) publication-quality heatmaps ─
        ARC_BASE_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/Figure"

        ARC_DATASETS = [
            ("ARC-Easy", "easy"),
            ("ARC-Challenge", "challenge")
        ]

        print("=" * 60)
        print("Gemma-3-4B-Instruct ARC Publication Heatmap Generator")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for dataset_name, dataset_dir in ARC_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")

            input_file = os.path.join(ARC_BASE_DIR, dataset_dir, "al_syn_red_pairwise.csv")

            if not os.path.exists(input_file):
                print(f"Warning: Input file not found: {input_file}")
                print(f"   Skipping...")
                continue

            print(f"Input: {input_file}")

            # Plot (Syn-Red) Rank
            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=f"Gemma3-4B-{dataset_name}"
            )

            # Plot Syn/(Syn+Red) Rank
            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=f"Gemma3-4B-{dataset_name}"
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (4 heatmaps):")
        print(f"  - gemma3_4b_arc_easy_syn_red_diff_heatmap.png")
        print(f"  - gemma3_4b_arc_easy_syn_ratio_rank_heatmap.png")
        print(f"  - gemma3_4b_arc_challenge_syn_red_diff_heatmap.png")
        print(f"  - gemma3_4b_arc_challenge_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'arc_gemma3_12b':
        # ── Gemma-3-12B-Instruct ARC (Easy + Challenge) publication-quality heatmaps ─
        ARC_BASE_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure"

        ARC_DATASETS = [
            ("ARC-Easy", "easy"),
            ("ARC-Challenge", "challenge")
        ]

        print("=" * 60)
        print("Gemma-3-12B-Instruct ARC Publication Heatmap Generator")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for dataset_name, dataset_dir in ARC_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print(f"{'='*60}")

            input_file = os.path.join(ARC_BASE_DIR, dataset_dir, "al_syn_red_pairwise.csv")

            if not os.path.exists(input_file):
                print(f"Warning: Input file not found: {input_file}")
                print(f"   Skipping...")
                continue

            print(f"Input: {input_file}")

            # Plot (Syn-Red) Rank
            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=f"Gemma3-12B-{dataset_name}"
            )

            # Plot Syn/(Syn+Red) Rank
            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_file,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=f"Gemma3-12B-{dataset_name}"
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (4 heatmaps):")
        print(f"  - gemma3_12b_arc_easy_syn_red_diff_heatmap.png")
        print(f"  - gemma3_12b_arc_easy_syn_ratio_rank_heatmap.png")
        print(f"  - gemma3_12b_arc_challenge_syn_red_diff_heatmap.png")
        print(f"  - gemma3_12b_arc_challenge_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
        # ── Qwen3-14B-Base GSM8K heatmap generator ─────────────────────
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/L2_Norm"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/Figure"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Qwen3-14B-Base GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                rank_path = os.path.join(INPUT_BASE_DIR, f"gsm8k_{proxy_type}_syn_red_rank.csv")
                if os.path.exists(rank_path):
                    input_path = rank_path
                else:
                    print(f"Warning: Input file not found for {proxy_type}")
                    print(f"   Skipping...")
                    continue

            title_prefix = f"Qwen3-14B-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - qwen3_14b_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - qwen3_14b_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
        # ── Llama-3.1-8B GSM8K heatmap generator ─────────────────────
        INPUT_BASE_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/L2_Norm"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/Figure"
        PROXY_TYPES = ["al", "ml", "al_plus_ml"]

        print("=" * 60)
        print("Llama-3.1-8B GSM8K Heatmap Generator")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for proxy_type in PROXY_TYPES:
            print(f"\n--- Proxy Type: {proxy_type} ---")

            input_path = os.path.join(INPUT_BASE_DIR, f"{proxy_type}_syn_red_pairwise.csv")

            if not os.path.exists(input_path):
                print(f"Warning: Input file not found: {input_path}")
                print(f"   Skipping...")
                continue

            title_prefix = f"Llama3-8B-GSM8K {proxy_type.upper()}"

            print(f"\nPlotting (Syn-Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_red_rank',
                level_name=title_prefix
            )

            print(f"\nPlotting Syn/(Syn+Red) Rank...")
            plot_syn_ratio_rank_gsm8k(
                input_path,
                output_dir=OUTPUT_DIR,
                metric='syn_ratio_rank',
                level_name=title_prefix
            )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files (6 heatmaps):")
        for proxy_type in PROXY_TYPES:
            print(f"  - llama3_8b_gsm8k_{proxy_type}_syn_red_diff_heatmap.png")
            print(f"  - llama3_8b_gsm8k_{proxy_type}_syn_ratio_rank_heatmap.png")

    else:
        INPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/2048_length/pairwise"
        OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/gsm8k/plots"

        INPUT_FILE = os.path.join(INPUT_DIR, "pairwise_2_syn_red.csv")

        print("=" * 60)
        print("Qwen3-8B-GSM8K Pairwise Syn-Red Rank Plotting")
        print("=" * 60)
        print(f"\nInput: {INPUT_FILE}")
        print(f"Output: {OUTPUT_DIR}")

        if not os.path.exists(INPUT_FILE):
            print(f"\nError: Input file not found: {INPUT_FILE}")
            exit(1)

        print("\nPlotting (Syn-Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_red_rank'
        )

        print("\nPlotting Syn/(Syn+Red) Rank...")
        plot_syn_ratio_rank_gsm8k(
            INPUT_FILE,
            output_dir=OUTPUT_DIR,
            metric='syn_ratio_rank'
        )

        print("\n" + "=" * 60)
        print("All plots complete!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print(f"  - qwen3_gsm8k_al_syn_red_rank_profile.png")
        print(f"  - qwen3_gsm8k_al_syn_ratio_rank_profile.png")
