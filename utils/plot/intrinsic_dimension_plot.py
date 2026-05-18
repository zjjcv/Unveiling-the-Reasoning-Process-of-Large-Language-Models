"""
Plot intrinsic dimension vs layer index with synergistic core region highlighted.

Baseline mode:
  Reads intrinsic_dimension.csv, plots single ID curve.

Ablation mode:
  Reads intrinsic_dimension_ablation.csv, plots baseline + syn-ablated + red-ablated curves.

Usage:
    python utils/plot/intrinsic_dimension_plot.py                  # Qwen3-8B-Base baseline
    python utils/plot/intrinsic_dimension_plot.py qwen3_4b_base     # Qwen3-4B-Base baseline
    python utils/plot/intrinsic_dimension_plot.py qwen3_8b_base ablation   # with ablation curves
    python utils/plot/intrinsic_dimension_plot.py qwen3_14b_base ablation
    python utils/plot/intrinsic_dimension_plot.py gemma3_4b_base ablation
    python utils/plot/intrinsic_dimension_plot.py gemma3_12b_it ablation
    python utils/plot/intrinsic_dimension_plot.py llama3_8b ablation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Noto Serif', 'DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'font.size': 22,
    'font.weight': 'bold',
    'axes.labelsize': 28,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'standard',
    'savefig.pad_inches': 0.1,
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

# ── Paths (selected via CLI) ──────────────────────────────────────────
if len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/Figure"
    MODEL_LABEL = "Qwen3-4B-Base"
    SYN_CORE = (6, 22)
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/Figure"
    MODEL_LABEL = "Qwen3-14B-Base"
    SYN_CORE = (5, 18)
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure"
    MODEL_LABEL = "Gemma3-4B-Base"
    SYN_CORE = (6, 21)
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b_it':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure"
    MODEL_LABEL = "Gemma3-12B-IT"
    SYN_CORE = (10, 29)
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/Figure"
    MODEL_LABEL = "Llama-3.1-8B"
    SYN_CORE = (3, 20)
else:
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/Figure"
    MODEL_LABEL = "Qwen3-8B-Base"
    SYN_CORE = (8, 22)

# Check for ablation mode (second CLI argument)
_ABLATION = len(sys.argv) > 2 and sys.argv[2] == 'ablation'

# Colours (matching head_ablation_plot.py: gray/red/blue)
_COLOR_BASELINE = '#888888'   # gray
_COLOR_SYN = '#E64B35'        # warm red
_COLOR_RED = '#6495ED'        # light blue


def main():
    mode_str = "Ablation" if _ABLATION else "Baseline"
    print("=" * 60)
    print(f"Intrinsic Dimension Plot — {MODEL_LABEL} ({mode_str})")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    core_start, core_end = (SYN_CORE if SYN_CORE else (None, None))

    if _ABLATION:
        _plot_ablation(core_start, core_end)
    else:
        _plot_baseline(core_start, core_end)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def _plot_baseline(core_start, core_end):
    """Plot single baseline ID curve."""
    data_path = os.path.join(DATA_DIR, "intrinsic_dimension.csv")

    if not os.path.exists(data_path):
        print(f"\n  Error: No data found: {data_path}")
        return

    stats = pd.read_csv(data_path)
    print(f"\n  Loaded data: {len(stats)} layers")
    print(f"  ID range: [{stats['intrinsic_dimension'].min():.2f}, "
          f"{stats['intrinsic_dimension'].max():.2f}]")

    if core_start is not None:
        print(f"  Synergistic core: layers {core_start}-{core_end}")

    fig, ax = plt.subplots(figsize=(14, 7))

    if core_start is not None:
        ax.axvspan(core_start - 0.5, core_end + 0.5,
                    alpha=0.08, color='#999999', zorder=0, label='Synergistic Core')
        for x in [core_start - 0.5, core_end + 0.5]:
            ax.axvline(x=x, color='#555555', linestyle='--', linewidth=1.5,
                        alpha=0.6, zorder=1)

    ax.plot(stats['layer'], stats['intrinsic_dimension'],
             '-o', color=_COLOR_BASELINE, linewidth=2.8, markersize=8,
             markeredgewidth=1.5, markeredgecolor='white',
             label='Intrinsic Dimension (k-NN MLE)', zorder=3)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Intrinsic Dimension')
    ax.set_xlim(-0.5, stats['layer'].max() + 0.5)

    ax.grid(True, axis='y', linestyle='-', alpha=0.12, linewidth=0.6, color='#888888')
    ax.set_axisbelow(True)

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.0, -0.18, 1.0, 0.0),
                     ncol=2, framealpha=0.95, edgecolor='#CCCCCC',
                     fancybox=True, shadow=False, borderpad=0.6,
                     handletextpad=0.6, columnspacing=2.5,
                     fontsize=22, mode='expand', borderaxespad=0.0)
    leg.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.22)
    output_path = os.path.join(OUTPUT_DIR, "intrinsic_dimension.png")
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()


def _plot_ablation(core_start, core_end):
    """Plot baseline + syn-ablated + red-ablated ID curves."""
    abl_path = os.path.join(DATA_DIR, "intrinsic_dimension_ablation.csv")

    if not os.path.exists(abl_path):
        print(f"\n  Error: No ablation data found: {abl_path}")
        print(f"  Run: python src/intrinsic_dimension_ablation.py [model]")
        return

    df = pd.read_csv(abl_path)
    # Read ablation fraction from CSV (same value in every row)
    frac = int(df['ablate_fraction'].iloc[0] * 100)
    print(f"\n  Loaded ablation data: {len(df)} layers (top {frac}% ablated)")
    print(f"  Baseline  ID range: [{df['baseline'].min():.2f}, {df['baseline'].max():.2f}]")
    print(f"  Ablate-Syn ID range: [{df['ablate_syn'].min():.2f}, {df['ablate_syn'].max():.2f}]")
    print(f"  Ablate-Red ID range: [{df['ablate_red'].min():.2f}, {df['ablate_red'].max():.2f}]")

    if core_start is not None:
        print(f"  Synergistic core: layers {core_start}-{core_end}")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Synergistic core region — shaded band + dashed boundaries
    if core_start is not None:
        ax.axvspan(core_start - 0.5, core_end + 0.5,
                    alpha=0.08, color='#999999', zorder=0, label='Synergistic Core')
        for x in [core_start - 0.5, core_end + 0.5]:
            ax.axvline(x=x, color='#555555', linestyle='--', linewidth=1.5,
                        alpha=0.6, zorder=1)

    # Marker edge white for contrast, larger markers, thicker lines
    _MKWS = dict(markersize=8, markeredgewidth=1.5, markeredgecolor='white')

    # Three curves
    ax.plot(df['layer'], df['baseline'],
             '-o', color=_COLOR_BASELINE, linewidth=2.8, zorder=3,
             label='Baseline', **_MKWS)
    ax.plot(df['layer'], df['ablate_syn'],
             '-s', color=_COLOR_SYN, linewidth=2.8, zorder=3,
             label=f'Syn Ablation ({frac}%)', **_MKWS)
    ax.plot(df['layer'], df['ablate_red'],
             '-^', color=_COLOR_RED, linewidth=2.8, zorder=3,
             label=f'Red Ablation ({frac}%)', **_MKWS)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Intrinsic Dimension')
    ax.set_xlim(-0.5, df['layer'].max() + 0.5)

    # Subtle grid
    ax.grid(True, axis='y', linestyle='-', alpha=0.12, linewidth=0.6, color='#888888')
    ax.set_axisbelow(True)

    # Legend below x-axis, flattened into one row
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.0, -0.20, 1.0, 0.0),
                     ncol=4, framealpha=0.95, edgecolor='#CCCCCC',
                     fancybox=True, shadow=False, borderpad=0.6,
                     handletextpad=0.8, columnspacing=3.0,
                     fontsize=18, mode='expand', borderaxespad=0.0)
    leg.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.22)
    output_path = os.path.join(OUTPUT_DIR, "intrinsic_dimension_ablation.png")
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()

    # ── Also save a baseline-only plot ──
    _save_baseline_only(df, core_start, core_end)


def _save_baseline_only(df, core_start, core_end):
    """Save a separate baseline-only ID curve from ablation data."""
    fig, ax = plt.subplots(figsize=(14, 7))

    if core_start is not None:
        ax.axvspan(core_start - 0.5, core_end + 0.5,
                    alpha=0.08, color='#999999', zorder=0, label='Synergistic Core')
        for x in [core_start - 0.5, core_end + 0.5]:
            ax.axvline(x=x, color='#555555', linestyle='--', linewidth=1.5,
                        alpha=0.6, zorder=1)

    ax.plot(df['layer'], df['baseline'],
             '-o', color=_COLOR_BASELINE, linewidth=2.8, markersize=8,
             markeredgewidth=1.5, markeredgecolor='white',
             label='Intrinsic Dimension (k-NN MLE)', zorder=3)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Intrinsic Dimension')
    ax.set_xlim(-0.5, df['layer'].max() + 0.5)

    ax.grid(True, axis='y', linestyle='-', alpha=0.12, linewidth=0.6, color='#888888')
    ax.set_axisbelow(True)

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.0, -0.18, 1.0, 0.0),
                     ncol=2, framealpha=0.95, edgecolor='#CCCCCC',
                     fancybox=True, shadow=False, borderpad=0.6,
                     handletextpad=0.6, columnspacing=2.5,
                     fontsize=22, mode='expand', borderaxespad=0.0)
    leg.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.22)
    output_path = os.path.join(OUTPUT_DIR, "intrinsic_dimension.png")
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
