"""
Plot Cross-vocabulary Abstraction Probes — Figure 5c & 5d (separate images).

Figure 5c: Cross-vocab rule accuracy (6-class) — peaks in synergistic core
Figure 5d: Visible token identity accuracy (30-class) — dips in core

Usage:
    python utils/plot/probe_abstraction_plot.py                  # Qwen3-8B-Base
    python utils/plot/probe_abstraction_plot.py qwen3_4b_base
    python utils/plot/probe_abstraction_plot.py qwen3_14b_base
    python utils/plot/probe_abstraction_plot.py gemma3_4b_base
    python utils/plot/probe_abstraction_plot.py gemma3_12b_it
    python utils/plot/probe_abstraction_plot.py llama3_8b
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global style ─────────────────────────────────────────────────────
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Noto Serif', 'DejaVu Serif', 'Times New Roman',
                    'Liberation Serif'],
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

# ── Paths (selected via CLI) ────────────────────────────────────────
if len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/Figure"
    MODEL_LABEL = "Qwen3-4B-Base"
    CORE_LAYERS = (6, 22)
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/Figure"
    MODEL_LABEL = "Qwen3-14B-Base"
    CORE_LAYERS = (5, 18)
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure"
    MODEL_LABEL = "Gemma3-4B-Base"
    CORE_LAYERS = (6, 21)
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b_it':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure"
    MODEL_LABEL = "Gemma3-12B-IT"
    CORE_LAYERS = (10, 29)
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/Figure"
    MODEL_LABEL = "Llama-3.1-8B"
    CORE_LAYERS = (3, 20)
else:
    DATA_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/intrinsic_dimension"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/Figure"
    MODEL_LABEL = "Qwen3-8B-Base"
    CORE_LAYERS = (8, 22)

_COLOR_LINE = '#888888'     # gray — consistent with ID baseline


def main():
    print("=" * 60)
    print(f"Probe Abstraction Plot — {MODEL_LABEL}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "probe_abstraction.csv")
    if not os.path.exists(csv_path):
        print(f"\n  Error: No data found: {csv_path}")
        print(f"  Run: python src/probe_abstraction.py [model]")
        return

    df = pd.read_csv(csv_path)
    core_start = int(df['core_start'].iloc[0])
    core_end = int(df['core_end'].iloc[0])
    # Override with correct CLI values (CSV may contain stale core range)
    core_start, core_end = CORE_LAYERS

    print(f"\n  Loaded: {len(df)} layers, core: {core_start}-{core_end}")
    print(f"  Rule acc range: [{df['rule_accuracy'].min():.4f}, "
          f"{df['rule_accuracy'].max():.4f}]")
    print(f"  Token acc range: [{df['vocab_accuracy'].min():.4f}, "
          f"{df['vocab_accuracy'].max():.4f}]")

    mkws = dict(markersize=8, markeredgewidth=1.5, markeredgecolor='white')
    xlim = (-0.5, df['layer'].max() + 0.5)

    def draw_core(ax):
        ax.axvspan(core_start - 0.5, core_end + 0.5,
                    alpha=0.08, color='#999999', zorder=0, label='Synergistic Core')
        for x in [core_start - 0.5, core_end + 0.5]:
            ax.axvline(x=x, color='#555555', linestyle='--', linewidth=1.5,
                        alpha=0.6, zorder=1)

    # ── Figure 5c: Rule accuracy ──
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    draw_core(ax1)
    ax1.plot(df['layer'], df['rule_accuracy'],
             '-o', color=_COLOR_LINE, linewidth=2.8, zorder=3,
             label='Rule Accuracy', **mkws)
    ax1.axhline(y=1/6, color='#AAAAAA', linestyle=':', linewidth=1.2,
                alpha=0.7, zorder=2, label='Chance (1/6)')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Cross-vocab Rule Accuracy')
    ax1.set_xlim(xlim)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis='y', linestyle='-', alpha=0.12, linewidth=0.6,
             color='#888888')
    ax1.set_axisbelow(True)

    leg1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.0, -0.18, 1.0, 0.0),
                       ncol=3, framealpha=0.95, edgecolor='#CCCCCC',
                       fancybox=True, shadow=False, borderpad=0.6,
                       handletextpad=0.6, columnspacing=2.5,
                       fontsize=22, mode='expand', borderaxespad=0.0)
    leg1.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.22)
    out1 = os.path.join(OUTPUT_DIR, "probe_rule_accuracy.png")
    plt.savefig(out1)
    plt.savefig(out1.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {out1}")
    plt.close()

    # ── Figure 5d: Token accuracy ──
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    draw_core(ax2)
    ax2.plot(df['layer'], df['vocab_accuracy'],
             '-s', color=_COLOR_LINE, linewidth=2.8, zorder=3,
             label='Token Identity Accuracy', **mkws)
    ax2.axhline(y=1/30, color='#AAAAAA', linestyle=':', linewidth=1.2,
                alpha=0.7, zorder=2, label='Chance (1/30)')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Token Identity Accuracy')
    ax2.set_xlim(xlim)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, axis='y', linestyle='-', alpha=0.12, linewidth=0.6,
             color='#888888')
    ax2.set_axisbelow(True)

    leg2 = ax2.legend(loc='upper center', bbox_to_anchor=(0.0, -0.18, 1.0, 0.0),
                       ncol=3, framealpha=0.95, edgecolor='#CCCCCC',
                       fancybox=True, shadow=False, borderpad=0.6,
                       handletextpad=0.6, columnspacing=2.5,
                       fontsize=22, mode='expand', borderaxespad=0.0)
    leg2.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.22)
    out2 = os.path.join(OUTPUT_DIR, "probe_token_accuracy.png")
    plt.savefig(out2)
    plt.savefig(out2.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {out2}")
    plt.close()

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
