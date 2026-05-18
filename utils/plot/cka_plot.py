"""
Plot CKA Similarity vs layer index — Baseline.

Shows that middle layers (synergistic core) extract abstract,
vocabulary-invariant rule representations more strongly than
early/late layers.

Usage:
    python utils/plot/cka_plot.py                  # Qwen3-8B-Base, RBF
    python utils/plot/cka_plot.py --linear         # Linear kernel
    python utils/plot/cka_plot.py qwen3_4b_base
    python utils/plot/cka_plot.py qwen3_14b_base
    python utils/plot/cka_plot.py gemma3_4b_base
    python utils/plot/cka_plot.py gemma3_12b_it
    python utils/plot/cka_plot.py llama3_8b
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

# ── Kernel selection ─────────────────────────────────────────────────
_USE_LINEAR = '--linear' in sys.argv
if _USE_LINEAR:
    sys.argv.remove('--linear')
_CKA_TAG = "linear" if _USE_LINEAR else "rbf"
_CKA_LABEL = "Linear" if _USE_LINEAR else "RBF"

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

_COLOR = '#888888'


def main():
    print("=" * 60)
    print(f"CKA Plot ({_CKA_LABEL}) — {MODEL_LABEL}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, f"cka_baseline_{_CKA_TAG}.csv")
    if not os.path.exists(csv_path):
        print(f"\n  Error: No data found: {csv_path}")
        print(f"  Run: python src/cka_aba_rbf.py [--linear] [model]")
        return

    df = pd.read_csv(csv_path)
    core_start = int(df['core_start'].iloc[0])
    core_end = int(df['core_end'].iloc[0])
    # Override with correct CLI values (CSV may contain stale core range)
    core_start, core_end = CORE_LAYERS

    print(f"\n  Loaded: {len(df)} layers, core: {core_start}-{core_end}")
    print(f"  CKA range: [{df['cka'].min():.4f}, {df['cka'].max():.4f}]")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 7))

    # Core region
    ax.axvspan(core_start - 0.5, core_end + 0.5,
               alpha=0.08, color='#999999', zorder=0, label='Synergistic Core')
    for x in [core_start - 0.5, core_end + 0.5]:
        ax.axvline(x=x, color='#555555', linestyle='--', linewidth=1.5,
                    alpha=0.6, zorder=1)

    mkws = dict(markersize=8, markeredgewidth=1.5, markeredgecolor='white')

    ax.plot(df['layer'], df['cka'],
            '-o', color=_COLOR, linewidth=2.8, zorder=3,
            label='CKA Similarity', **mkws)

    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA Similarity')
    ax.set_xlim(-0.5, df['layer'].max() + 0.5)
    ax.set_ylim(0, 1.05)

    ax.grid(True, axis='y', linestyle='-', alpha=0.12, linewidth=0.6,
            color='#888888')
    ax.set_axisbelow(True)

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.0, -0.18, 1.0, 0.0),
                    ncol=2, framealpha=0.95, edgecolor='#CCCCCC',
                    fancybox=True, shadow=False, borderpad=0.6,
                    handletextpad=0.6, columnspacing=2.5,
                    fontsize=20, mode='expand',
                    borderaxespad=0.0)
    leg.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.22)
    output_path = os.path.join(OUTPUT_DIR, f"cka_baseline_{_CKA_TAG}.png")
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()

    peak_val = df['cka'].max()
    peak_layer = df['cka'].idxmax()
    core_mean = df.loc[(df['layer'] >= core_start) &
                        (df['layer'] <= core_end), 'cka'].mean()
    print(f"\n  Peak: {peak_val:.4f} @ L{peak_layer}")
    print(f"  Core mean: {core_mean:.4f}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
