"""
Syn vs Red Rank Scatter — from pairwise CSV.

Reads pairwise syn/red CSV, aggregates to per-head level,
ranks by Syn and Red separately, plots scatter.

Usage:
    python utils/plot/syn_red_scatter_pairwise.py                  # Qwen3-8B-Base
    python utils/plot/syn_red_scatter_pairwise.py qwen3_4b_base
    python utils/plot/syn_red_scatter_pairwise.py qwen3_14b_base
    python utils/plot/syn_red_scatter_pairwise.py gemma3_4b_base
    python utils/plot/syn_red_scatter_pairwise.py llama3_8b
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# ── Global style ─────────────────────────────────────────────────────
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Noto Serif', 'DejaVu Serif', 'Times New Roman',
                    'Liberation Serif'],
    'font.size': 20,
    'font.weight': 'bold',
    'axes.labelsize': 24,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
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

# ── Paths (selected via CLI) ────────────────────────────────────────
if len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/Figure"
    MODEL_LABEL = "Qwen3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/Figure"
    MODEL_LABEL = "Qwen3-14B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure"
    MODEL_LABEL = "Gemma3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b':
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure"
    MODEL_LABEL = "Gemma3-12B-Instruct"
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/Figure"
    MODEL_LABEL = "Llama-3.1-8B"
else:
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/Figure"
    MODEL_LABEL = "Qwen3-8B-Base"


def aggregate_to_per_head(csv_path):
    """Load pairwise CSV, average across questions, aggregate to per-head Syn/Red."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded pairwise CSV: {len(df)} rows")

    # Average across questions
    df_avg = df.groupby(['layer_1', 'head_1', 'layer_2', 'head_2']).agg({
        'syn': 'mean', 'red': 'mean'
    }).reset_index()

    # Aggregate to per-head level
    syn_by_head, red_by_head = {}, {}
    for _, row in df_avg.iterrows():
        for (l, h) in [(row['layer_1'], row['head_1']),
                        (row['layer_2'], row['head_2'])]:
            syn_by_head.setdefault((l, h), []).append(row['syn'])
            red_by_head.setdefault((l, h), []).append(row['red'])

    records = []
    for (l, h) in sorted(syn_by_head.keys()):
        records.append({
            'Layer': l, 'Head': h,
            'Syn': np.mean(syn_by_head[(l, h)]),
            'Red': np.mean(red_by_head[(l, h)]),
        })

    head_df = pd.DataFrame(records)
    head_df['Syn_Red_Diff'] = head_df['Syn'] - head_df['Red']
    head_df['Syn_Rank'] = head_df['Syn'].rank(ascending=False, method='min')
    head_df['Red_Rank'] = head_df['Red'].rank(ascending=False, method='min')
    head_df['Diff_Rank'] = head_df['Syn_Red_Diff'].rank(ascending=False, method='min')

    print(f"  Total heads: {len(head_df)}")
    return head_df


def main():
    print("=" * 60)
    print(f"Syn vs Red Rank Scatter — {MODEL_LABEL}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    head_df = aggregate_to_per_head(PAIRWISE_PATH)

    # Correlation: Syn_Rank vs Diff_Rank, Red_Rank vs Diff_Rank
    sr_syn, sp_syn = spearmanr(head_df['Syn_Rank'], head_df['Diff_Rank'])
    sr_red, sp_red = spearmanr(head_df['Red_Rank'], head_df['Diff_Rank'])
    print(f"  Syn_Rank  vs Diff_Rank: Spearman r={sr_syn:.4f} (p={sp_syn:.2e})")
    print(f"  Red_Rank  vs Diff_Rank: Spearman r={sr_red:.4f} (p={sp_red:.2e})")

    # Plot 1×2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    y_max = head_df['Diff_Rank'].max() + 10
    x_max = max(head_df['Syn_Rank'].max(), head_df['Red_Rank'].max()) + 10

    # Left: Syn_Rank vs Diff_Rank
    ax1.scatter(head_df['Syn_Rank'], head_df['Diff_Rank'],
                s=18, alpha=0.5, color='#888888', edgecolors='none', zorder=3)
    ax1.plot([0, x_max], [0, x_max], '--', color='#BBBBBB', linewidth=1.2, zorder=2)
    ax1.set_xlabel('Synergy Rank')
    ax1.set_ylabel('Syn-Red Diff Rank')
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, y_max)
    ax1.grid(True, linestyle='-', alpha=0.12, linewidth=0.6, color='#888888')
    ax1.set_axisbelow(True)
    ax1.text(0.05, 0.95,
             f'$\\rho$ = {sr_syn:.3f}\n$p$ = {sp_syn:.1e}',
             transform=ax1.transAxes, fontsize=20,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#CCCCCC', alpha=0.9))

    # Right: Red_Rank vs Diff_Rank
    ax2.scatter(head_df['Red_Rank'], head_df['Diff_Rank'],
                s=18, alpha=0.5, color='#888888', edgecolors='none', zorder=3)
    ax2.plot([0, x_max], [0, x_max], '--', color='#BBBBBB', linewidth=1.2, zorder=2)
    ax2.set_xlabel('Redundancy Rank')
    ax2.set_ylabel('Syn-Red Diff Rank')
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, y_max)
    ax2.grid(True, linestyle='-', alpha=0.12, linewidth=0.6, color='#888888')
    ax2.set_axisbelow(True)
    ax2.text(0.05, 0.95,
             f'$\\rho$ = {sr_red:.3f}\n$p$ = {sp_red:.1e}',
             transform=ax2.transAxes, fontsize=20,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#CCCCCC', alpha=0.9))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "syn_red_rank_scatter.png")
    plt.savefig(out_path)
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {out_path}")
    plt.close()

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
