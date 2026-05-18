"""
Ablation Study Plot V2 - Custom Color Scheme

Color scheme:
- Synergistic First (High Syn-Red): RED  - fastest decline
- Random: BLUE - medium decline
- Redundant First (Low Syn-Red): GREEN - slowest decline
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

# Publication quality settings
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Noto Serif', 'DejaVu Serif', 'Times New Roman'],
    'font.size': 18,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# Configuration (selected via CLI, default Qwen3-4B-Base)
if len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b':
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/ablation/head_ablation.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure"
    TOTAL_HEADS = 768
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_8b_base':
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/ablation/head_ablation.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/Figure"
    TOTAL_HEADS = 1152
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/ablation/head_ablation.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/Figure"
    TOTAL_HEADS = 1600
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/ablation/head_ablation.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/Figure"
    TOTAL_HEADS = 272
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/ablation/head_ablation.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/Figure"
    TOTAL_HEADS = 1024
else:
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/ablation/head_ablation.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/Figure"
    TOTAL_HEADS = 1152

# Colors
COLOR_SYNERGISTIC = '#E64B35'  # RED
COLOR_REDUNDANT = '#00A087'     # GREEN
COLOR_RANDOM = '#4DBBD5'        # BLUE


def load_data(csv_path: str) -> pd.DataFrame:
    """Load ablation data."""
    df = pd.read_csv(csv_path)

    if 'num_ablated' in df.columns:
        df['pct_ablated'] = df['num_ablated'] / TOTAL_HEADS * 100

    return df


def plot_curves(df: pd.DataFrame, output_path: str):
    """Plot ablation curves."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    baseline = df[df['num_ablated'] == 0]['accuracy'].mean()

    # Synergistic First (RED - fastest decline)
    syn_data = df[df['strategy'] == 'high_to_low'].sort_values('pct_ablated')
    if len(syn_data) > 0:
        x = syn_data['pct_ablated'].values
        y = syn_data['accuracy'].values
        if len(y) >= 7:
            y = savgol_filter(y, 7, 2)
        ax.plot(x, y, color=COLOR_SYNERGISTIC, linewidth=3.5,
               label='Synergistic First', marker='o', markersize=5,
               markevery=max(1, len(x)//12))

    # Random (BLUE - medium decline)
    rand_data = df[df['strategy'].str.contains('random', case=False, na=False)]
    if len(rand_data) > 0:
        rand_agg = rand_data.groupby('pct_ablated')['accuracy'].agg(['mean', 'std']).reset_index()
        x = rand_agg['pct_ablated'].values
        y_mean = rand_agg['mean'].values
        y_std = rand_agg['std'].values
        if len(y_mean) >= 7:
            y_mean = savgol_filter(y_mean, 7, 2)
        ax.plot(x, y_mean, color=COLOR_RANDOM, linewidth=3, label='Random')
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                       color=COLOR_RANDOM, alpha=0.25)

    # Redundant First (GREEN - slowest decline)
    red_data = df[df['strategy'] == 'low_to_high'].sort_values('pct_ablated')
    if len(red_data) > 0:
        x = red_data['pct_ablated'].values
        y = red_data['accuracy'].values
        if len(y) >= 7:
            y = savgol_filter(y, 7, 2)
        ax.plot(x, y, color=COLOR_REDUNDANT, linewidth=3.5,
               label='Redundant First', marker='o', markersize=5,
               markevery=max(1, len(x)//12))

    ax.set_xlabel('Percentage of Ablated Attention Heads (%)', fontsize=24)
    ax.set_ylabel('GSM8K Accuracy', fontsize=24)
    ax.set_title('Ablation Study: Synergistic vs Redundant Attention Heads',
                 fontsize=26, pad=20)

    ax.legend(loc='upper right', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=baseline, color='gray', linestyle=':', linewidth=2, alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    df = load_data(INPUT_CSV)
    output = os.path.join(OUTPUT_DIR, "ablation_curves_v2.png")
    plot_curves(df, output)
    print(f"Done! Saved to {output}")


if __name__ == "__main__":
    main()
