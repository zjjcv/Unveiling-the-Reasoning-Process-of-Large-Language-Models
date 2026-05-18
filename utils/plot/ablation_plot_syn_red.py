"""
Ablation Study Plot - Paper Quality

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
from matplotlib import rcParams

# Publication quality settings
sns.set_theme(style="white")
rcParams.update({
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
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# Configuration (selected via CLI, default Qwen3-4B-Base)
if len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b':
    INPUT_CSV = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/ablation/head_ablation.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/Figure"
    TOTAL_HEADS = 768       # 48 layers × 16 heads
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
COLOR_SYNERGISTIC = '#00A087'  # GREEN - Synergistic First (High Syn-Red)
COLOR_REDUNDANT = '#E64B35'     # RED - Redundant First (Low Syn-Red)
COLOR_RANDOM = '#4DBBD5'        # BLUE - Random


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process ablation data."""
    print(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)

    # Convert to percentage
    if 'num_ablated' in df.columns:
        total_heads = TOTAL_HEADS
        df['pct_ablated'] = df['num_ablated'] / total_heads * 100

    print(f"  Total records: {len(df)}")
    print(f"  Strategies: {df['strategy'].unique()}")
    print(f"  Baseline accuracy: {df[df['num_ablated'] == 0]['accuracy'].mean():.3f}")

    return df


def smooth_curve(x, y, window=5):
    """Apply moving average smoothing."""
    if len(x) < window:
        return x, y

    x_smooth = x[window//2:-window//2+1] if window % 2 == 1 else x[window//2:-window//2]
    y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')

    # Ensure same length
    if len(x_smooth) > len(y_smooth):
        x_smooth = x_smooth[:len(y_smooth)]
    elif len(x_smooth) < len(y_smooth):
        y_smooth = y_smooth[:len(x_smooth)]

    return x_smooth, y_smooth


def plot_ablation_curves(df: pd.DataFrame, output_path: str):
    """Plot ablation curves with specified color scheme."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Get baseline accuracy
    baseline_acc = df[df['num_ablated'] == 0]['accuracy'].mean()

    # Plot each strategy
    for strategy, color, label, linewidth in [
        ('high_to_low', COLOR_SYNERGISTIC,
         'Synergistic First\n(High Syn-Red)', 3.0),
        ('random_run1', COLOR_RANDOM, 'Random', 2.5),
        ('low_to_high', COLOR_REDUNDANT,
         'Redundant First\n(Low Syn-Red)', 3.0),
    ]:
        strategy_data = df[df['strategy'] == strategy].copy()

        if len(strategy_data) == 0:
            continue

        # Sort by percentage ablated
        strategy_data = strategy_data.sort_values('pct_ablated')

        x = strategy_data['pct_ablated'].values
        y = strategy_data['accuracy'].values

        # For random, aggregate all runs
        if 'random' in strategy:
            all_random = df[df['strategy'].str.contains('random', case=False, na=False)]
            random_agg = all_random.groupby('pct_ablated')['accuracy'].agg(['mean', 'std']).reset_index()
            x = random_agg['pct_ablated'].values
            y_mean = random_agg['mean'].values
            y_std = random_agg['std'].values

            # Plot with error band
            ax.plot(x, y_mean, color=color, linewidth=linewidth,
                   label='Random', alpha=0.9)
            ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                           color=color, alpha=0.2, linewidth=0)
        else:
            # Plot single line with slight smoothing
            x_smooth, y_smooth = smooth_curve(x, y, window=3)
            ax.plot(x_smooth, y_smooth, color=color, linewidth=linewidth,
                   label=label, alpha=0.9, marker='o', markersize=4,
                   markevery=len(x_smooth)//10)

    # Style
    ax.set_xlabel('Percentage of Ablated Attention Heads (%)', fontsize=22)
    ax.set_ylabel('GSM8K Accuracy', fontsize=22)
    ax.set_title('Impact of Attention Head Ablation on Model Performance',
                 fontsize=24, pad=20)

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True,
             shadow=True, fontsize=16)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)

    # Add baseline reference line
    ax.axhline(y=baseline_acc, color='gray', linestyle=':',
              linewidth=1.5, alpha=0.5, label=f'Baseline ({baseline_acc:.2f})')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_ablation_curves_with_shaded_area(df: pd.DataFrame, output_path: str):
    """Plot ablation curves with shaded areas showing variance."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    baseline_acc = df[df['num_ablated'] == 0]['accuracy'].mean()

    # Aggregate data by strategy
    strategies_data = {
        'high_to_low': {'color': COLOR_SYNERGISTIC, 'label': 'Synergistic First'},
        'low_to_high': {'color': COLOR_REDUNDANT, 'label': 'Redundant First'},
    }

    # Get all random data
    random_data = df[df['strategy'].str.contains('random', case=False, na=False)]

    for strategy_key, config in strategies_data.items():
        strategy_df = df[df['strategy'] == strategy_key].sort_values('pct_ablated')

        if len(strategy_df) == 0:
            continue

        x = strategy_df['pct_ablated'].values
        y = strategy_df['accuracy'].values

        # Smooth the curve
        from scipy.interpolate import make_interp_spline
        from scipy.signal import savgol_filter

        # Use Savitzky-Golay filter for smoothing
        if len(y) >= 7:
            y_smooth = savgol_filter(y, window_length=min(7, len(y) if len(y) % 2 == 1 else len(y)-1), polyorder=2)
        else:
            y_smooth = y

        ax.plot(x, y_smooth, color=config['color'], linewidth=3.5,
               label=config['label'], alpha=0.95, marker='o',
               markersize=5, markevery=max(1, len(x)//12))

    # Plot random with error band
    if len(random_data) > 0:
        random_agg = random_data.groupby('pct_ablated')['accuracy'].agg(['mean', 'std']).reset_index()
        x_rand = random_agg['pct_ablated'].values
        y_rand_mean = random_agg['mean'].values
        y_rand_std = random_agg['std'].values

        # Smooth random curve
        if len(y_rand_mean) >= 7:
            y_rand_smooth = savgol_filter(y_rand_mean,
                                        window_length=min(7, len(y_rand_mean) if len(y_rand_mean) % 2 == 1 else len(y_rand_mean)-1),
                                        polyorder=2)
            y_rand_std_smooth = savgol_filter(y_rand_std,
                                            window_length=min(7, len(y_rand_std) if len(y_rand_std) % 2 == 1 else len(y_rand_std)-1),
                                            polyorder=2)
        else:
            y_rand_smooth = y_rand_mean
            y_rand_std_smooth = y_rand_std

        ax.plot(x_rand, y_rand_smooth, color=COLOR_RANDOM, linewidth=3,
               label='Random', alpha=0.9)
        ax.fill_between(x_rand, y_rand_smooth - y_rand_std_smooth,
                       y_rand_smooth + y_rand_std_smooth,
                       color=COLOR_RANDOM, alpha=0.25, linewidth=0)

    # Formatting
    ax.set_xlabel('Percentage of Ablated Attention Heads (%)', fontsize=24)
    ax.set_ylabel('GSM8K Accuracy', fontsize=24)
    ax.set_title('Ablation Study: Synergistic vs Redundant Attention Heads',
                 fontsize=26, pad=20)

    # Legend with custom location
    ax.legend(loc='upper right', frameon=True, fancybox=True,
             shadow=True, fontsize=18, framealpha=0.95)

    # Grid and limits
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)

    # Baseline
    ax.axhline(y=baseline_acc, color='gray', linestyle=':',
              linewidth=2, alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Ablation Study Visualization - Paper Quality")
    print("=" * 70)

    # Load data
    df = load_and_process_data(INPUT_CSV)

    # Generate plots
    print("\nGenerating plots...")

    output1 = os.path.join(OUTPUT_DIR, "ablation_curves_simple.png")
    plot_ablation_curves(df, output1)

    output2 = os.path.join(OUTPUT_DIR, "ablation_curves_smooth.png")
    plot_ablation_curves_with_shaded_area(df, output2)

    print("\n" + "=" * 70)
    print("Complete! Generated plots:")
    print(f"  1. {output1}")
    print(f"  2. {output2}")
    print("=" * 70)


if __name__ == "__main__":
    main()
