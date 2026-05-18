#!/usr/bin/env python3
"""Replot ablation_results from existing head_ablation.csv as PDF."""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

MODELS = {
    "gemma3_12b": {
        "dir": "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/ablation",
        "label": "Gemma-3-12B-Instruct",
    },
    "llama3_8b": {
        "dir": "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/ablation",
        "label": "Llama-3.1-8B",
    },
    "qwen3_8b_base": {
        "dir": "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/ablation",
        "label": "Qwen3-8B-Base",
    },
    "qwen3_4b_base": {
        "dir": "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/ablation",
        "label": "Qwen3-4B-Base",
    },
    "qwen3_14b_base": {
        "dir": "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/ablation",
        "label": "Qwen3-14B-Base",
    },
}


def replot(model_key):
    cfg = MODELS[model_key]
    csv_path = os.path.join(cfg["dir"], "head_ablation.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP {model_key}: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for strategy in ["high_to_low", "low_to_high"]:
        if strategy not in df["strategy"].values:
            continue
        data = df[df["strategy"] == strategy]
        if strategy == "high_to_low":
            label, color = "Synergistic First (High→Low)", "#d62728"
        else:
            label, color = "Redundant First (Low→High)", "#1f77b4"
        ax.plot(data["num_ablated"], data["accuracy"],
                label=label, marker="o", color=color, linewidth=2)

    random_strategies = [s for s in df["strategy"].unique() if "random" in s]
    if random_strategies:
        random_data = df[df["strategy"].isin(random_strategies)]
        stats = random_data.groupby("num_ablated").agg({"accuracy": ["mean", "std"]}).reset_index()
        stats.columns = ["num_ablated", "mean", "std"]
        ax.plot(stats["num_ablated"], stats["mean"],
                label="Random (mean)", marker="o", color="gray", alpha=0.7)
        ax.fill_between(stats["num_ablated"],
                        stats["mean"] - stats["std"],
                        stats["mean"] + stats["std"],
                        color="gray", alpha=0.2)

    ax.set_xlabel("Number of Ablated Attention Heads", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy", fontsize=12)
    ax.set_title(f"Attention Head Ablation Study — {cfg['label']}", fontsize=14, pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0, df["num_ablated"].max())
    ax.set_ylim(0, 1)

    plt.tight_layout()
    pdf_path = os.path.join(cfg["dir"], "ablation_results.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {pdf_path}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        keys = [sys.argv[1]]
    else:
        keys = list(MODELS.keys())
    for k in keys:
        print(f"Replotting ablation for {k}...")
        replot(k)
