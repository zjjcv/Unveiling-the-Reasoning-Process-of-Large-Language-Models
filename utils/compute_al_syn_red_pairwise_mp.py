"""
Compute pairwise Synergy-Redundancy for GSM8K/ARC datasets using multiprocessing.

Reads L2 norm time series from proxy collection, computes
PhiID-based synergy and redundancy for all pairs of attention heads/layers,
using multiprocessing at the PAIR level for maximum parallelism.

Input: gsm8k_{proxy_type}.csv or arc_{difficulty}_{proxy_type}.csv (L2 norm time series)
Output: {proxy_type}_syn_red_pairwise.csv (pairwise syn/red for each question)

Usage:
    python utils/compute_al_syn_red_pairwise_mp.py                    # Gemma3-12B-IT ARC
    python utils/compute_al_syn_red_pairwise_mp.py gemma3_12b_base    # Gemma3-12B-Base GSM8K
    python utils/compute_al_syn_red_pairwise_mp.py gemma3_4b_base     # Gemma3-4B-Base GSM8K
    python utils/compute_al_syn_red_pairwise_mp.py gemma3_4b_it       # Gemma-3-4B-Instruct GSM8K
    python utils/compute_al_syn_red_pairwise_mp.py qwen3_8b_base      # Qwen3-8B-Base GSM8K
    python utils/compute_al_syn_red_pairwise_mp.py qwen3_8b_base_arc  # Qwen3-8B-Base ARC
    python utils/compute_al_syn_red_pairwise_mp.py qwen3_4b_base      # Qwen3-4B-Base GSM8K
    python utils/compute_al_syn_red_pairwise_mp.py qwen3_14b_base     # Qwen3-14B-Base GSM8K
    python utils/compute_al_syn_red_pairwise_mp.py llama3_8b          # Llama-3.1-8B GSM8K
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from multiprocessing import Pool, cpu_count

# Try to import PhiID library
try:
    from phyid.calculate import calc_PhiID
except ImportError:
    print("Error: phyid module not found. Please install integrated-info-decomp:")
    print("   pip install -e /path/to/integrated-info-decomp/")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================
if len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b_base':
    INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/data")
    OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/data")
    DATASET_TYPE = os.environ.get("DATASET_TYPE", "GSM8K")
    _MODEL_LABEL = "Gemma3-12B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data")
    OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data")
    DATASET_TYPE = os.environ.get("DATASET_TYPE", "GSM8K")
    _MODEL_LABEL = "Gemma3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_it':
    INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data")
    OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data")
    DATASET_TYPE = os.environ.get("DATASET_TYPE", "GSM8K")
    _MODEL_LABEL = "Gemma-3-4B-Instruct"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_8b_base':
    INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data")
    OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data")
    DATASET_TYPE = os.environ.get("DATASET_TYPE", "GSM8K")
    _MODEL_LABEL = "Qwen3-8B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_8b_base_arc':
    INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data")
    OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data")
    DATASET_TYPE = os.environ.get("DATASET_TYPE", "ARC")
    _MODEL_LABEL = "Qwen3-8B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
    INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data")
    OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data")
    DATASET_TYPE = os.environ.get("DATASET_TYPE", "GSM8K")
    _MODEL_LABEL = "Qwen3-4B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
    INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data")
    OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data")
    DATASET_TYPE = os.environ.get("DATASET_TYPE", "GSM8K")
    _MODEL_LABEL = "Qwen3-14B-Base"
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data")
    OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data")
    DATASET_TYPE = os.environ.get("DATASET_TYPE", "GSM8K")
    _MODEL_LABEL = "Llama-3.1-8B"
else:
    INPUT_BASE_DIR = os.environ.get("INPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data")
    OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data")
    DATASET_TYPE = os.environ.get("DATASET_TYPE", "ARC")
    _MODEL_LABEL = "Gemma3-12B-IT"

# GSM8K specific
DATASET_NAME = "L2_Norm"

# ARC specific (difficulties)
DIFFICULTIES = ['easy', 'challenge']

# Multiprocessing Configuration
N_WORKERS = 100
CHUNK_SIZE = 500  # Number of pairs per chunk submitted to pool
TAU = 1
KIND = "gaussian"
REDUNDANCY = "MMI"

# ============================================================
# Global shared data — written once before forking, read by all workers via fork CoW
# ============================================================
_shared_ts_arrays = {}  # (q_idx, comp_idx) -> numpy array
_shared_components = []  # list of component UIDs


# ============================================================
# PhiID Computation
# ============================================================

def _compute_pair_worker(task):
    """Worker: compute PhiID for a batch of pairs for one question.

    Args:
        task: (q_idx, pair_indices, proxy_type)
            pair_indices: list of (comp_idx_1, comp_idx_2)

    Returns:
        list of result tuples
    """
    q_idx, pair_indices, proxy_type = task
    results = []
    ts_dict = _shared_ts_arrays  # read shared data

    for ci1, ci2 in pair_indices:
        ts1 = ts_dict.get((q_idx, ci1))
        ts2 = ts_dict.get((q_idx, ci2))

        if ts1 is None or ts2 is None:
            if proxy_type == 'al':
                results.append((
                    _shared_components[ci1][0], _shared_components[ci1][1],
                    _shared_components[ci2][0], _shared_components[ci2][1],
                    np.nan, np.nan))
            else:
                results.append((
                    _shared_components[ci1], _shared_components[ci2],
                    np.nan, np.nan))
            continue

        min_len = min(len(ts1), len(ts2))
        if min_len < 5:
            if proxy_type == 'al':
                results.append((
                    _shared_components[ci1][0], _shared_components[ci1][1],
                    _shared_components[ci2][0], _shared_components[ci2][1],
                    np.nan, np.nan))
            else:
                results.append((
                    _shared_components[ci1], _shared_components[ci2],
                    np.nan, np.nan))
            continue

        t1, t2 = ts1[:min_len], ts2[:min_len]
        try:
            atoms_res, _ = calc_PhiID(t1, t2, tau=TAU, kind=KIND, redundancy=REDUNDANCY)
            syn = float(np.nanmean(np.asarray(atoms_res["sts"])))
            red = float(np.nanmean(np.asarray(atoms_res["rtr"])))
        except Exception:
            syn, red = np.nan, np.nan

        if proxy_type == 'al':
            results.append((
                _shared_components[ci1][0], _shared_components[ci1][1],
                _shared_components[ci2][0], _shared_components[ci2][1],
                syn, red))
        else:
            results.append((
                _shared_components[ci1], _shared_components[ci2],
                syn, red))

    return (q_idx, results)


def compute_pairwise_for_proxy_type(proxy_type: str, difficulty=None):
    """Compute pairwise syn/red for one proxy type for all questions."""
    global _shared_ts_arrays, _shared_components

    if DATASET_TYPE == "ARC" and difficulty:
        print(f"\n{'=' * 60}")
        print(f"Processing ARC-{difficulty.capitalize()} - {proxy_type.upper()}")
        print(f"{'=' * 60}")

        difficulty_dir = os.path.join(OUTPUT_BASE_DIR, difficulty)
        os.makedirs(difficulty_dir, exist_ok=True)

        input_file = os.path.join(INPUT_BASE_DIR, difficulty, f"arc_{difficulty}_{proxy_type}.csv")
        output_file = os.path.join(difficulty_dir, f"{proxy_type}_syn_red_pairwise.csv")
        effective_lengths_file = os.path.join(INPUT_BASE_DIR, difficulty, f"arc_{difficulty}_effective_lengths.csv")
    else:
        print(f"\n{'=' * 60}")
        print(f"Processing GSM8K - {proxy_type.upper()}")
        print(f"{'=' * 60}")

        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

        input_file = os.path.join(INPUT_BASE_DIR, DATASET_NAME, f"gsm8k_{proxy_type}.csv")
        output_file = os.path.join(OUTPUT_BASE_DIR, DATASET_NAME, f"{proxy_type}_syn_red_pairwise.csv")
        effective_lengths_file = os.path.join(INPUT_BASE_DIR, DATASET_NAME, f"gsm8k_effective_lengths.csv")

    if not os.path.exists(input_file):
        print(f"Warning: Input file not found: {input_file}")
        return

    # Load input data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)

    # Load effective lengths
    if os.path.exists(effective_lengths_file):
        effective_lengths_df = pd.read_csv(effective_lengths_file)
        effective_lengths_dict = dict(zip(effective_lengths_df['question_id'], effective_lengths_df['effective_length']))
        print(f"Loaded effective lengths for {len(effective_lengths_dict)} questions")
    else:
        print("Warning: Effective lengths file not found, using all steps")
        max_steps = len([c for c in df.columns if c.startswith('step_')])
        effective_lengths_dict = {qid: max_steps for qid in df['question_id'].unique()}

    # Get unique questions and components
    question_ids = sorted(df['question_id'].unique())

    if proxy_type == 'al':
        components = sorted(set(zip(df['layer'], df['head'])))
    else:
        components = sorted(df['layer'].unique())

    comp_to_idx = {c: i for i, c in enumerate(components)}
    n_components = len(components)
    n_pairs = n_components * (n_components - 1) // 2
    total_pairs = len(question_ids) * n_pairs

    print(f"Questions: {len(question_ids)}")
    print(f"Components: {n_components}")
    print(f"Pairs per question: {n_pairs:,}")
    print(f"Total pairs: {total_pairs:,}")

    # ---- Pre-extract numpy arrays (shared via fork CoW) ----
    print("\nPre-extracting time series arrays...")
    _shared_ts_arrays.clear()
    _shared_components = components

    for q_local_idx, q_id in enumerate(question_ids):
        q_df = df[df['question_id'] == q_id]
        actual_steps = len([c for c in q_df.columns if c.startswith('step_')])
        eff_len = min(effective_lengths_dict.get(q_id, actual_steps), actual_steps)
        step_cols = [f'step_{i+1}' for i in range(eff_len)]

        for comp in components:
            if proxy_type == 'al':
                layer, head = comp
                comp_df = q_df[(q_df['layer'] == layer) & (q_df['head'] == head)]
            else:
                comp_df = q_df[q_df['layer'] == comp]

            if len(comp_df) > 0:
                ts = comp_df[step_cols].values.flatten().astype(np.float64)
            else:
                ts = np.zeros(eff_len, dtype=np.float64)

            _shared_ts_arrays[(q_local_idx, comp_to_idx[comp])] = ts

    # ---- Build chunked tasks: (q_idx, [(ci1, ci2), ...], proxy_type) ----
    print("Building pair-level tasks...")
    all_pair_indices = list(combinations(range(n_components), 2))

    tasks = []
    for q_local_idx in range(len(question_ids)):
        # Split pairs for this question into chunks
        for chunk_start in range(0, len(all_pair_indices), CHUNK_SIZE):
            chunk = all_pair_indices[chunk_start:chunk_start + CHUNK_SIZE]
            tasks.append((q_local_idx, chunk, proxy_type))

    print(f"Total tasks (chunks): {len(tasks):,}  (chunk_size={CHUNK_SIZE})")

    # ---- Process in parallel ----
    print(f"\nProcessing with {N_WORKERS} workers...")

    all_results = []
    with Pool(N_WORKERS) as pool:
        for q_idx, chunk_results in tqdm(
                pool.imap_unordered(_compute_pair_worker, tasks),
                total=len(tasks),
                desc="Computing pairs"):
            all_results.extend([(q_idx,) + r for r in chunk_results])

    # ---- Save results ----
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    # Map local q_idx back to original question_id
    qid_map = {i: qid for i, qid in enumerate(question_ids)}
    for i in range(len(all_results)):
        q_local, *rest = all_results[i]
        all_results[i] = (qid_map[q_local],) + tuple(rest)

    if proxy_type == 'al':
        result_df = pd.DataFrame(all_results, columns=[
            'question_id', 'layer_1', 'head_1', 'layer_2', 'head_2', 'syn', 'red'
        ])
        result_df = result_df.sort_values(['question_id', 'layer_1', 'head_1', 'layer_2', 'head_2'])
    else:
        result_df = pd.DataFrame(all_results, columns=[
            'question_id', 'layer_1', 'layer_2', 'syn', 'red'
        ])
        result_df = result_df.sort_values(['question_id', 'layer_1', 'layer_2'])

    result_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")
    print(f"Total records: {len(result_df):,}")

    valid_syn = result_df['syn'].dropna()
    valid_red = result_df['red'].dropna()
    print(f"\nSummary statistics:")
    print(f"  Syn - mean: {valid_syn.mean():.6f}, std: {valid_syn.std():.6f}")
    print(f"  Red - mean: {valid_red.mean():.6f}, std: {valid_red.std():.6f}")
    print(f"  Valid pairs: {len(valid_syn):,} / {len(result_df):,}")

    # Free shared data
    _shared_ts_arrays.clear()
    _shared_components = []


def main():
    print("=" * 60)
    print(f"Pairwise Syn-Red Computation ({_MODEL_LABEL}) - {DATASET_TYPE}")
    print("=" * 60)
    print(f"\nWorkers: {N_WORKERS}")
    print(f"CPU cores: {cpu_count()}")

    proxy_types = ['al', 'ml', 'al_plus_ml']

    if DATASET_TYPE == "ARC":
        for difficulty in DIFFICULTIES:
            print(f"\n{'#'*60}")
            print(f"# Processing ARC-{difficulty.capitalize()}")
            print(f"{'#'*60}")

            for proxy_type in proxy_types:
                compute_pairwise_for_proxy_type(proxy_type, difficulty=difficulty)

        print("\n" + "=" * 60)
        print("All Done!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_BASE_DIR}")
        print("\nGenerated files for each difficulty (easy/challenge):")
        print("  - al_syn_red_pairwise.csv")
        print("  - ml_syn_red_pairwise.csv")
        print("  - al_plus_ml_syn_red_pairwise.csv")
    else:
        print(f"\n{'#'*60}")
        print(f"# Processing GSM8K")
        print(f"{'#'*60}")

        for proxy_type in proxy_types:
            compute_pairwise_for_proxy_type(proxy_type)

        print("\n" + "=" * 60)
        print("All Done!")
        print("=" * 60)
        print(f"\nOutput directory: {OUTPUT_BASE_DIR}")
        print("\nGenerated files:")
        print("  - al_syn_red_pairwise.csv")
        print("  - ml_syn_red_pairwise.csv")
        print("  - al_plus_ml_syn_red_pairwise.csv")


if __name__ == "__main__":
    main()
