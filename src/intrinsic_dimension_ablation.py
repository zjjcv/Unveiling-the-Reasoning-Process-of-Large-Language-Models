"""
Intrinsic Dimension with Head Ablation — k-NN MLE, GPU-accelerated.

Computes per-layer intrinsic dimension of hidden states under three conditions:
  1. Baseline (no ablation)
  2. Synergistic heads ablated (per-layer top 30% by Syn_Red_Diff)
  3. Redundant heads ablated (per-layer bottom 30% by Syn_Red_Diff)

Each condition independently generates responses via model.generate(),
then computes k-NN MLE intrinsic dimension on hidden states.

Method:
  - model.generate() per condition, collect hidden states at last position per step
  - Ablation via forward pre-hooks on o_proj (zero selected heads' activations)
  - Layer-matched head selection: per-layer top/bottom 30%, skip first/last layer
  - k-NN MLE with k=20

Usage:
    python src/intrinsic_dimension_ablation.py                   # Qwen3-8B-Base
    python src/intrinsic_dimension_ablation.py qwen3_4b_base     # Qwen3-4B-Base
    python src/intrinsic_dimension_ablation.py qwen3_14b_base    # Qwen3-14B-Base
    python src/intrinsic_dimension_ablation.py gemma3_4b_base    # Gemma3-4B-Base
    python src/intrinsic_dimension_ablation.py gemma3_12b_it     # Gemma3-12B-IT
    python src/intrinsic_dimension_ablation.py llama3_8b         # Llama-3.1-8B
"""

import os
import sys
import json
import random
import gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============================================================
# Configuration (selected via CLI)
# ============================================================
if len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_4B_Base"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/intrinsic_dimension"
    _MODEL_LABEL = "Qwen3-4B-Base"
    _TORCH_DTYPE = torch.bfloat16
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_14B_Base"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/intrinsic_dimension"
    _MODEL_LABEL = "Qwen3-14B-Base"
    _TORCH_DTYPE = torch.float16
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-4B-Base"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/intrinsic_dimension"
    _MODEL_LABEL = "Gemma3-4B-Base"
    _TORCH_DTYPE = torch.bfloat16
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b_it':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-12B-IT"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/intrinsic_dimension"
    _MODEL_LABEL = "Gemma3-12B-IT"
    _TORCH_DTYPE = torch.bfloat16
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Llama-3.1-8B"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/intrinsic_dimension"
    _MODEL_LABEL = "Llama-3.1-8B"
    _TORCH_DTYPE = torch.bfloat16
else:
    MODEL_PATH = "/data/zjj/Synergistic_Core/Qwen-3-8B-base"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/intrinsic_dimension"
    _MODEL_LABEL = "Qwen3-8B-Base"
    _TORCH_DTYPE = torch.bfloat16

GSM8K_DATA_DIR = "/data/zjj/Synergistic_Core/data/gsm8k"
NUM_SAMPLES = 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 30
_K = 20
_ABLATE_FRACTION = 0.30  # per-layer top/bottom 30%

# 1-shot example
_ONE_SHOT_PREFIX = """Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: Janet sells 16 - 3 - 4 = 9 duck eggs a day.
She makes 9 * 2 = $18 every day at the farmer's market.
#### 18

Question: """


# ============================================================
# Intrinsic Dimension Computation (k-NN MLE, GPU)
# ============================================================
def compute_knn_mle_id_gpu(matrix_np, k=20, device='cuda'):
    """k-NN MLE intrinsic dimension using GPU."""
    n = matrix_np.shape[0]
    if n < k + 1:
        return 0.0

    matrix = torch.tensor(matrix_np, dtype=torch.float32, device=device)

    # Pairwise L2 distance [n, n]
    dist = torch.cdist(matrix, matrix)
    dist.fill_diagonal_(float('inf'))

    # Sort ascending
    sorted_dist, _ = torch.sort(dist, dim=1)

    # Filter degenerate points
    valid = sorted_dist[:, 0] > 1e-10
    if valid.sum() < k + 1:
        return 0.0

    sd = sorted_dist[valid]

    r_k = sd[:, k - 1].clamp(min=1e-12)       # [n_valid]
    r_j = sd[:, :k - 1].clamp(min=1e-12)       # [n_valid, k-1]

    log_ratios = torch.log(r_k.unsqueeze(1) / r_j)
    sum_log = log_ratios.sum(dim=1)

    id_per_point = (k - 1) / sum_log
    return max(id_per_point.mean().item(), 0.0)


# ============================================================
# Model Helpers
# ============================================================
def get_attn(model, layer_idx):
    """Get self_attn module of a specific decoder layer (Gemma3 & Qwen3)."""
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            return lm.model.layers[layer_idx].self_attn
        elif hasattr(lm, 'layers'):
            return lm.layers[layer_idx].self_attn
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx].self_attn
    raise ValueError(f"Cannot find layer {layer_idx}")


def get_num_heads(model):
    """Auto-detect number of attention heads from config."""
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    return getattr(config, 'num_attention_heads', 32)


def load_pairwise_and_rank(csv_path):
    """Load pairwise syn/red CSV and compute per-head Syn-Red diff."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded pairwise CSV: {len(df)} rows")

    # Average across questions
    df_avg = df.groupby(['layer_1', 'head_1', 'layer_2', 'head_2']).agg({
        'syn': 'mean', 'red': 'mean'
    }).reset_index()

    # Aggregate to per-head level
    syn_by_head, red_by_head = {}, {}
    for _, row in df_avg.iterrows():
        for (l, h) in [(row['layer_1'], row['head_1']), (row['layer_2'], row['head_2'])]:
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

    print(f"  Total heads: {len(head_df)}")
    print(f"  Syn_Red_Diff range: [{head_df['Syn_Red_Diff'].min():.4f}, "
          f"{head_df['Syn_Red_Diff'].max():.4f}]")
    return head_df


# ============================================================
# Layer-Matched Head Selection
# ============================================================
def select_layer_matched_heads(head_df, num_layers, fraction=0.3):
    """Per-layer top/bottom fraction by Syn_Red_Diff, skip first/last layer."""
    syn_per_layer = {}   # {layer_idx: [head_indices]}
    red_per_layer = {}

    for l in range(1, num_layers - 1):
        layer_heads = head_df[head_df['Layer'] == l].copy()
        if len(layer_heads) == 0:
            continue
        n = max(1, int(len(layer_heads) * fraction))

        syn_per_layer[l] = layer_heads.nlargest(n, 'Syn_Red_Diff')['Head'].tolist()
        red_per_layer[l] = layer_heads.nsmallest(n, 'Syn_Red_Diff')['Head'].tolist()

    return syn_per_layer, red_per_layer


# ============================================================
# Hook-Based Attention Head Ablation (o_proj pre-hook)
# ============================================================
def register_ablation_hooks(model, heads_per_layer, num_heads):
    """Register pre-hooks on o_proj to zero ablated head activations."""
    hooks = []
    for layer_idx, head_indices in heads_per_layer.items():
        attn = get_attn(model, layer_idx)
        head_dim = attn.o_proj.in_features // num_heads
        ablated = set(int(h) for h in head_indices)

        def make_prehook(ablated_set, hd):
            def prehook(module, args):
                x = args[0]  # [batch, seq, num_heads * head_dim]
                for h in ablated_set:
                    x[:, :, h * hd:(h + 1) * hd] = 0
                return (x,) + args[1:] if len(args) > 1 else (x,)
            return prehook

        hooks.append(attn.o_proj.register_forward_pre_hook(make_prehook(ablated, head_dim)))
    return hooks


def remove_hooks(hooks):
    """Remove all registered hooks."""
    for h in hooks:
        h.remove()


# ============================================================
# Hidden State Collection via model.generate()
# ============================================================
def collect_hidden_states(model, tokenizer, samples, num_layers, desc="Generating"):
    """Run generation and collect hidden states per layer.

    Each question generates independently via model.generate().
    Collects hidden_states[l+1][0, -1, :] at each generation step.
    """
    device = next(model.parameters()).device

    # Stop tokens
    eos_token_id = tokenizer.eos_token_id
    stop_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id) if eos_token_id else []
    try:
        if '<end_of_turn>' in tokenizer.get_vocab():
            eid = tokenizer.convert_tokens_to_ids('<end_of_turn>')
            if eid not in stop_ids:
                stop_ids.append(eid)
    except Exception:
        pass
    if not stop_ids:
        stop_ids = [1, 106]

    layer_vectors = {l: [] for l in range(num_layers)}
    total_tokens = 0

    for q_idx, sample in enumerate(tqdm(samples, desc=f"  {desc}")):
        prompt = _ONE_SHOT_PREFIX + sample['question'] + "\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_ids,
                use_cache=True,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        for step_idx, step_hs in enumerate(outputs.hidden_states):
            for l in range(num_layers):
                vec = step_hs[l + 1][0, -1, :].detach().cpu().float().numpy()
                layer_vectors[l].append(vec)

        eff_len = len(outputs.sequences[0]) - input_len
        total_tokens += eff_len

        if q_idx == 0:
            gen_text = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)
            print(f"    [Q0] {eff_len} tokens: {gen_text[:200]}")

    print(f"  Total tokens collected: {total_tokens}")
    return layer_vectors


def compute_ids_per_layer(layer_vectors, num_layers):
    """Compute k-NN MLE ID per layer from collected vectors."""
    results = []
    for l in range(num_layers):
        mat = np.stack(layer_vectors[l], axis=0)
        id_val = compute_knn_mle_id_gpu(mat, k=_K, device='cuda')
        results.append(id_val)
        print(f"    Layer {l:2d}: ID = {id_val:8.2f}  ({mat.shape[0]} vectors)")
    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print(f"Intrinsic Dimension with Layer-Matched Ablation — k-NN MLE (k={_K})")
    print(f"  Model: {_MODEL_LABEL}")
    print(f"  Questions: {NUM_SAMPLES}, Max tokens/Q: {MAX_NEW_TOKENS}")
    print(f"  Ablation: per-layer top/bottom {_ABLATE_FRACTION * 100:.0f}% (Syn_Red_Diff)")
    print(f"  Method: model.generate() + o_proj pre-hook ablation")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load GSM8K
    random.seed(RANDOM_SEED)
    test_file = os.path.join(GSM8K_DATA_DIR, "json", "test.json")
    with open(test_file, 'r') as f:
        data = json.load(f)
    samples = random.sample(data, min(NUM_SAMPLES, len(data)))
    print(f"\n  {len(samples)} questions loaded")

    # Load model
    print(f"  Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=_TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = get_num_heads(model)
    print(f"  Layers: {num_layers}, Heads: {num_heads}, Hidden: {hidden_size}")

    # Load pairwise data and compute per-head Syn_Red_Diff
    print(f"\n  Loading pairwise data: {PAIRWISE_PATH}")
    head_df = load_pairwise_and_rank(PAIRWISE_PATH)

    # Layer-matched head selection
    syn_per_layer, red_per_layer = select_layer_matched_heads(
        head_df, num_layers, _ABLATE_FRACTION)

    n_syn_layers = len(syn_per_layer)
    n_red_layers = len(red_per_layer)
    total_syn = sum(len(v) for v in syn_per_layer.values())
    total_red = sum(len(v) for v in red_per_layer.values())
    print(f"\n  Layer-matched selection ({_ABLATE_FRACTION * 100:.0f}% per layer):")
    print(f"  Syn: {total_syn} heads across {n_syn_layers} layers "
          f"(L{min(syn_per_layer)}-L{max(syn_per_layer)})")
    print(f"  Red: {total_red} heads across {n_red_layers} layers "
          f"(L{min(red_per_layer)}-L{max(red_per_layer)})")

    # ── Experiment 1: Baseline (no ablation) ──
    print(f"\n{'=' * 60}")
    print("Experiment 1: Baseline (no ablation)")
    print(f"{'=' * 60}")
    baseline_vectors = collect_hidden_states(
        model, tokenizer, samples, num_layers, desc="Baseline")
    baseline_ids = compute_ids_per_layer(baseline_vectors, num_layers)
    del baseline_vectors
    gc.collect()
    torch.cuda.empty_cache()

    # ── Experiment 2: Ablate synergistic heads ──
    print(f"\n{'=' * 60}")
    print(f"Experiment 2: Ablating synergistic heads ({total_syn} heads)")
    print(f"{'=' * 60}")
    abl_hooks = register_ablation_hooks(model, syn_per_layer, num_heads)
    syn_ablated_vectors = collect_hidden_states(
        model, tokenizer, samples, num_layers, desc="Syn ablation")
    syn_ablated_ids = compute_ids_per_layer(syn_ablated_vectors, num_layers)
    del syn_ablated_vectors
    remove_hooks(abl_hooks)
    gc.collect()
    torch.cuda.empty_cache()

    # ── Experiment 3: Ablate redundant heads ──
    print(f"\n{'=' * 60}")
    print(f"Experiment 3: Ablating redundant heads ({total_red} heads)")
    print(f"{'=' * 60}")
    abl_hooks = register_ablation_hooks(model, red_per_layer, num_heads)
    red_ablated_vectors = collect_hidden_states(
        model, tokenizer, samples, num_layers, desc="Red ablation")
    red_ablated_ids = compute_ids_per_layer(red_ablated_vectors, num_layers)
    del red_ablated_vectors
    remove_hooks(abl_hooks)
    gc.collect()
    torch.cuda.empty_cache()

    # ── Save results ──
    results = pd.DataFrame({
        'layer': range(num_layers),
        'baseline': baseline_ids,
        'ablate_syn': syn_ablated_ids,
        'ablate_red': red_ablated_ids,
        'ablate_fraction': _ABLATE_FRACTION,
    })
    output_path = os.path.join(OUTPUT_DIR, "intrinsic_dimension_ablation.csv")
    results.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved: {output_path}")
    print(f"{'=' * 60}")
    print(f"\n  Baseline  ID range: [{min(baseline_ids):.2f}, {max(baseline_ids):.2f}]")
    print(f"  Syn-abl   ID range: [{min(syn_ablated_ids):.2f}, {max(syn_ablated_ids):.2f}]")
    print(f"  Red-abl   ID range: [{min(red_ablated_ids):.2f}, {max(red_ablated_ids):.2f}]")

    # Key findings
    mean_bl = np.mean(baseline_ids)
    mean_syn = np.mean(syn_ablated_ids)
    mean_red = np.mean(red_ablated_ids)
    print(f"\n  Mean ID — Baseline: {mean_bl:.2f}, Syn-ablated: {mean_syn:.2f}, Red-ablated: {mean_red:.2f}")
    print(f"  Delta Syn: {mean_syn - mean_bl:+.2f},  Delta Red: {mean_red - mean_bl:+.2f}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
