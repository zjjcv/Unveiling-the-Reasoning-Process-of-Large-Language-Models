"""
CKA Similarity — Multi-Rule Synthetic Pattern (RBF Kernel).

Two groups follow the SAME abstract rules (ABA, ABB, AAB, ABAB, ABBA, ABBB)
but use completely different single-token vocabularies.  RBF-CKA is computed
only at the last (prediction) position per layer.

If middle layers extract abstract, vocabulary-invariant rules, CKA should
peak in the synergistic core region.

Usage:
    python src/cka_aba_rbf.py                   # Qwen3-8B-Base
    python src/cka_aba_rbf.py qwen3_4b_base
    python src/cka_aba_rbf.py qwen3_14b_base
    python src/cka_aba_rbf.py gemma3_4b_base
    python src/cka_aba_rbf.py gemma3_12b_it
    python src/cka_aba_rbf.py llama3_8b
    python src/cka_aba_rbf.py --linear           # Use linear kernel
"""

import os
import sys
import random
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============================================================
# Configuration (selected via CLI)
# ============================================================
if len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_4B_Base"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/intrinsic_dimension"
    _MODEL_LABEL = "Qwen3-4B-Base"
    _TORCH_DTYPE = torch.bfloat16
    CORE_LAYERS = (10, 26)
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_14b_base':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_14B_Base"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/intrinsic_dimension"
    _MODEL_LABEL = "Qwen3-14B-Base"
    _TORCH_DTYPE = torch.float16
    CORE_LAYERS = (5, 18)
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-4B-Base"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/intrinsic_dimension"
    _MODEL_LABEL = "Gemma3-4B-Base"
    _TORCH_DTYPE = torch.bfloat16
    CORE_LAYERS = (6, 21)
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b_it':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-12B-IT"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/intrinsic_dimension"
    _MODEL_LABEL = "Gemma3-12B-IT"
    _TORCH_DTYPE = torch.bfloat16
    CORE_LAYERS = (10, 29)
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Llama-3.1-8B"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/intrinsic_dimension"
    _MODEL_LABEL = "Llama-3.1-8B"
    _TORCH_DTYPE = torch.bfloat16
    CORE_LAYERS = (3, 20)
else:
    MODEL_PATH = "/data/zjj/Synergistic_Core/Qwen-3-8B-base"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/intrinsic_dimension"
    _MODEL_LABEL = "Qwen3-8B-Base"
    _TORCH_DTYPE = torch.bfloat16
    CORE_LAYERS = (8, 22)

RANDOM_SEED = 42
NUM_SEQS_PER_RULE = 50
NUM_COMPLETE = 2

_WORDS_A = [
    "cat", "dog", "run", "big", "red", "old", "new", "day", "sun", "car",
    "box", "cup", "hat", "map", "pen", "top", "win", "air", "arm", "bed",
    "bus", "ear", "egg", "fan", "gas", "ice", "jam", "key", "leg", "net",
]
_WORDS_B = [
    "oil", "pig", "rod", "sky", "tea", "van", "war", "zoo", "art", "bag",
    "bit", "bow", "boy", "bug", "can", "cow", "cry", "dip", "dry", "dug",
    "eat", "end", "eye", "fit", "fix", "fly", "fog", "fun", "gap", "gem",
]

PATTERNS = {
    'ABA':  ([0, 1, 0],    [0, 1]),
    'ABB':  ([0, 1, 1],    [0, 1]),
    'AAB':  ([0, 0, 1],    [0, 0]),
    'ABAB': ([0, 1, 0, 1], [0, 1, 0]),
    'ABBA': ([0, 1, 1, 0], [0, 1, 1]),
    'ABBB': ([0, 1, 1, 1], [0, 1, 1]),
}
PATTERN_NAMES = list(PATTERNS.keys())


# ============================================================
# Sequence Generation
# ============================================================
def apply_template(template, a, b):
    return [a if x == 0 else b for x in template]


def generate_sequence(rng, vocab, pattern_name, num_complete):
    full_tmpl, inc_tmpl = PATTERNS[pattern_name]
    parts = []
    for _ in range(num_complete):
        a, b = rng.sample(vocab, 2)
        parts.append(" ".join(apply_template(full_tmpl, a, b)))
    a, b = rng.sample(vocab, 2)
    parts.append(" ".join(apply_template(inc_tmpl, a, b)))
    return ", ".join(parts)


def generate_balanced_sequences(rng, vocab_a, vocab_b, num_per_rule, num_complete):
    seqs_a, seqs_b, rules = [], [], []
    for rule in PATTERN_NAMES:
        for _ in range(num_per_rule):
            seqs_a.append(generate_sequence(rng, vocab_a, rule, num_complete))
            seqs_b.append(generate_sequence(rng, vocab_b, rule, num_complete))
            rules.append(rule)
    return seqs_a, seqs_b, rules


# ============================================================
# CKA Kernels (GPU)
# ============================================================
def linear_cka(X_np, Y_np, device='cuda'):
    """Linear CKA (Kornblith et al., 2019)."""
    n = X_np.shape[0]
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_np, dtype=torch.float32, device=device)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    hs = lambda A, B: torch.sum(A * B) / (n - 1) ** 2
    return (hs(K, L) / (torch.sqrt(hs(K, K) * hs(L, L)) + 1e-10)).item()


def rbf_cka(X_np, Y_np, device='cuda'):
    """RBF (Gaussian) kernel CKA with median heuristic bandwidth."""
    n = X_np.shape[0]
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_np, dtype=torch.float32, device=device)
    dist_X = torch.cdist(X, X) ** 2
    dist_Y = torch.cdist(Y, Y) ** 2
    sigma_X = torch.median(dist_X[dist_X > 0]).clamp(min=1e-6)
    sigma_Y = torch.median(dist_Y[dist_Y > 0]).clamp(min=1e-6)
    K = torch.exp(-dist_X / (2 * sigma_X))
    L = torch.exp(-dist_Y / (2 * sigma_Y))
    K_c = K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()
    L_c = L - L.mean(1, keepdim=True) - L.mean(0, keepdim=True) + L.mean()
    hs = lambda A, B: torch.sum(A * B) / (n - 1) ** 2
    return (hs(K_c, L_c) / (torch.sqrt(hs(K_c, K_c) * hs(L_c, L_c)) + 1e-10)).item()


# ============================================================
# Kernel selection (after function definitions)
# ============================================================
_USE_LINEAR = '--linear' in sys.argv
if _USE_LINEAR:
    sys.argv.remove('--linear')
_CKA_FN = linear_cka if _USE_LINEAR else rbf_cka
_CKA_LABEL = "Linear" if _USE_LINEAR else "RBF"
_CKA_TAG = "linear" if _USE_LINEAR else "rbf"


# ============================================================
# Hidden-state collection
# ============================================================
def collect_last_hidden_states(model, tokenizer, sequences, num_layers, label=""):
    device = next(model.parameters()).device
    layer_vectors = {l: [] for l in range(num_layers)}
    desc = f"  [{label}]" if label else "  Collecting"
    for s_idx, seq in enumerate(tqdm(sequences, desc=desc)):
        inputs = tokenizer(seq, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for l in range(num_layers):
            vec = outputs.hidden_states[l + 1][0, -1, :].detach().cpu().float().numpy()
            layer_vectors[l].append(vec)
        if s_idx == 0:
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            print(f"    [S0] {len(tokens)} tokens: {tokens[:25]}...")
    return layer_vectors


# ============================================================
# Main
# ============================================================
def main():
    core_start, core_end = CORE_LAYERS
    total = NUM_SEQS_PER_RULE * len(PATTERN_NAMES)

    print("=" * 60)
    print(f"CKA Similarity — Multi-Rule Pattern ({_CKA_LABEL})")
    print(f"  Model: {_MODEL_LABEL}")
    print(f"  Sequences: {NUM_SEQS_PER_RULE}/rule × {len(PATTERN_NAMES)} = {total}")
    print(f"  Core layers: {core_start}-{core_end}")
    print(f"  CKA position: last token only")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    seqs_a, seqs_b, rules = generate_balanced_sequences(
        rng, _WORDS_A, _WORDS_B, NUM_SEQS_PER_RULE, NUM_COMPLETE)

    print(f"\n  Rule distribution: "
          f"{', '.join(f'{r}:{rules.count(r)}' for r in PATTERN_NAMES)}")
    print(f"\n  Group A example [{rules[0]}]: {seqs_a[0]}")
    print(f"  Group B example [{rules[0]}]: {seqs_b[0]}")

    print(f"\n  Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=_TORCH_DTYPE,
        device_map="auto", trust_remote_code=True)
    model.eval()

    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    num_layers = config.num_hidden_layers
    print(f"  Layers: {num_layers}")

    # Collect hidden states
    vec_A = collect_last_hidden_states(model, tokenizer, seqs_a,
                                        num_layers, label="Group A")
    vec_B = collect_last_hidden_states(model, tokenizer, seqs_b,
                                        num_layers, label="Group B")

    # Compute CKA per layer
    cka_results = []
    for l in range(num_layers):
        X = np.stack(vec_A[l], axis=0)
        Y = np.stack(vec_B[l], axis=0)
        n = min(X.shape[0], Y.shape[0])
        X, Y = X[:n], Y[:n]
        cka_val = _CKA_FN(X, Y, device='cuda')
        cka_results.append(cka_val)
        if l % 6 == 0 or l == num_layers - 1:
            print(f"    Layer {l:2d}: CKA = {cka_val:.4f}  ({n} vectors)")

    del vec_A, vec_B; gc.collect(); torch.cuda.empty_cache()

    # Save
    import pandas as pd
    results = pd.DataFrame({
        'layer': range(num_layers),
        'cka': cka_results,
        'kernel': _CKA_TAG,
        'num_sequences': total,
        'core_start': core_start,
        'core_end': core_end,
    })
    output_path = os.path.join(OUTPUT_DIR, f"cka_baseline_{_CKA_TAG}.csv")
    results.to_csv(output_path, index=False)

    # Summary
    core_cka = np.mean(cka_results[core_start:core_end + 1])
    all_cka = np.mean(cka_results)
    peak_val = max(cka_results)
    peak_layer = cka_results.index(peak_val)
    print(f"\n{'=' * 60}")
    print(f"Results saved: {output_path}")
    print(f"{'=' * 60}")
    print(f"  CKA ({_CKA_LABEL}) — all mean: {all_cka:.4f}, "
          f"core mean: {core_cka:.4f}, peak: {peak_val:.4f} @ L{peak_layer}")
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
