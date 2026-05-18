"""
Cross-vocabulary Abstraction Probes — Figure 5c & 5d.

Uses synthetic multi-rule sequences with two non-overlapping single-token
vocabularies. All rules are length-4, eliminating length/position shortcuts.

Rules: AABA, AABB, ABAA, ABAB, ABBA, ABBB
Format: 2 complete 4-token examples + incomplete 3-token query prefix
  e.g. "cat cat dog cat, dog cat dog dog, sun sun day"

Uses per-layer delta (h_{l+1} - h_l) to isolate each layer's contribution,
removing residual-stream embedding leakage.

Figure 5c — Cross-vocabulary rule probe:
  6-class linear probe per layer on delta_l, train A→test B + reverse.
  Peaks in synergistic core where delta encodes abstract rules.

Figure 5d — Visible query-token identity probe:
  Predict the specific word at the last position of the query prefix,
  using delta_l features. Dips in core where delta is abstract.

Usage:
    python src/probe_abstraction.py                   # Qwen3-8B-Base
    python src/probe_abstraction.py qwen3_4b_base
    python src/probe_abstraction.py qwen3_14b_base
    python src/probe_abstraction.py gemma3_4b_base
    python src/probe_abstraction.py gemma3_12b_it
    python src/probe_abstraction.py llama3_8b
"""

import os
import sys
import random
import gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

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
NUM_SEQS_PER_RULE = 20
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

# All length-4 rules — eliminates length/position shortcuts
# Incomplete prefixes share same 3-token form for rule pairs, forcing context use:
#   [a,a,b] → AABA or AABB,  [a,b,a] → ABAA or ABAB,  [a,b,b] → ABBA or ABBB
PATTERNS = {
    'AABA': ([0, 0, 1, 0], [0, 0, 1]),
    'AABB': ([0, 0, 1, 1], [0, 0, 1]),
    'ABAA': ([0, 1, 0, 0], [0, 1, 0]),
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
    """Generate sequence, return (sequence_string, last_visible_token_word)."""
    full_tmpl, inc_tmpl = PATTERNS[pattern_name]
    parts = []
    for _ in range(num_complete):
        a, b = rng.sample(vocab, 2)
        parts.append(" ".join(apply_template(full_tmpl, a, b)))
    a, b = rng.sample(vocab, 2)
    inc_tokens = apply_template(inc_tmpl, a, b)
    parts.append(" ".join(inc_tokens))
    last_token_word = inc_tokens[-1]  # last visible token in query prefix
    return ", ".join(parts), last_token_word


def generate_balanced_sequences(rng, vocab_a, vocab_b, num_per_rule, num_complete):
    seqs_a, seqs_b, rules = [], [], []
    last_tokens_a, last_tokens_b = [], []
    for rule in PATTERN_NAMES:
        for _ in range(num_per_rule):
            seq_a, lt_a = generate_sequence(rng, vocab_a, rule, num_complete)
            seq_b, lt_b = generate_sequence(rng, vocab_b, rule, num_complete)
            seqs_a.append(seq_a)
            seqs_b.append(seq_b)
            rules.append(rule)
            last_tokens_a.append(lt_a)
            last_tokens_b.append(lt_b)
    return seqs_a, seqs_b, rules, last_tokens_a, last_tokens_b


# ============================================================
# Hidden State Collection
# ============================================================
def collect_last_hidden_states(model, tokenizer, sequences, num_layers, label=""):
    """Collect per-layer delta: delta_l = h_{l+1} - h_l (isolates each layer's contribution)."""
    device = next(model.parameters()).device
    layer_vectors = {l: [] for l in range(num_layers)}
    desc = f"  [{label}]" if label else "  Collecting"
    for s_idx, seq in enumerate(tqdm(sequences, desc=desc)):
        inputs = tokenizer(seq, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Per-layer delta: delta_l = hidden_states[l+1] - hidden_states[l]
        for l in range(num_layers):
            h_l = outputs.hidden_states[l][0, -1, :].detach().cpu().float().numpy()
            h_next = outputs.hidden_states[l + 1][0, -1, :].detach().cpu().float().numpy()
            delta = h_next - h_l
            layer_vectors[l].append(delta)
        if s_idx == 0:
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            print(f"    [S0] {len(tokens)} tokens: {tokens[:25]}...")
    return layer_vectors


# ============================================================
# Figure 5c: Cross-vocabulary Rule Probe (6-class)
# ============================================================
def run_rule_probe(vec_A, vec_B, rules, num_layers):
    le = LabelEncoder()
    y = le.fit_transform(rules)

    rule_acc = []
    for l in range(num_layers):
        X_A = np.stack(vec_A[l], axis=0)
        X_B = np.stack(vec_B[l], axis=0)

        clf = LogisticRegression(
            max_iter=500, C=1.0, random_state=RANDOM_SEED,
            solver='lbfgs', n_jobs=-1, tol=1e-3)

        # Train A → Test B
        clf.fit(X_A, y)
        acc_AB = clf.score(X_B, y)

        # Train B → Test A
        clf.fit(X_B, y)
        acc_BA = clf.score(X_A, y)

        avg_acc = (acc_AB + acc_BA) / 2
        rule_acc.append(avg_acc)

        if l % 6 == 0 or l == num_layers - 1:
            print(f"    Layer {l:2d}: Rule acc = {avg_acc:.4f} "
                  f"(A→B={acc_AB:.4f}, B→A={acc_BA:.4f})")

    return rule_acc


# ============================================================
# Figure 5d: Visible Query-Token Identity Probe (30-class per group)
# ============================================================
def run_vocab_probe(vec_A, vec_B, last_tokens_a, last_tokens_b, num_layers):
    """Per-group 30-class probe: predict last visible token word identity.
    PCA(200) + 3-fold stratified CV, average accuracy across both groups."""
    _PCA_DIM = 200
    _N_SPLITS = 3
    results = []

    for group_name, vec, last_tokens in [("A", vec_A, last_tokens_a),
                                           ("B", vec_B, last_tokens_b)]:
        le = LabelEncoder()
        y_word = le.fit_transform(last_tokens)
        n_classes = len(le.classes_)

        # Filter classes with < _N_SPLITS samples
        class_counts = pd.Series(y_word).value_counts()
        valid_classes = class_counts[class_counts >= _N_SPLITS].index
        mask = np.isin(y_word, valid_classes)
        if mask.sum() < n_classes * _N_SPLITS:
            print(f"    [{group_name}] Filtering to {len(valid_classes)}/{n_classes} "
                  f"classes (need ≥{_N_SPLITS} samples), {mask.sum()}/{len(y_word)} seqs")
        y_filtered = y_word[mask]
        # Re-encode
        le2 = LabelEncoder()
        y_filtered = le2.fit_transform(y_filtered)
        n_classes_eff = len(le2.classes_)

        skf = StratifiedKFold(n_splits=_N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

        for l in range(num_layers):
            X_full = np.stack(vec[l], axis=0)
            X = X_full[mask]

            # PCA per layer, fit on full data of this group
            n_comp = min(_PCA_DIM, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
            X_reduced = pca.fit_transform(X)

            fold_accs = []
            for train_idx, test_idx in skf.split(X_reduced, y_filtered):
                clf = LogisticRegression(
                    max_iter=500, C=1.0, random_state=RANDOM_SEED,
                    solver='lbfgs', n_jobs=-1, tol=1e-3)
                clf.fit(X_reduced[train_idx], y_filtered[train_idx])
                fold_accs.append(clf.score(X_reduced[test_idx], y_filtered[test_idx]))

            results.append({
                'group': group_name,
                'layer': l,
                'accuracy': np.mean(fold_accs),
                'std': np.std(fold_accs),
                'n_classes': n_classes_eff,
            })

    # Average across groups per layer
    vocab_acc = []
    for l in range(num_layers):
        acc_a = [r['accuracy'] for r in results if r['group'] == 'A' and r['layer'] == l][0]
        acc_b = [r['accuracy'] for r in results if r['group'] == 'B' and r['layer'] == l][0]
        avg = (acc_a + acc_b) / 2
        vocab_acc.append(avg)
        if l % 6 == 0 or l == num_layers - 1:
            print(f"    Layer {l:2d}: Token acc = {avg:.4f} "
                  f"(A={acc_a:.4f}, B={acc_b:.4f})")

    return vocab_acc


# ============================================================
# Main
# ============================================================
def main():
    core_start, core_end = CORE_LAYERS
    total = NUM_SEQS_PER_RULE * len(PATTERN_NAMES)

    print("=" * 60)
    print(f"Cross-vocabulary Abstraction Probes — {_MODEL_LABEL}")
    print(f"  Rules: {', '.join(PATTERN_NAMES)} (all length-4)")
    print(f"  Sequences: {NUM_SEQS_PER_RULE}/rule x {len(PATTERN_NAMES)} = {total}/group")
    print(f"  Core layers: {core_start}-{core_end}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate sequences
    rng = random.Random(RANDOM_SEED)
    seqs_a, seqs_b, rules, lt_a, lt_b = generate_balanced_sequences(
        rng, _WORDS_A, _WORDS_B, NUM_SEQS_PER_RULE, NUM_COMPLETE)

    print(f"\n  Rule distribution: "
          f"{', '.join(f'{r}:{rules.count(r)}' for r in PATTERN_NAMES)}")
    print(f"\n  Group A example [{rules[0]}]: {seqs_a[0]}")
    print(f"    Last visible token: '{lt_a[0]}'")
    print(f"  Group B example [{rules[0]}]: {seqs_b[0]}")
    print(f"    Last visible token: '{lt_b[0]}'")

    # Load model
    print(f"\n  Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=_TORCH_DTYPE,
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

    del model; gc.collect(); torch.cuda.empty_cache()

    # Figure 5c: Cross-vocabulary rule probe
    print(f"\n{'=' * 60}")
    print("Figure 5c: Cross-vocabulary Rule Probe (6-class)")
    print(f"{'=' * 60}")
    rule_acc = run_rule_probe(vec_A, vec_B, rules, num_layers)

    # Figure 5d: Visible query-token identity probe
    print(f"\n{'=' * 60}")
    print("Figure 5d: Visible Query-Token Identity Probe (30-class)")
    print(f"{'=' * 60}")
    vocab_acc = run_vocab_probe(vec_A, vec_B, lt_a, lt_b, num_layers)

    # Save
    results = pd.DataFrame({
        'layer': range(num_layers),
        'rule_accuracy': rule_acc,
        'vocab_accuracy': vocab_acc,
        'num_sequences': total,
        'core_start': core_start,
        'core_end': core_end,
    })
    output_path = os.path.join(OUTPUT_DIR, "probe_abstraction.csv")
    results.to_csv(output_path, index=False)

    # Summary
    core_rule = np.mean(rule_acc[core_start:core_end + 1])
    all_rule = np.mean(rule_acc)
    peak_rule = max(rule_acc)
    peak_l = rule_acc.index(peak_rule)
    core_vocab = np.mean(vocab_acc[core_start:core_end + 1])
    all_vocab = np.mean(vocab_acc)
    min_vocab = min(vocab_acc)
    min_l = vocab_acc.index(min_vocab)

    print(f"\n{'=' * 60}")
    print(f"Results saved: {output_path}")
    print(f"{'=' * 60}")
    print(f"\n  Rule Probe  — all mean: {all_rule:.4f}, core mean: {core_rule:.4f}, "
          f"peak: {peak_rule:.4f} @ L{peak_l}")
    print(f"  Token Probe — all mean: {all_vocab:.4f}, core mean: {core_vocab:.4f}, "
          f"min: {min_vocab:.4f} @ L{min_l}")
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
