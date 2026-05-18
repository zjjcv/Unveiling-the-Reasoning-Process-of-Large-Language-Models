"""
Collect layer-wise relative change data for 4 experiments (Multi-GPU).

Experiments:
  1. Full ablation:        Skip layer s for all generation steps → measure impact on layers l > s
  2. Future prediction:    Skip layer s only during prefill → measure impact during decode steps
  3. Circuit localization (all):  Remove contribution_s from layer l input (forward pass on generated seq)
  4. Circuit localization (future): Same as #3 but only at positions > ts

Metric (exps 1 & 2):  ||normal_delta - ablated_delta|| / ||normal_delta||  (last-step mean vectors)
Metric (exps 3 & 4):  max over positions of ||normal_contrib - modified_contrib|| / ||normal_contrib||

Output: 4 CSV files with columns: question_id, ablated_layer_s, affected_layer_l, metric_value

Usage:
    python src/layer_relative_change.py                       # Gemma-3-4B-Instruct
    python src/layer_relative_change.py gemma3_4b_base        # Gemma3-4B-Base
    python src/layer_relative_change.py gemma3_12b             # Gemma3-12B-IT
    python src/layer_relative_change.py qwen3_8b_base          # Qwen3-8B-Base
    python src/layer_relative_change.py qwen3_4b_base          # Qwen3-4B-Base
    python src/layer_relative_change.py qwen3_14b_base         # Qwen3-14B-Base
    python src/layer_relative_change.py llama3_8b              # Llama-3.1-8B
"""

import json
import os
import random
from typing import Dict, List, Tuple
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============== Configuration ==============
import sys as _sys
_MULTI_GPU = False  # default: 1 GPU per model
if len(_sys.argv) > 1 and _sys.argv[1] == 'gemma3_4b_base':
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-4B-Base"
    _DEFAULT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/layer_relative_change"
    _MODEL_LABEL = "Gemma3-4B-Base"
elif len(_sys.argv) > 1 and _sys.argv[1] == 'gemma3_12b':
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-12B-IT"
    _DEFAULT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/layer_relative_change"
    _MODEL_LABEL = "Gemma3-12B-Instruct"
    _MULTI_GPU = True  # 2 GPUs per model instance
elif len(_sys.argv) > 1 and _sys.argv[1] == 'qwen3_8b_base':
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Qwen-3-8B-base"
    _DEFAULT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/layer_relative_change"
    _MODEL_LABEL = "Qwen3-8B-Base"
elif len(_sys.argv) > 1 and _sys.argv[1] == 'qwen3_4b_base':
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_4B_Base"
    _DEFAULT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/layer_relative_change"
    _MODEL_LABEL = "Qwen3-4B-Base"
elif len(_sys.argv) > 1 and _sys.argv[1] == 'qwen3_14b_base':
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_14B_Base"
    _DEFAULT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/layer_relative_change"
    _MODEL_LABEL = "Qwen3-14B-Base"
    _MULTI_GPU = True  # 2 GPUs per model instance
elif len(_sys.argv) > 1 and _sys.argv[1] == 'llama3_8b':
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Llama-3.1-8B"
    _DEFAULT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/layer_relative_change"
    _MODEL_LABEL = "Llama-3.1-8B"
else:
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Gemma-3-4B-Instruct"
    _DEFAULT_OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/layer_relative_change"
    _MODEL_LABEL = "Gemma-3-4B-Instruct"

MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
GSM8K_DATA_DIR = "/data/zjj/Synergistic_Core/data/gsm8k"
OUTPUT_DIR = _DEFAULT_OUTPUT_DIR

NUM_QUESTIONS = 7
NUM_GPUS = 7
RANDOM_SEED = 42
MAX_NEW_TOKENS = 2048


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_gsm8k_questions(data_dir: str, num_questions: int, seed: int) -> List[Dict]:
    """Load and sample GSM8K questions."""
    random.seed(seed)
    test_file = os.path.join(data_dir, "json", "test.json")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"GSM8K data not found at {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    num_to_sample = min(num_questions, len(data))
    sampled = random.sample(data, num_to_sample)
    return sampled


def create_gsm8k_prompt(question: str) -> str:
    """Create zero-shot prompt for GSM8K question."""
    if "Question:" in question:
        return question
    return f"Question: {question}\nAnswer:"


def get_model_layers(model):
    """Get transformer layers from various model architectures."""
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            return lm.model.layers
        elif hasattr(lm, 'layers'):
            return lm.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in model: {type(model)}")


def process_single_question(gpu_id, q_idx: int, question_text: str,
                            model_path: str, max_new_tokens: int,
                            output_dir: str) -> Tuple[List, List, List, List]:
    """Process a single question on a specific GPU — run all 4 experiments.

    gpu_id: int for single-GPU, or str like "0,1" for multi-GPU.
    """
    gpu_str = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    if ',' not in gpu_str:
        device = f"cuda:0"
        torch.cuda.set_device(0)
    else:
        device = None  # device_map="auto" handles placement

    print(f"[GPU {gpu_str}] Processing question {q_idx}...")

    # Load model on this GPU
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto" if device is None else {"": device},
        trust_remote_code=True
    )
    model.eval()

    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        num_layers = model.config.num_hidden_layers

    layers = get_model_layers(model)
    prompt = create_gsm8k_prompt(question_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    results_exp1 = []
    results_exp2 = []
    results_exp3 = []
    results_exp4 = []

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Normal generation — capture last-step layer outputs
    # Hook captures out.mean(axis=(0,1)) = [hidden_size] vector per layer
    # Overwrites each step → only last step is kept
    # ══════════════════════════════════════════════════════════════════════
    print(f"[GPU {gpu_str}] Q{q_idx}: Step 1 — Normal generation...")

    layer_outputs = {}

    def create_normal_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            out_np = out.detach().cpu().float().numpy()
            out_mean = out_np.mean(axis=(0, 1))
            layer_outputs[layer_idx] = out_mean
        return hook

    hooks = [layers[idx].register_forward_hook(create_normal_hook(idx))
             for idx in range(num_layers)]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    for hook in hooks:
        hook.remove()

    # Compute normal deltas: Δ_l = output_{l+1} - output_l  (last step)
    normal_deltas = {}
    for l in range(num_layers - 1):
        if l in layer_outputs and (l + 1) in layer_outputs:
            normal_deltas[l] = layer_outputs[l + 1] - layer_outputs[l]

    prompt_len = inputs["input_ids"].shape[1]
    seq_len = generated_ids.shape[1]
    ts = seq_len // 2

    if q_idx == 0:
        generated_text = tokenizer.decode(
            generated_ids[0, prompt_len:], skip_special_tokens=True)
        print(f"[GPU {gpu_str}] Q{q_idx}: prompt_len={prompt_len}, "
              f"seq_len={seq_len}, ts={ts}")
        print(f"[GPU {gpu_str}] Q{q_idx}: Generated: "
              f"{generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Experiment 1 — Full Ablation
    # Skip layer s for ALL generation steps (prefill + decode)
    # ══════════════════════════════════════════════════════════════════════
    print(f"[GPU {gpu_str}] Q{q_idx}: Step 2 — Exp 1: Full ablation...")

    for s in range(num_layers):
        if s % 10 == 0:
            print(f"[GPU {gpu_str}] Q{q_idx}: Exp 1 — layer {s}/{num_layers}")

        ablated_layer_outputs = {}

        def create_ablation_hook(layer_idx, ablated_s):
            def hook(module, input, output):
                if layer_idx == ablated_s:
                    # Skip this layer: return input hidden states, preserving output format
                    hidden = input[0] if isinstance(input, tuple) else input
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                else:
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    out_np = out.detach().cpu().float().numpy()
                    ablated_layer_outputs[layer_idx] = out_np.mean(axis=(0, 1))
                    return output
            return hook

        hooks = [layers[idx].register_forward_hook(create_ablation_hook(idx, s))
                 for idx in range(num_layers)]

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for hook in hooks:
            hook.remove()

        # Compute relative changes for affected layers (l > s)
        for l in range(s + 1, num_layers):
            if l in ablated_layer_outputs and (l + 1) in ablated_layer_outputs:
                ablated_delta = ablated_layer_outputs[l + 1] - ablated_layer_outputs[l]
            elif l in ablated_layer_outputs:
                ablated_delta = ablated_layer_outputs[l]
            else:
                continue

            normal_delta = normal_deltas.get(l)
            if normal_delta is None:
                continue

            diff = normal_delta - ablated_delta
            normal_norm = np.linalg.norm(normal_delta)
            metric = np.linalg.norm(diff) / normal_norm if normal_norm > 1e-10 else 0.0

            results_exp1.append({
                'question_id': q_idx,
                'ablated_layer_s': s,
                'affected_layer_l': l,
                'metric_value': metric
            })

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Experiment 2 — Future Prediction
    # Skip layer s only during prefill (1st hook call = prompt processing)
    # During decode steps, layer s operates normally → measure impact
    # ══════════════════════════════════════════════════════════════════════
    print(f"[GPU {gpu_str}] Q{q_idx}: Step 3 — Exp 2: Future prediction...")

    for s in range(num_layers):
        if s % 10 == 0:
            print(f"[GPU {gpu_str}] Q{q_idx}: Exp 2 — layer {s}/{num_layers}")

        ablated_layer_outputs = {}
        call_count = [0]

        def create_future_hook(layer_idx, ablated_s):
            def hook(module, input, output):
                if layer_idx == ablated_s:
                    call_count[0] += 1
                    if call_count[0] == 1:
                        # First call (prefill): skip layer s, preserving output format
                        hidden = input[0] if isinstance(input, tuple) else input
                        if isinstance(output, tuple):
                            return (hidden,) + output[1:]
                        return hidden
                    else:
                        # Decode steps: normal operation, capture output
                        if isinstance(output, tuple):
                            out = output[0]
                        else:
                            out = output
                        out_np = out.detach().cpu().float().numpy()
                        ablated_layer_outputs[layer_idx] = out_np.mean(axis=(0, 1))
                        return output
                else:
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    out_np = out.detach().cpu().float().numpy()
                    ablated_layer_outputs[layer_idx] = out_np.mean(axis=(0, 1))
                    return output
            return hook

        hooks = [layers[idx].register_forward_hook(create_future_hook(idx, s))
                 for idx in range(num_layers)]
        call_count[0] = 0

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for hook in hooks:
            hook.remove()

        for l in range(s + 1, num_layers):
            if l in ablated_layer_outputs and (l + 1) in ablated_layer_outputs:
                ablated_delta = ablated_layer_outputs[l + 1] - ablated_layer_outputs[l]
            elif l in ablated_layer_outputs:
                ablated_delta = ablated_layer_outputs[l]
            else:
                continue

            normal_delta = normal_deltas.get(l)
            if normal_delta is None:
                continue

            diff = normal_delta - ablated_delta
            normal_norm = np.linalg.norm(normal_delta)
            metric = np.linalg.norm(diff) / normal_norm if normal_norm > 1e-10 else 0.0

            results_exp2.append({
                'question_id': q_idx,
                'ablated_layer_s': s,
                'affected_layer_l': l,
                'metric_value': metric
            })

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Experiments 3 & 4 — Circuit Localization
    # Forward pass on generated sequence → get full hidden states
    # Then run individual layers with modified inputs (no propagation)
    # ══════════════════════════════════════════════════════════════════════
    print(f"[GPU {gpu_str}] Q{q_idx}: Step 4 — Exps 3 & 4: Circuit localization...")

    captured_kwargs = {}

    def make_capture_hook(layer_idx):
        def pre_hook(module, args, kwargs):
            captured_kwargs[layer_idx] = kwargs.copy()
        return pre_hook

    capture_hooks = [layers[idx].register_forward_pre_hook(
        make_capture_hook(idx), with_kwargs=True) for idx in range(num_layers)]

    with torch.no_grad():
        outputs = model(generated_ids, output_hidden_states=True)

    for h in capture_hooks:
        h.remove()

    normal_hs = [hs[0].cpu().float().numpy() for hs in outputs.hidden_states]
    normal_hs_tensors = [hs[0] for hs in outputs.hidden_states]

    # Clean kwargs for individual layer calls
    clean_kwargs = {}
    for layer_idx, kw in captured_kwargs.items():
        clean_kwargs[layer_idx] = {
            k: v for k, v in kw.items()
            if k not in ['past_key_value', 'use_cache', 'output_attentions',
                         'cache_position', 'past_key_values']
        }

    future_slice = slice(ts, seq_len)

    for s in range(num_layers):
        if s % 10 == 0:
            print(f"[GPU {gpu_str}] Q{q_idx}: Circuit — source layer {s}/{num_layers}")

        contribution_s = normal_hs_tensors[s + 1] - normal_hs_tensors[s]

        for l in range(s + 1, num_layers):
            modified_input = normal_hs_tensors[l] - contribution_s
            modified_input_batch = modified_input.unsqueeze(0)

            with torch.no_grad():
                try:
                    output = layers[l](modified_input_batch, **clean_kwargs[l])
                except TypeError:
                    minimal_kw = {}
                    for k in ['attention_mask', 'position_ids', 'position_embeddings']:
                        if k in clean_kwargs[l]:
                            minimal_kw[k] = clean_kwargs[l][k]
                    output = layers[l](modified_input_batch, **minimal_kw)

            if isinstance(output, tuple):
                output = output[0]

            output_np = output[0].cpu().float().numpy()
            modified_np = modified_input.cpu().float().numpy()

            normal_contrib = normal_hs[l + 1] - normal_hs[l]
            modified_contrib = output_np - modified_np
            diff = normal_contrib - modified_contrib

            n_norms = np.linalg.norm(normal_contrib, axis=1)
            d_norms = np.linalg.norm(diff, axis=1)

            # Experiment 3: all positions, max over positions
            valid = n_norms > 1e-10
            metric_all = float((d_norms[valid] / n_norms[valid]).mean()) if valid.any() else 0.0
            results_exp3.append({
                'question_id': q_idx,
                'ablated_layer_s': s,
                'affected_layer_l': l,
                'metric_value': metric_all
            })

            # Experiment 4: future positions only, max over positions
            n_f = np.linalg.norm(normal_contrib[future_slice], axis=1)
            d_f = np.linalg.norm(diff[future_slice], axis=1)
            valid_f = n_f > 1e-10
            metric_future = float((d_f[valid_f] / n_f[valid_f]).mean()) if valid_f.any() else 0.0
            results_exp4.append({
                'question_id': q_idx,
                'ablated_layer_s': s,
                'affected_layer_l': l,
                'metric_value': metric_future
            })

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print(f"[GPU {gpu_str}] Q{q_idx}: Complete — "
          f"Exp1={len(results_exp1)}, Exp2={len(results_exp2)}, "
          f"Exp3={len(results_exp3)}, Exp4={len(results_exp4)} records")

    return results_exp1, results_exp2, results_exp3, results_exp4


def save_results(all_results: List[Dict], output_dir: str, filename: str):
    """Save collected results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} records → {output_path}")
    return df


def _worker_wrapper(gpu_id, q_idx, q_text, model_path, max_new_tokens, output_dir, result_queue):
    """Wrapper that catches exceptions and puts results in queue."""
    try:
        r1, r2, r3, r4 = process_single_question(
            gpu_id, q_idx, q_text, model_path, max_new_tokens, output_dir)
        result_queue.put((r1, r2, r3, r4))
    except Exception as e:
        print(f"[GPU {gpu_id}] Q{q_idx} FAILED: {e}")
        result_queue.put(([], [], [], []))


def main():
    """Main execution function - multi-GPU parallel processing."""
    print("=" * 60)
    print(f"Layer Relative Change Collection ({_MODEL_LABEL})")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  GPUs: {NUM_GPUS}")
    print(f"  Questions: {NUM_QUESTIONS}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Output: {OUTPUT_DIR}")

    set_seed(RANDOM_SEED)

    print(f"\nLoading GSM8K questions...")
    questions = load_gsm8k_questions(GSM8K_DATA_DIR, NUM_QUESTIONS, RANDOM_SEED)
    print(f"  Sampled {len(questions)} questions")

    question_data = []
    for i, q in enumerate(questions):
        if isinstance(q, dict):
            q_text = q.get('question', '')
        else:
            q_text = str(q)
        question_data.append((i, q_text))

    print(f"\nStarting parallel processing...")
    print("=" * 60)

    if _MULTI_GPU:
        # ── Multi-GPU mode: 2 GPUs per question ──
        GPU_PAIRS = ["0,1", "2,3", "4,5", "6,7"]

        # Single batch: 4 questions across 8 GPUs (2 GPUs each)
        batch_data = question_data[:4]
        print(f"\nMulti-GPU batch: {len(batch_data)} questions on GPU pairs {GPU_PAIRS}")
        pool = mp.Pool(processes=len(batch_data))
        results = pool.starmap(
            process_single_question,
            [
                (gpu, q_idx, q_text, MODEL_PATH, MAX_NEW_TOKENS, OUTPUT_DIR)
                for gpu, (q_idx, q_text) in zip(GPU_PAIRS, batch_data)
            ]
        )
        pool.close()
        pool.join()
    else:
        # ── Single-GPU mode: 1 GPU per question, using Process + Queue ──
        _AVAIL_GPUS = [0, 1, 2, 4, 5, 6, 7]  # skip GPU 3 (hardware issue)
        result_queue = mp.Queue()
        processes = []
        for gpu_id, (q_idx, q_text) in zip(_AVAIL_GPUS, question_data):
            p = mp.Process(
                target=_worker_wrapper,
                args=(gpu_id, q_idx, q_text, MODEL_PATH, MAX_NEW_TOKENS, OUTPUT_DIR, result_queue)
            )
            p.start()
            processes.append(p)

        results = []
        for _ in range(len(processes)):
            results.append(result_queue.get())

        for p in processes:
            p.join()

    print("\n" + "=" * 60)
    print("Merging results from all GPUs...")

    all_exp1, all_exp2, all_exp3, all_exp4 = [], [], [], []
    for exp1, exp2, exp3, exp4 in results:
        all_exp1.extend(exp1)
        all_exp2.extend(exp2)
        all_exp3.extend(exp3)
        all_exp4.extend(exp4)

    print(f"\nSaving results to: {OUTPUT_DIR}")
    df1 = save_results(all_exp1, OUTPUT_DIR, "layer_relative_change.csv")
    df2 = save_results(all_exp2, OUTPUT_DIR, "future_prediction.csv")
    df3 = save_results(all_exp3, OUTPUT_DIR, "circuit_localization_all.csv")
    df4 = save_results(all_exp4, OUTPUT_DIR, "circuit_localization_future.csv")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for name, df in [("Exp 1: Full Ablation", df1),
                     ("Exp 2: Future Prediction", df2),
                     ("Exp 3: Circuit Loc (All)", df3),
                     ("Exp 4: Circuit Loc (Future)", df4)]:
        print(f"\n  {name}:")
        print(f"    Records: {len(df):,}")
        print(f"    Mean metric: {df['metric_value'].mean():.6f}")
        print(f"    Range: [{df['metric_value'].min():.6f}, {df['metric_value'].max():.6f}]")

    print(f"\n{'='*60}")
    print("All Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
