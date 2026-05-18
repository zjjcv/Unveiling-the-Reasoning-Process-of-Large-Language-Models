"""
Parallel Ablation Study — 4 experiments on 4 GPUs.

Experiments (one per GPU):
  1. high_to_low  — ablate synergistic heads first
  2. low_to_high  — ablate redundant heads first
  3. random_run1
  4. random_run2

Supports: gemma3_4b_base, qwen3_8b_base, qwen3_4b_base, qwen3_14b_base (default)

Usage:
    python src/ablation_study.py gemma3_4b_base     # Gemma3-4B-Base
    python src/ablation_study.py qwen3_8b_base      # Qwen3-8B-Base
    python src/ablation_study.py qwen3_4b_base      # Qwen3-4B-Base
    python src/ablation_study.py llama3_8b           # Llama-3.1-8B
    python src/ablation_study.py                     # Qwen3-14B-Base
"""

import os
import sys
import gc
import io
import json
import re
import random
from multiprocessing import Process, Queue
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm


class BatchAnswerStopCriteria(StoppingCriteria):
    """Stop batch generation when ALL sequences have produced '####' followed by a number."""

    def __init__(self, tokenizer, prompt_lengths: List[int]):
        self.tokenizer = tokenizer
        self.prompt_lengths = prompt_lengths
        self._done = [False] * len(prompt_lengths)

    def __call__(self, input_ids, scores, **kwargs):
        all_done = True
        for i in range(input_ids.shape[0]):
            if self._done[i]:
                continue
            generated = input_ids[i, self.prompt_lengths[i]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            idx = text.find("####")
            if idx == -1:
                all_done = False
                continue
            after = text[idx + 4:].lstrip()
            if after and after[0].isdigit() and not text[-1].isdigit():
                self._done[i] = True
            else:
                all_done = False
        return all_done

# ============== Configuration ==============
if len(sys.argv) > 1 and sys.argv[1] == 'gemma3_4b_base':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-4B-Base"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/ablation"
    _MODEL_LABEL = "Gemma3-4B-Base"
    _IS_BASE = True
    _NUM_HEADS = 8
    _DTYPE = torch.bfloat16
    _GPUS = [0, 1, 2, 4]
    _EXPERIMENTS = ['high_to_low', 'low_to_high', 'random_run1', 'random_run2']
    _EXPERIMENTS_BATCH2 = []
    _GPUS_BATCH2 = []
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_8b_base':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Qwen-3-8B-base"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/ablation"
    _MODEL_LABEL = "Qwen3-8B-Base"
    _IS_BASE = True
    _NUM_HEADS = 32
    _DTYPE = torch.float16
    _GPUS = [0, 1, 2, 3, 4]
    _EXPERIMENTS = ['high_to_low', 'low_to_high', 'random_run1', 'random_run2', 'random_run3']
    _EXPERIMENTS_BATCH2 = []
    _GPUS_BATCH2 = []
elif len(sys.argv) > 1 and sys.argv[1] == 'qwen3_4b_base':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_4B_Base"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/ablation"
    _MODEL_LABEL = "Qwen3-4B-Base"
    _IS_BASE = True
    _NUM_HEADS = 32
    _DTYPE = torch.bfloat16
    _GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
    _EXPERIMENTS = ['high_to_low', 'low_to_high', 'random_run1', 'random_run2', 'random_run3', 'random_run4', 'random_run5', 'random_run6']
    _EXPERIMENTS_BATCH2 = []
    _GPUS_BATCH2 = []
elif len(sys.argv) > 1 and sys.argv[1] == 'gemma3_12b':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-12B-IT"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-12B-Instruct/data/ablation"
    _MODEL_LABEL = "Gemma3-12B-Instruct"
    _IS_BASE = True
    _NUM_HEADS = 16
    _DTYPE = torch.bfloat16
    _GPUS = ["0,1", "2,3", "4,5", "6,7"]
    _EXPERIMENTS = ['high_to_low', 'low_to_high', 'random_run1', 'random_run2']
    _EXPERIMENTS_BATCH2 = []
    _GPUS_BATCH2 = []
    _BATCH_SIZE_OVERRIDE = 12
elif len(sys.argv) > 1 and sys.argv[1] == 'llama3_8b':
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Llama-3.1-8B"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/ablation"
    _MODEL_LABEL = "Llama-3.1-8B"
    _IS_BASE = True
    _NUM_HEADS = 32
    _DTYPE = torch.bfloat16
    _GPUS = [0, 1, 2, 4, 5]
    _EXPERIMENTS = ['high_to_low', 'low_to_high', 'random_run1', 'random_run2', 'random_run3']
    _EXPERIMENTS_BATCH2 = []
    _GPUS_BATCH2 = []
    _BATCH_SIZE_OVERRIDE = 12  # Llama-3.1-8B OOMs with batch_size=24
else:
    MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_14B_Base"
    PAIRWISE_PATH = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/L2_Norm/al_syn_red_pairwise.csv"
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/ablation"
    _MODEL_LABEL = "Qwen3-14B-Base"
    _IS_BASE = True
    _NUM_HEADS = 40
    _DTYPE = torch.float16
    _GPUS = ["0,1", "2,3", "4,5", "6,7"]  # 2 GPUs per experiment
    _EXPERIMENTS = ['high_to_low', 'low_to_high', 'random_run1', 'random_run2']
    _EXPERIMENTS_BATCH2 = ['random_run3']
    _GPUS_BATCH2 = ["0,1"]

NUM_QUESTIONS = 100
MAX_NEW_TOKENS = 512
NUM_ABLATION_STEPS = 100
BATCH_SIZE = _BATCH_SIZE_OVERRIDE if '_BATCH_SIZE_OVERRIDE' in dir() else 24
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============== Data Loading ==============

def load_syn_red_data(csv_path: str) -> pd.DataFrame:
    """Load pairwise syn/red CSV and compute per-head Syn-Red rank."""
    df = pd.read_csv(csv_path)
    print(f"   Loaded pairwise CSV: {len(df)} rows")

    df_avg = df.groupby(['layer_1', 'head_1', 'layer_2', 'head_2']).agg({
        'syn': 'mean', 'red': 'mean'
    }).reset_index()

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
    head_df['Syn_Red_Rank'] = head_df['Syn_Red_Diff'].rank(method='first').astype(int)
    head_df['head_uid'] = head_df['Layer'].astype(str) + '_' + head_df['Head'].astype(str)

    print(f"   Total heads: {len(head_df)}")
    print(f"   Syn_Red_Rank range: [{head_df['Syn_Red_Rank'].min()}, {head_df['Syn_Red_Rank'].max()}]")
    return head_df


def load_gsm8k_samples(num_samples: int) -> List[Dict]:
    """Load GSM8K dataset samples from local JSON file."""
    gsm8k_file = "/data/zjj/Synergistic_Core/data/gsm8k/json/test.json"
    with open(gsm8k_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in indices]
    print(f"Loaded {len(samples)} questions from GSM8K")
    return samples


# ============== Model Helpers ==============

def get_layer(model, layer_idx: int):
    """Get self_attn of a specific decoder layer (supports Gemma3 and Qwen3)."""
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            return lm.model.layers[layer_idx].self_attn
        elif hasattr(lm, 'layers'):
            return lm.layers[layer_idx].self_attn
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx].self_attn
    raise ValueError(f"Cannot find layer {layer_idx}")


def get_num_heads(model) -> int:
    """Auto-detect number of attention heads."""
    attn = get_layer(model, 0)
    if hasattr(attn, 'head_dim'):
        head_dim = attn.head_dim
        q_out = attn.q_proj.out_features
        return q_out // head_dim
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    return getattr(config, 'num_attention_heads', _NUM_HEADS)


def ablate_head(model, layer_idx: int, head_idx: int, num_heads: int):
    """Ablate a specific attention head by zeroing out q_proj and o_proj weights."""
    layer = get_layer(model, layer_idx)
    q_proj = layer.q_proj
    o_proj = layer.o_proj
    hidden_dim = q_proj.weight.shape[0]
    head_dim = hidden_dim // num_heads

    with torch.no_grad():
        q_proj.weight[head_idx * head_dim:(head_idx + 1) * head_dim, :] = 0
        o_proj.weight[:, head_idx * head_dim:(head_idx + 1) * head_dim] = 0


def save_original_weights(model, layer_indices: List[int]) -> Dict:
    """Save original q_proj and o_proj weights for specified layers."""
    original_weights = {}
    with torch.no_grad():
        for layer_idx in layer_indices:
            layer = get_layer(model, layer_idx)
            original_weights[(layer_idx, 'q_proj')] = layer.q_proj.weight.clone()
            original_weights[(layer_idx, 'o_proj')] = layer.o_proj.weight.clone()
    return original_weights


def reset_model_weights(model, original_weights: Dict):
    """Reset model weights to original state."""
    with torch.no_grad():
        for key, tensor in original_weights.items():
            layer_idx, proj_type = key
            layer = get_layer(model, layer_idx)
            if proj_type == 'q_proj':
                layer.q_proj.weight.copy_(tensor)
            elif proj_type == 'o_proj':
                layer.o_proj.weight.copy_(tensor)


# ============== Prompt Formatting ==============

_FEW_SHOT_PREFIX = """Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
A: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: It takes 2/2 = <<2/2=1>>1 bolt of white fiber.
So the total amount of fabric is 2 + 1 = <<2+1=3>>3 bolts of fabric.
#### 3

Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
A: The value of the house increased by 80,000 * 1.5 = $<<80,000*1.5=120,000>>120,000.
So the new value of the house is 80,000 + 120,000 = $<<80,000-80,000+120,000=200,000>>200,000.
The profit is 200,000 - 80,000 - 50,000 = $<<200,000-80,000-50,000=70,000>>70,000.
#### 70,000

Q: """


def create_gsm8k_prompt(question: str) -> str:
    """Create 3-shot prompt for GSM8K (base models)."""
    return _FEW_SHOT_PREFIX + question + "\nA:"


def extract_gsm8k_answer(text: str) -> str:
    """Extract final answer from GSM8K output (robust for base models)."""
    # Truncate at stop sequences to avoid hallucinated extra Q&A
    for stop in ["\nQ:", "</s>", "<|im_end|>", "<|eot_id|>"]:
        if stop in text:
            text = text[:text.index(stop)]

    text = text.strip()

    # Method 1: GSM8K #### format (highest priority)
    if "####" in text:
        after_hash = text.split("####")[-1].strip()
        if after_hash:
            # Use \d+ to avoid capturing trailing periods (e.g. "36." → "36")
            numbers = re.findall(r'-?\d+', after_hash.split('\n')[0].replace(',', ''))
            if numbers:
                return numbers[0]

    # Method 2: "The answer is" patterns
    for pattern in [r'[Tt]he answer is\s+[:#]?\s*', r'[Tt]he final answer is\s+[:#]?\s*', r'[Aa]nswer\s*[:#]\s*']:
        matches = list(re.finditer(pattern, text))
        if matches:
            match = matches[-1]
            after = text[match.end():].strip().split('\n')[0].split('.')[0]
            numbers = re.findall(r'-?\d+', after.replace(',', '').replace('$', ''))
            if numbers:
                return numbers[0]

    # Method 3: Last number in the text
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return ""


# ============== Evaluation ==============

def evaluate_accuracy(model, tokenizer, prompts, ground_truth_answers, batch_size, strategy, step, total_steps):
    """Evaluate model accuracy on GSM8K samples using pre-built prompts."""
    model.eval()
    correct = 0
    total = len(prompts)

    pbar = tqdm(range(0, total, batch_size),
                desc=f"[{strategy}] Step {step}/{total_steps}",
                leave=False)

    for i in pbar:
        batch_prompts = prompts[i:i + batch_size]
        batch_answers = ground_truth_answers[i:i + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Stopping criteria: stop when "####" + number detected
        prompt_lens = [inputs['input_ids'].shape[1]] * len(batch_prompts)
        stop_criteria = StoppingCriteriaList([BatchAnswerStopCriteria(tokenizer, prompt_lens)])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=stop_criteria,
            )

        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted = extract_gsm8k_answer(generated_text)
            correct_ans = batch_answers[j]
            if predicted and correct_ans:
                if predicted.replace(',', '').replace(' ', '') == correct_ans.replace(',', '').replace(' ', ''):
                    correct += 1

        current_acc = correct / min(i + batch_size, total)
        pbar.set_postfix({'Acc': f'{current_acc:.3f}'})

    return correct / total


# ============== Worker ==============

def worker_fn(gpu_id, strategy: str, syn_df_json: str, prompts_json: str, answers_json: str, result_queue: Queue):
    """Worker process: loads model on GPU(s) and runs one ablation experiment.

    gpu_id: int for single-GPU, or str like "0,1" for multi-GPU.
    """
    gpu_str = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    if ',' not in gpu_str:
        torch.cuda.set_device(0)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    syn_df = pd.read_json(io.StringIO(syn_df_json))
    prompts = json.loads(prompts_json)
    ground_truth_answers = json.loads(answers_json)
    num_steps = NUM_ABLATION_STEPS

    print(f"[{strategy}] Starting on GPU {gpu_id}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=_DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = True

    num_heads = get_num_heads(model)
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[{strategy}] GPU(s) {gpu_str}: model loaded ({alloc:.1f} GB on dev 0), {num_heads} heads/layer")

    # Print first prompt
    print(f"\n{'='*60}")
    print(f"[{strategy}] First prompt:")
    print(f"{'─'*60}")
    print(prompts[0][:500] + ('...' if len(prompts[0]) > 500 else ''))
    print(f"{'='*60}")

    # Sort heads
    if strategy == "low_to_high":
        heads = syn_df.sort_values('Syn_Red_Rank', ascending=True)
    elif strategy == "high_to_low":
        heads = syn_df.sort_values('Syn_Red_Rank', ascending=False)
    else:
        seed = int(strategy.split('run')[-1]) * 42
        heads = syn_df.sample(frac=1, random_state=seed)

    # Exclude first and last layers from ablation
    all_layers = sorted(syn_df['Layer'].unique().astype(int).tolist())
    first_layer, last_layer = all_layers[0], all_layers[-1]
    heads = heads[(heads['Layer'] != first_layer) & (heads['Layer'] != last_layer)]
    print(f"[{strategy}] Excluding layer {first_layer} and layer {last_layer} from ablation")

    head_uids = heads['head_uid'].tolist()
    total_heads = len(head_uids)
    step_size = max(1, total_heads // num_steps)

    # Save original weights
    layer_indices = sorted(syn_df['Layer'].unique().astype(int).tolist())
    original_weights = save_original_weights(model, layer_indices)

    print(f"[{strategy}] {total_heads} heads, {num_steps} steps, ~{step_size} heads/step")

    # Run ablation
    results = {'strategy': strategy, 'num_ablated': [], 'accuracy': []}
    num_ablated = 0

    for step in range(num_steps + 1):
        if step > 0:
            n = min(step_size, total_heads - num_ablated)
            if n <= 0:
                break
            for uid in head_uids[num_ablated:num_ablated + n]:
                l, h = map(lambda x: int(float(x)), uid.split('_'))
                ablate_head(model, l, h, num_heads)
            num_ablated += n

        acc = evaluate_accuracy(model, tokenizer, prompts, ground_truth_answers, BATCH_SIZE, strategy, step, num_steps)
        results['num_ablated'].append(num_ablated)
        results['accuracy'].append(acc)

        pct = num_ablated / total_heads * 100
        print(f"[{strategy}] Step {step}/{num_steps} | Ablated {num_ablated}/{total_heads} ({pct:.1f}%) | Acc {acc:.3f}")

    reset_model_weights(model, original_weights)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    result_queue.put(results)
    print(f"[{strategy}] Done on GPU {gpu_id}")


# ============== Plot ==============

def plot_ablation_results(results_df: pd.DataFrame, output_dir: str):
    """Plot ablation results with error bars for random runs."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for strategy in ['high_to_low', 'low_to_high']:
        if strategy not in results_df['strategy'].values:
            continue
        data = results_df[results_df['strategy'] == strategy]
        if strategy == "high_to_low":
            label, color = "Synergistic First (High→Low)", "#d62728"
        else:
            label, color = "Redundant First (Low→High)", "#1f77b4"
        ax.plot(data['num_ablated'], data['accuracy'], label=label, marker='o', color=color, linewidth=2)

    # Random runs: mean ± std
    random_strategies = [s for s in results_df['strategy'].unique() if 'random' in s]
    if random_strategies:
        random_data = results_df[results_df['strategy'].isin(random_strategies)]
        stats = random_data.groupby('num_ablated').agg({'accuracy': ['mean', 'std']}).reset_index()
        stats.columns = ['num_ablated', 'mean', 'std']
        ax.plot(stats['num_ablated'], stats['mean'], label='Random (mean)', marker='o', color='gray', alpha=0.7)
        ax.fill_between(stats['num_ablated'], stats['mean'] - stats['std'], stats['mean'] + stats['std'], color='gray', alpha=0.2)

    ax.set_xlabel('Number of Ablated Attention Heads', fontsize=12)
    ax.set_ylabel('GSM8K Accuracy', fontsize=12)
    ax.set_title(f'Attention Head Ablation Study — {_MODEL_LABEL}', fontsize=14, pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, results_df['num_ablated'].max())
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ablation_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()


# ============== Main ==============

def _run_batch(experiments, gpus, syn_df_json, prompts_json, answers_json):
    """Launch a batch of experiments on specified GPUs and collect results."""
    result_queue = Queue()
    processes = []
    for exp, gpu in zip(experiments, gpus):
        p = Process(target=worker_fn, args=(gpu, exp, syn_df_json, prompts_json, answers_json, result_queue))
        p.start()
        processes.append(p)

    batch_results = []
    for _ in range(len(experiments)):
        try:
            result = result_queue.get(timeout=86400)  # 24h timeout
            batch_results.append(result)
        except Exception:
            print(f"  Warning: a worker timed out or crashed, skipping")

    for p in processes:
        p.join()

    return batch_results


def main():
    experiments = _EXPERIMENTS
    gpus = _GPUS[:len(experiments)]
    has_batch2 = bool(_EXPERIMENTS_BATCH2)

    print("=" * 60)
    print(f"{_MODEL_LABEL} Ablation Study")
    print("=" * 60)

    # Load data
    print("\n1. Loading pairwise syn/red data...")
    syn_df = load_syn_red_data(PAIRWISE_PATH)

    print("\n2. Loading GSM8K samples...")
    samples = load_gsm8k_samples(NUM_QUESTIONS)

    # Pre-build prompts and extract ground-truth answers (identical for all workers)
    prompts = [create_gsm8k_prompt(s['question']) for s in samples]
    ground_truth_answers = [extract_gsm8k_answer(s['answer']) for s in samples]
    print(f"   Built {len(prompts)} prompts, {len(ground_truth_answers)} ground-truth answers")

    print(f"\n3. Configuration:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Questions: {NUM_QUESTIONS}")
    print(f"   Heads: {len(syn_df)}")
    print(f"   Steps: {NUM_ABLATION_STEPS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Experiments (batch 1): {experiments}")
    print(f"   GPUs (batch 1): {gpus}")
    if has_batch2:
        print(f"   Experiments (batch 2): {_EXPERIMENTS_BATCH2}")
        print(f"   GPUs (batch 2): {_GPUS_BATCH2}")

    # Serialize data for workers
    syn_df_json = syn_df.to_json()
    prompts_json = json.dumps(prompts)
    answers_json = json.dumps(ground_truth_answers)

    all_results = []

    # ── Batch 1: parallel ──
    print(f"\n{'='*60}")
    print(f"Batch 1: {len(experiments)} experiments in parallel")
    print(f"{'='*60}")
    batch1_results = _run_batch(experiments, gpus, syn_df_json, prompts_json, answers_json)
    all_results.extend(batch1_results)
    print(f"Batch 1 complete: {len(batch1_results)} results collected")

    # ── Batch 2: if applicable ──
    if has_batch2:
        print(f"\n{'='*60}")
        print(f"Batch 2: {len(_EXPERIMENTS_BATCH2)} experiments in parallel")
        print(f"{'='*60}")
        batch2_results = _run_batch(
            _EXPERIMENTS_BATCH2, _GPUS_BATCH2,
            syn_df_json, prompts_json, answers_json
        )
        all_results.extend(batch2_results)
        print(f"Batch 2 complete: {len(batch2_results)} results collected")

    # Combine and save
    print("\n4. Saving results...")
    results_df = pd.DataFrame()
    for result in all_results:
        df = pd.DataFrame({
            'num_ablated': result['num_ablated'],
            'accuracy': result['accuracy'],
            'strategy': [result['strategy']] * len(result['num_ablated'])
        })
        results_df = pd.concat([results_df, df], ignore_index=True)

    csv_path = os.path.join(OUTPUT_DIR, "head_ablation.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # Plot
    print("\n5. Plotting...")
    plot_ablation_results(results_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Ablation study complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
