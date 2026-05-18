"""
Collect layer-wise cosine similarity and ratio scalars from models on GSM8K.

Computes per-token-position cosine and ratio at each layer, then averages
across all token positions for each question. Saves scalar CSVs.

Metrics (per token position, per layer l):
  - cos(al, h_l)           — self-attention output vs residual stream input
  - cos(ml, h_l)           — MLP output vs residual stream input
  - cos(al+ml, h_l)        — full layer contribution vs residual stream input
  - ||al||² / ||h_l||²     — attention ratio
  - ||ml||² / ||h_l||²     — MLP ratio
  - ||al+ml||² / ||h_l||²  — full layer ratio

where:
  h_l   = residual stream entering layer l
  al    = attention contribution to residual stream
  ml    = MLP contribution to residual stream
  al + ml = h_{l+1} - h_l

Usage:
    python src/layer_proxy_collection.py                       # Gemma3-12B-IT
    python src/layer_proxy_collection.py gemma3_12b_base       # Gemma3-12B-Base
    python src/layer_proxy_collection.py gemma3_4b_base        # Gemma3-4B-Base
    python src/layer_proxy_collection.py gemma3_4b_it          # Gemma-3-4B-Instruct
    python src/layer_proxy_collection.py qwen3_8b_base         # Qwen3-8B-Base
    python src/layer_proxy_collection.py qwen3_4b_base         # Qwen3-4B-Base
    python src/layer_proxy_collection.py qwen3_14b_base        # Qwen3-14B-Base
    python src/layer_proxy_collection.py llama3_8b             # Llama-3.1-8B
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5"

import json
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============== Configuration ==============
import sys as _sys
if len(_sys.argv) > 1 and _sys.argv[1] == 'gemma3_12b_base':
    MODEL_PATH = os.environ.get("MODEL_PATH", "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-12B-Base")
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Base/data/residual_stream"
    _MODEL_LABEL = "Gemma3-12B-Base"
elif len(_sys.argv) > 1 and _sys.argv[1] == 'gemma3_4b_base':
    MODEL_PATH = os.environ.get("MODEL_PATH", "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-4B-Base")
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Base/data/residual_stream"
    _MODEL_LABEL = "Gemma3-4B-Base"
elif len(_sys.argv) > 1 and _sys.argv[1] == 'gemma3_4b_it':
    MODEL_PATH = os.environ.get("MODEL_PATH", "/data/zjj/Synergistic_Core/Gemma-3-4B-Instruct")
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma-3-4B-Instruct/data/residual_stream"
    _MODEL_LABEL = "Gemma-3-4B-Instruct"
elif len(_sys.argv) > 1 and _sys.argv[1] == 'qwen3_8b_base':
    MODEL_PATH = os.environ.get("MODEL_PATH", "/data/zjj/Synergistic_Core/Qwen-3-8B-base")
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen-3-8B-base/data/residual_stream"
    _MODEL_LABEL = "Qwen3-8B-Base"
elif len(_sys.argv) > 1 and _sys.argv[1] == 'qwen3_4b_base':
    MODEL_PATH = os.environ.get("MODEL_PATH", "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_4B_Base")
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_4B_Base/data/residual_stream"
    _MODEL_LABEL = "Qwen3-4B-Base"
elif len(_sys.argv) > 1 and _sys.argv[1] == 'qwen3_14b_base':
    MODEL_PATH = os.environ.get("MODEL_PATH", "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_14B_Base")
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data/residual_stream"
    _MODEL_LABEL = "Qwen3-14B-Base"
elif len(_sys.argv) > 1 and _sys.argv[1] == 'llama3_8b':
    MODEL_PATH = os.environ.get("MODEL_PATH", "/data/zjj/Synergistic_Core/Checkpoints/Llama-3.1-8B")
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Llama-3.1-8B/data/residual_stream"
    _MODEL_LABEL = "Llama-3.1-8B"
else:
    MODEL_PATH = os.environ.get("MODEL_PATH", "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-12B-IT")
    OUTPUT_DIR = "/data/zjj/Synergistic_Core/results/Gemma3-12B-Instruct/data/residual_stream"
    _MODEL_LABEL = "Gemma3-12B-IT"

GSM8K_DATA_DIR = "/data/zjj/Synergistic_Core/data/gsm8k"

NUM_QUESTIONS = 10
RANDOM_SEED = 42
MAX_NEW_TOKENS = 2048


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gsm8k_questions(data_dir: str, num_questions: int, seed: int) -> List[Dict]:
    random.seed(seed)
    test_file = os.path.join(data_dir, "json", "test.json")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"GSM8K data not found at {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    num_to_sample = min(num_questions, len(data))
    sampled = random.sample(data, num_to_sample)
    print(f"Sampled {num_to_sample} questions from GSM8K")
    return sampled


def create_gsm8k_prompt(question: str) -> str:
    """Create zero-shot prompt for GSM8K question."""
    return f"Question: {question}\nAnswer:"


def _batch_cosine(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Vectorized cosine similarity. v1, v2: [seq, hidden] -> [seq]."""
    norms1 = np.linalg.norm(v1, axis=1)
    norms2 = np.linalg.norm(v2, axis=1)
    dots = np.sum(v1 * v2, axis=1)
    mask = (norms1 > 1e-10) & (norms2 > 1e-10)
    result = np.zeros(v1.shape[0], dtype=np.float64)
    result[mask] = dots[mask] / (norms1[mask] * norms2[mask])
    return result


def _batch_ratio_sq(v: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Vectorized ||v||² / ||h||² per row. v, h: [seq, hidden] -> [seq]."""
    v_sq = np.sum(v ** 2, axis=1)
    h_sq = np.sum(h ** 2, axis=1)
    mask = h_sq > 1e-10
    result = np.zeros(v.shape[0], dtype=np.float64)
    result[mask] = v_sq[mask] / h_sq[mask]
    return result


class LayerMetricCollector:
    """Collect per-token cosine similarities and ratios during generation.

    Residual-state approach (unified across architectures):
        al = h_after_attn - h_l    (residual contribution of attention)
        ml = h_{l+1} - h_after_attn  (residual contribution of MLP)
        al + ml = h_{l+1} - h_l

    Hook targets differ by architecture:
        Gemma3:   al ← post_attention_layernorm output,  ml ← post_feedforward_layernorm output
        Qwen3:    al ← self_attn output,                ml ← mlp output
    Both targets output exactly what gets added to the residual stream.
    """

    def __init__(self, model, tokenizer, num_layers: int, num_heads: int, hidden_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Scalar accumulators per layer (summed across all token positions)
        self.cos_al_accum = np.zeros(num_layers, dtype=np.float64)
        self.cos_ml_accum = np.zeros(num_layers, dtype=np.float64)
        self.cos_al_plus_ml_accum = np.zeros(num_layers, dtype=np.float64)
        self.ratio_al_accum = np.zeros(num_layers, dtype=np.float64)
        self.ratio_ml_accum = np.zeros(num_layers, dtype=np.float64)
        self.ratio_al_plus_ml_accum = np.zeros(num_layers, dtype=np.float64)
        self.token_count = 0

        # Running residual stream
        self._h_running = None                          # [seq, hidden]
        self._h_at_layer_start = {}                     # layer_idx -> h_l

        self.hooks = []
        self._register_hooks()

    def _get_layers(self):
        if hasattr(self.model, 'language_model'):
            lm = self.model.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                return lm.model.layers
            elif hasattr(lm, 'layers'):
                return lm.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        raise ValueError(f"Unsupported model structure: {type(self.model)}")

    def _get_embed_tokens(self):
        if hasattr(self.model, 'language_model'):
            lm = self.model.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'embed_tokens'):
                return lm.model.embed_tokens
            elif hasattr(lm, 'embed_tokens'):
                return lm.embed_tokens
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens
        return None

    def _register_hooks(self):
        layers = self._get_layers()
        embed_tokens = self._get_embed_tokens()
        is_gemma3 = hasattr(layers[0], 'post_feedforward_layernorm')

        # ── Embedding hook: h_0 = embed_tokens(input_ids) ──
        if embed_tokens is not None:
            def emb_hook(module, input, output):
                self._h_running = output.detach().cpu().float().numpy()[0].copy()
            self.hooks.append(embed_tokens.register_forward_hook(emb_hook))

        # ── Per-layer hooks ──
        for idx in range(len(layers)):
            layer = layers[idx]

            # Select target modules based on architecture
            if is_gemma3:
                # Gemma3: post-norm outputs are the residual contributions
                al_module = layer.post_attention_layernorm
                ml_module = layer.post_feedforward_layernorm
            else:
                # Qwen3/standard: raw attn/mlp outputs are the residual contributions
                al_module = layer.self_attn
                ml_module = layer.mlp

            # ── AL hook: al = h_after_attn - h_l ──
            def get_al_hook(li):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    al = output.detach().cpu().float().numpy()[0]   # [seq, hidden]
                    h_l = self._h_running
                    self._h_at_layer_start[li] = h_l.copy() if h_l is not None else None

                    if h_l is not None and al.shape[0] == h_l.shape[0]:
                        self.cos_al_accum[li] += _batch_cosine(al, h_l).sum()
                        self.ratio_al_accum[li] += _batch_ratio_sq(al, h_l).sum()
                        self._h_running = h_l + al   # h_l → h_l + al = h_after_attn
                return hook
            self.hooks.append(al_module.register_forward_hook(get_al_hook(idx)))

            # ── ML hook: ml = h_{l+1} - h_after_attn ──
            def get_ml_hook(li):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    ml = output.detach().cpu().float().numpy()[0]
                    h_l = self._h_at_layer_start.get(li)
                    h_after_attn = self._h_running   # h_l + al

                    if h_l is not None and h_after_attn is not None and ml.shape[0] == h_l.shape[0]:
                        al = h_after_attn - h_l
                        self.cos_ml_accum[li] += _batch_cosine(ml, h_l).sum()
                        self.cos_al_plus_ml_accum[li] += _batch_cosine(al + ml, h_l).sum()
                        self.ratio_ml_accum[li] += _batch_ratio_sq(ml, h_l).sum()
                        self.ratio_al_plus_ml_accum[li] += _batch_ratio_sq(al + ml, h_l).sum()

                    self.token_count += ml.shape[0]
                    if h_after_attn is not None and ml.shape[0] == h_after_attn.shape[0]:
                        self._h_running = h_after_attn + ml   # h_after_attn + ml = h_{l+1}
                    self._h_at_layer_start.pop(li, None)
                return hook
            self.hooks.append(ml_module.register_forward_hook(get_ml_hook(idx)))

    def collect(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
        """Run generation and collect metrics. Returns per-layer averages."""
        self.cos_al_accum[:] = 0
        self.cos_ml_accum[:] = 0
        self.cos_al_plus_ml_accum[:] = 0
        self.ratio_al_accum[:] = 0
        self.ratio_ml_accum[:] = 0
        self.ratio_al_plus_ml_accum[:] = 0
        self.token_count = 0
        self._h_running = None
        self._h_at_layer_start.clear()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs['input_ids'].shape[1]

        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            stop_token_ids = [eos_token_id]
        else:
            stop_token_ids = list(eos_token_id) if eos_token_id else []
        try:
            if '<end_of_turn>' in self.tokenizer.get_vocab():
                end_of_turn_id = self.tokenizer.convert_tokens_to_ids('<end_of_turn>')
                if end_of_turn_id not in stop_token_ids:
                    stop_token_ids.append(end_of_turn_id)
        except:
            pass
        if not stop_token_ids:
            stop_token_ids = [1, 106]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_hidden_states=False,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else 0,
                eos_token_id=stop_token_ids,
            )

        effective_length = len(outputs.sequences[0]) - input_len

        if self.token_count > 0:
            per_layer_count = self.token_count / self.num_layers
            result = {
                'cos_al': self.cos_al_accum / per_layer_count,
                'cos_ml': self.cos_ml_accum / per_layer_count,
                'cos_al_plus_ml': self.cos_al_plus_ml_accum / per_layer_count,
                'ratio_al': self.ratio_al_accum / per_layer_count,
                'ratio_ml': self.ratio_ml_accum / per_layer_count,
                'ratio_al_plus_ml': self.ratio_al_plus_ml_accum / per_layer_count,
            }
        else:
            result = {k: np.zeros(self.num_layers) for k in
                      ['cos_al', 'cos_ml', 'cos_al_plus_ml',
                       'ratio_al', 'ratio_ml', 'ratio_al_plus_ml']}

        return result, effective_length, outputs, input_len

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def save_results(all_metrics, num_layers, output_dir):
    """Save per-question per-layer scalar metrics to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    num_questions = len(all_metrics)

    metric_names = ['cos_al', 'cos_ml', 'cos_al_plus_ml',
                    'ratio_al', 'ratio_ml', 'ratio_al_plus_ml']

    for metric in metric_names:
        rows = []
        for q_id in range(num_questions):
            for layer in range(num_layers):
                rows.append({
                    'question_id': q_id,
                    'layer': layer,
                    'value': all_metrics[q_id][metric][layer]
                })
        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, f"{metric}.csv")
        df.to_csv(path, index=False)
        print(f"  {path}")


def main():
    print("=" * 60)
    print(f"Layer Metric Collection ({_MODEL_LABEL})")
    print("=" * 60)

    set_seed(RANDOM_SEED)

    print(f"\nLoading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    if hasattr(model.config, 'text_config'):
        config = model.config.text_config
    else:
        config = model.config

    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    layers_ref = None
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            layers_ref = lm.model.layers
            first_layer_attn = lm.model.layers[0].self_attn
        elif hasattr(lm, 'layers'):
            layers_ref = lm.layers
            first_layer_attn = lm.layers[0].self_attn
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers_ref = model.model.layers
        first_layer_attn = model.model.layers[0].self_attn
    else:
        first_layer_attn = None

    if first_layer_attn is not None:
        head_dim = first_layer_attn.head_dim if hasattr(first_layer_attn, 'head_dim') else hidden_size // getattr(config, 'num_attention_heads', 16)
        num_heads = first_layer_attn.q_proj.out_features // head_dim
    else:
        num_heads = getattr(config, 'num_attention_heads', 16)

    is_gemma3 = layers_ref is not None and hasattr(layers_ref[0], 'post_feedforward_layernorm')
    print(f"  Layers: {num_layers}, Heads: {num_heads}, Hidden: {hidden_size}")
    print(f"  Architecture: {'Gemma3 (post-norm)' if is_gemma3 else 'Standard (pre-norm)'}")

    questions = load_gsm8k_questions(GSM8K_DATA_DIR, NUM_QUESTIONS, RANDOM_SEED)

    collector = LayerMetricCollector(model, tokenizer, num_layers, num_heads, hidden_size)

    all_metrics = {}
    print(f"\nProcessing {len(questions)} questions...")
    for q_idx, question in enumerate(tqdm(questions)):
        question_text = question.get('question', '') if isinstance(question, dict) else str(question)
        prompt = create_gsm8k_prompt(question_text)

        metrics, effective_length, outputs, input_len = collector.collect(prompt, MAX_NEW_TOKENS)
        all_metrics[q_idx] = metrics

        if q_idx == 0:
            generated_ids = outputs.sequences[0][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"\n{'='*60}")
            print(f"[Question {q_idx}]")
            print(f"{'─'*60}")
            print(f"[PROMPT]:\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
            print(f"{'─'*60}")
            print(f"[RESPONSE] (len={effective_length} tokens):\n{generated_text[:1000]}{'...' if len(generated_text) > 1000 else ''}")
            print(f"{'='*60}")

    collector.remove_hooks()

    print(f"\nSaved results to:")
    save_results(all_metrics, num_layers, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
