# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase replicates and extends "A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning" (Urbina-Rodriguez et al., 2026). It analyzes synergistic vs redundant attention heads in LLMs through PhiID (Integrated Information Decomposition) and experiments with head perturbation, ablation, and fine-tuning.

**Key finding**: A "synergistic core" of attention heads (middle layers, high Syn-Red Rank) drives model behavior more than redundant heads.

## Hardware Environment

8x NVIDIA RTX 4090 (24GB VRAM each). This constrains which models fit on single GPUs vs requiring multi-GPU parallelism. Most 4B models fit on one GPU; 12B+ models require `device_map="auto"` or manual multi-GPU. The hardcoded `CUDA_VISIBLE_DEVICES` values in scripts (e.g., GPUs 5,6,7) reflect this setup.

## Common Development Patterns

All scripts share a common skeleton: seed setting (42), hardcoded model paths (some support `MODEL_PATH` env var override), `torch_dtype=torch.bfloat16` (critical for Gemma3), `device_map="auto"`, and CSV output.

### Multi-Model Selection Pattern

Many scripts support multiple models via if/elif chains with env var override:
```python
# Example from layer_proxy_collection.py, IG_collection.py, etc.
if MODEL_NAME == "gemma3-12b-base":
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-12B-Base"
elif MODEL_NAME == "gemma3-4b-base":
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-4B-Base"
elif MODEL_NAME == "gemma3-12b-it":
    _DEFAULT_MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-12B-IT"
MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
```
Scripts using this pattern: `3_proxy_collection.py`, `3_proxy_collection_arc.py`, `layer_proxy_collection.py`, `layer_relative_change.py`, `IG_collection.py`. Most plotting scripts also have per-model if/elif blocks for output paths.

### Hardcoded Single-Model Scripts

These scripts target exactly one model:
- **Gemma-3-4B-Instruct** (root dir): `figure4a_perturbation.py`, `figure4b_math_accuracy.py`, `figure5_finetuning.py`, `activation_collection.py`
- **Qwen-3-8B-base** (root dir): `simple_layer_ablation.py`
- **Checkpoints/Qwen3_14B_Base**: `ablation_study.py`
- **Checkpoints/Qwen3_4B_Base**: `ablation_study_parallel_v2.py`, `test_baseline_accuracy.py`, `quick_test_accuracy.py`

### Additional Experiment Scripts

These scripts perform supplementary analyses and are not part of the core paper replication pipeline:
- **`src/probe_abstraction.py`**: Probing abstraction levels in intermediate representations
- **`src/cka_aba_rbf.py`**: CKA (Centered Kernel Alignment) analysis with RBF kernel
- **`src/intrinsic_dimension_ablation.py`**: Ablation study on intrinsic dimensionality
- **`src/cross_layers_pro.py`**: Cross-layer alignment matrices (no GPU needed, reads pre-collected data)
- **`src/_ISOMORPHIC.py`** / **`src/_PARAPHRASES.py`**: Isomorphic/paraphrase analysis experiments

## Supported Models

### Root Directory Models (paper replication)
| Model | Path | Layers | Heads/Layer |
|-------|------|--------|-------------|
| Gemma3-4B-Instruct | `Gemma-3-4B-Instruct/` | 34 | 8 |
| Qwen3-8B-base | `Qwen-3-8B-base/` | 36 | 32 |
| DeepSeek-V2-List | `DeepSeek-V2-List/` | — | — |

### Checkpoints/ Directory Models (extended analyses)
| Model | Path | Notes |
|-------|------|-------|
| Gemma3-12B-Base | `Checkpoints/Gemma3-12B-Base/` | Larger Gemma variant |
| Gemma3-12B-IT | `Checkpoints/Gemma3-12B-IT/` | Instruct-tuned 12B |
| Gemma3-4B-Base | `Checkpoints/Gemma3-4B-Base/` | Base (non-instruct) 4B |
| Qwen3-14B-Base | `Checkpoints/Qwen3_14B_Base/` | `compute_syn_red_rank.py` default |
| Qwen3-4B-Base | `Checkpoints/Qwen3_4B_Base/` | Used by ablation_study_parallel_v2 |
| Llama-3.1-8B | `Checkpoints/Llama-3.1-8B/` | Available but not actively used |

### Results-Only Models (no model weights in repo)
- **MoE-Gemma3-4B-IT**: Has results in `results/MoE-Gemma3-4B-IT/` but model weights are not stored here
- **MATH**: Results in `results/MATH/`

## Architecture

### Core Data Flow (universal across models/datasets)

```
1. Proxy Collection (model inference) -> time series L2 norms + vectors
2. PhiID Pairwise Computation -> pairwise syn/red CSVs
3. Aggregation -> head-level or layer-level syn_red_rank CSVs
4. Visualization / Experiments -> plots, perturbation, ablation, fine-tuning
```

### Head Identification Convention

- UID format: `"{layer_idx}_{head_idx}"` (e.g., `"15_3"`)
- Layer-level proxies (ml, al_plus_ml) use head_idx=0

### Model Structure Access Patterns

**Gemma3-4B-Instruct**:
```python
# Layer access
layer = model.model.language_model.layers[layer_idx]

# Attention projections
attn = layer.self_attn
q_proj = attn.q_proj  # [hidden_dim, hidden_dim]
k_proj = attn.k_proj  # [hidden_dim, hidden_dim]
o_proj = attn.o_proj  # [hidden_dim, hidden_dim]

# Helper function
def get_gemma3_layer(model, layer_idx):
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers[layer_idx]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    elif hasattr(model, 'layers'):
        return model.layers[layer_idx]
    else:
        raise AttributeError(f"Cannot find layer_idx={layer_idx}")
```

**Qwen3-8B-base**:
```python
# Layer access
layer = model.model.layers[layer_idx]

# Attention projections (GQA: 32 heads, 8 KV heads)
attn = layer.self_attn
q_proj = attn.q_proj  # [hidden_dim, hidden_dim]
o_proj = attn.o_proj  # [hidden_dim, hidden_dim]

# Helper function
def get_qwen3_layer(model, layer_idx: int):
    return model.model.layers[layer_idx].self_attn
```

### Proxy Types

Scripts collect intermediate layer outputs via forward hooks:
- **al**: Per-head attention outputs (L2 norms + vectors)
- **ml**: Per-layer MLP outputs (L2 norms + vectors)
- **al_plus_ml**: Combined attention + MLP (L2 norms + vectors)

**Proxy Collection Hook Pattern** (from `3_proxy_collection.py`):
```python
class ProxyCollector:
    def __init__(self, model, tokenizer, num_layers, num_heads, hidden_size, max_steps=2048):
        # Time series buffers: dict[layer/head] -> list of L2 norms
        self.al_buffer = {(i, j): [] for i in range(num_layers) for j in range(num_heads)}
        self.ml_buffer = {i: [] for i in range(num_layers)}
        self.al_plus_ml_buffer = {i: [] for i in range(num_layers)}

    def _register_hooks(self):
        # Hook 1: Capture attention output (al) per head
        def get_al_hook(layer_idx):
            def hook(module, input, output):
                # output[0]: [batch, seq, hidden] - split by heads
                # Compute L2 norm per head and store in buffer
            return hook

        # Hook 2: Capture MLP output (ml)
        def get_ml_hook(layer_idx):
            def hook(module, input, output):
                # output: [batch, seq, hidden]
                # Compute L2 norm and store in buffer
            return hook
```

## Running Experiments

### Prerequisites

```bash
pip install torch transformers datasets numpy pandas matplotlib seaborn tqdm scipy
pip install -e integrated-info-decomp/   # Required for PhiID computation
pip install verl                          # Required for GRPO fine-tuning (Figure 5)
```

Model weights must be at the paths listed above. Datasets are in `data/` subdirectories: `gsm8k`, `mmlu`, `ai2_arc`, `MATH`. No `requirements.txt` or `setup.py` exists — install manually.

### Paper Replication Pipeline (Gemma3-4B-Instruct)

```bash
# 1. Collect proxies (model inference -> L2 norm time series)
python src/3_proxy_collection.py

# 2. Compute pairwise syn/red (CPU-intensive, uses multiprocessing)
python utils/compute_al_syn_red_pairwise_mp.py

# 3. Aggregate into head-level syn_red_rank (prerequisite for all experiments)
python utils/compute_syn_red_rank.py

# 4. Run figure experiments (any order)
python utils/plot/figure3b_network.py        # Network graphs
python utils/plot/figure3c_metrics.py        # Network metrics
python src/figure4a_perturbation.py          # KL divergence perturbation
python src/figure4b_math_accuracy.py         # GSM8K accuracy with perturbation
python src/figure5_finetuning.py             # SFT vs GRPO fine-tuning
```

### Multi-Model Scripts (with MODEL_PATH override)

```bash
# Proxy collection — supports gemma3-12b-base, gemma3-4b-base, gemma3-12b-it
python src/3_proxy_collection.py

# ARC proxy collection — same models + gemma-3-4b-instruct
MODEL_PATH=/path/to/model python src/3_proxy_collection_arc.py

# Layer analysis — supports gemma3-4b-base, gemma3-12b-it, gemma-3-4b-instruct
MODEL_PATH=/path/to/model python src/layer_proxy_collection.py
MODEL_PATH=/path/to/model python src/layer_relative_change.py
MODEL_PATH=/path/to/model python src/IG_collection.py
```

### Single-Model Scripts

```bash
# Ablation studies
python src/ablation_study.py                    # Qwen3-14B, incremental ablation
python src/ablation_study_parallel_v2.py        # Qwen3-4B, multi-GPU parallel
python src/simple_layer_ablation.py             # Qwen3-8B, 1% step ablation, 8 GPUs

# Monitoring & testing
python src/monitor_ablation_progress.py         # Check ablation progress
python src/test_baseline_accuracy.py            # Qwen3-4B baseline accuracy
python src/quick_test_accuracy.py               # Qwen3-4B quick accuracy test

# Data analysis (read pre-collected data, no GPU needed)
python src/residual_stream_plot.py              # Residual stream ratios & cosine sims
python src/energy_distribution.py               # Energy distribution: E(x) = (||x_l||/||x_{l-1}||) x (1 - cos(x, h_l))
python src/cross_layers_pro.py                  # Cross-layer alignment matrices
python src/probe_abstraction.py                 # Probe abstraction levels
python src/cka_aba_rbf.py                       # CKA analysis with RBF kernel
python src/intrinsic_dimension_ablation.py      # Intrinsic dimension ablation
```

### Data Conversion

```bash
python utils/convert_mmlu_to_json.py        # MMLU parquet -> JSON
python utils/convert_arc_to_json.py         # ARC parquet -> JSON
python utils/convert_gsm8k_to_json.py       # GSM8K -> JSON
python utils/convert_math_to_json.py        # MATH -> JSON
```

### Plotting

Most plotting scripts have per-model if/elif blocks and write to `results/<Model>/Figure/`:
```bash
python utils/plot/synergy_core_syn_ratio_rank.py   # Heatmaps (has per-model sections)
python utils/plot/syn_red_scatter.py <dataset>      # Scatter plots (mmlu, arc, math)
python utils/plot/syn_red_scatter_pairwise.py       # Pairwise scatter plots
python utils/plot/kl_plot.py                         # KL divergence plots
python utils/plot/head_ablation_plot.py              # Ablation result plots
python utils/plot/IG_plot.py                         # Layer sensitivity plots
python utils/plot/layer_relative_change_plot.py      # Layer change heatmaps
python utils/plot/ablation_plot_syn_red.py           # Ablation curves
python utils/plot/ablation_v2.py                     # Ablation curves v2
python utils/plot/overview_plot.py                   # Overview plots
python utils/plot/cka_plot.py                        # CKA analysis plots
python utils/plot/energy_distribution_plot.py        # Energy distribution plots
python utils/plot/intrinsic_dimension_plot.py        # Intrinsic dimension plots
python utils/plot/probe_abstraction_plot.py          # Abstraction probing plots
```

## Critical Implementation Details

### Head Perturbation (Figure 4a, 4b)

- Access layer via `get_gemma3_layer(model, layer_idx).self_attn`
- Target **q_proj and o_proj weight matrices only** (skip k_proj, v_proj)
- **Adaptive noise**: `noise = weight_std * noise_fraction` (not fixed std)

**Implementation pattern** (from `figure4a_perturbation.py`):
```python
def deactivate_head(model, layer_idx, head_idx):
    """Ablate head by zeroing o_proj weights for that head."""
    layer = get_gemma3_layer(model, layer_idx)
    o_proj = layer.self_attn.o_proj

    num_heads = model.config.num_attention_heads
    head_dim = layer.self_attn.q_proj.weight.shape[0] // num_heads

    # o_proj shape: [hidden_size, num_heads * head_dim]
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim

    # Save original weights and zero out
    original_weights = o_proj.weight[:, start_idx:end_idx].clone()
    o_proj.weight[:, start_idx:end_idx] = 0

    return original_weights, (start_idx, end_idx)

def perturb_head(model, layer_idx, head_idx, noise_fraction=0.1):
    """Add adaptive noise to q_proj and o_proj weights."""
    layer = get_gemma3_layer(model, layer_idx)
    attn = layer.self_attn

    for proj in [attn.q_proj, attn.o_proj]:
        weight_std = proj.weight.std()
        noise = torch.randn_like(proj.weight) * weight_std * noise_fraction
        proj.weight.data += noise
```

### Head Freezing for Fine-tuning (Figure 5)

Uses **layer-wise freezing** (not true head-level):
1. Convert selected head UIDs -> layer indices
2. Freeze all parameters except `q_proj`, `k_proj`, `o_proj` in selected layers
3. True head-level freezing requires surgery on projection matrices

### KL Divergence Computation (Figure 4a)

Uses **Teacher Forcing** (not free generation):
- Fixed reference sequence (first 100 tokens from GSM8K)
- `KL = sum(P * (log P - log Q))` over vocab, averaged per position

**Implementation pattern**:
```python
def compute_kl_divergence(model_orig, model_perturbed, input_ids):
    """Compute KL divergence using teacher forcing."""
    with torch.no_grad():
        # Get logits from original model
        outputs_orig = model_orig(input_ids)
        logits_orig = outputs_orig.logits  # [batch, seq, vocab]

        # Get logits from perturbed model
        outputs_pert = model_perturbed(input_ids)
        logits_pert = outputs_pert.logits

        # Compute KL at each position
        kl_per_position = []
        for t in range(1, input_ids.shape[1]):  # Skip first token
            p = torch.softmax(logits_orig[0, t-1], dim=-1)
            q = torch.softmax(logits_pert[0, t-1], dim=-1)
            kl = (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum()
            kl_per_position.append(kl.item())

        return np.mean(kl_per_position)
```

### GSM8K Answer Extraction

```python
def extract_final_answer(text: str) -> Optional[str]:
    if "####" in text:
        return text.split("####")[-1].strip()
```

### GRPO Fine-tuning via Verl (Figure 5)

- SFT uses HuggingFace Trainer with FSDP
- GRPO uses `python -m verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo`
- Merge FSDP checkpoints: `python -m verl.model_merger merge`

## Global Constants

Paths are hardcoded in each script. Two location patterns:

```python
# Root directory models (paper replication scripts)
MODEL_PATH = "/data/zjj/Synergistic_Core/Gemma-3-4B-Instruct"
MODEL_PATH = "/data/zjj/Synergistic_Core/Qwen-3-8B-base"

# Checkpoints directory models (extended analysis scripts)
MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Gemma3-12B-Base"
MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_14B_Base"
MODEL_PATH = "/data/zjj/Synergistic_Core/Checkpoints/Qwen3_4B_Base"
# etc.

# compute_syn_red_rank.py (supports env var override)
INPUT_DIR = os.environ.get("INPUT_DIR", "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/zjj/Synergistic_Core/results/Qwen3_14B_Base/data")
```

## Dependency: Integrated Info Decomp

The `integrated-info-decomp/` directory is a vendored dependency (not a git submodule). Install with:
```bash
pip install -e integrated-info-decomp/
```

```python
from phyid.calculate import calc_PhiID
# Returns: (Syn_XY, Red_XY, Un_XY, Syn_XY_given_Z, Red_XY_given_Z, ...)
```

Pairwise computation config in multiprocessing scripts:
```python
N_WORKERS = 100    # CPU cores for parallel computation
TAU = 1
KIND = "gaussian"
REDUNDANCY = "MMI"
```

## Data File Formats

**head_syn_red_ranks.csv**:
```
Layer,Head,Syn,Red,Syn_Red_Rank
0,0,0.0234,0.0156,0.0078
```

**Pairwise CSV** (from multiprocessing scripts):
```
question_id,layer_1,head_1,layer_2,head_2,syn,red
```

**Proxy CSV** (from proxy collection):
- Columns: `question_id,step,layer_0,layer_1,...` (L2 norms per head/layer per timestep)

## Output Directory Structure

```
results/
├── Gemma3-4B-Instruct/           # Paper replication outputs
│   ├── head_syn_red_ranks.csv
│   ├── data/L2_Norm/             # Proxy CSVs from 3_proxy_collection.py
│   └── data/pairwise/            # Pairwise syn/red CSVs
├── Gemma3-12B-Base/              # Extended Gemma 12B analysis
├── Gemma3-12B-IT/
├── Gemma-3-4B-Base/
├── Gemma-3-12B-Instruct/
├── Qwen-3-8B-base/
│   ├── data/mmlu/{metadata,pairwise,plots}/
│   ├── data/ai2arc/metadata/
│   └── data/gsm8k/               # Ablation, layer analysis outputs
├── Qwen3_4B_Base/                # Qwen3 4B ablation results
├── Qwen3_14B_Base/               # Qwen3 14B (compute_syn_red_rank.py default)
├── MoE-Gemma3-4B-IT/
├── MATH/
└── Plots/                        # Shared plot outputs
```

## Common Issues

- **Out of memory during PhiID**: Reduce `N_WORKERS` or use `USE_FAST_APPROXIMATION=True`
- **Gemma3 dtype errors**: Must use `torch_dtype=torch.bfloat16` (not float16)
- **FSDP checkpoint merging**: Ensure `verl.model_merger` is installed
- **Ray initialization failures**: Verl may fail if Ray cluster conflicts with existing processes
- **Variable length sequences**: ARC/GSM8K proxy collection uses auto-stop at EOS; lengths saved to `*_effective_lengths.csv`
- **Multi-GPU scripts**: `layer_relative_change.py` uses GPUs 5,6,7; `simple_layer_ablation.py` requires 8 GPUs
- **CUDA_VISIBLE_DEVICES**: Some scripts hardcode specific GPUs (e.g., `CUDA_VISIBLE_DEVICES = "5,6,7"` in `3_proxy_collection.py`)

## Multi-GPU Processing Pattern

Scripts requiring multiple GPUs use `multiprocessing.Process` with per-GPU workers consuming from a shared `Queue`. Workers load a model copy onto their GPU and process tasks until they receive a `None` poison pill. See `simple_layer_ablation.py` or `ablation_study.py` for reference implementations.

## Plotting Conventions

Publication-quality plots use these settings:
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Nature-style color palette
COLOR_AL = "#E64B35"      # Attention (AL)
COLOR_ML = "#4DBBD5"      # MLP (ML)
COLOR_AL_ML = "#00A087"   # Combined (AL+ML)

# For saving
plt.savefig("output.png", bbox_inches="tight", dpi=300)
```

## Other Directories

- **`DeepSeek-V2-List/`**: DeepSeek-V2 model weights and custom modeling/tokenization code. Not actively used by current experiment scripts but available for future analysis.
- **`lib/`**: Contains vis.js network visualization assets (`vis-9.1.2/`, `tom-select/`, `bindings/`) used for interactive network graph rendering in notebooks. Not imported by Python scripts.
- **`CLAUDE_PROMPT.md`**: Task pipeline instructions in Chinese — describes the 7-step automation workflow for running the full experiment pipeline across different models. Useful for understanding the intended execution order.
- **`test/`**: Ad-hoc test scripts for verifying model behavior under different prompt modes (chat template vs few-shot, with/without stop tokens, varying shot counts): `chat_or_fewshot.py`, `chat_or_fewshot_3samples.py`, `chat_or_fewshot_no_stop.py`, `plot_from_data.py`.
- **Root-level experiment scripts**: `chat_or_fewshot.py`, `chat_or_fewshot_3samples.py`, `chat_or_fewshot_no_stop.py` are standalone model probing scripts; `plot_from_data.py` plots from pre-collected data.
- **Data variant directories**: `data_3samples/`, `data_no_stop/`, `data_zeroshot/` hold data collected under different few-shot configurations. Corresponding `results_3samples/`, `results_no_stop/`, `results_zeroshot/` hold outputs.
