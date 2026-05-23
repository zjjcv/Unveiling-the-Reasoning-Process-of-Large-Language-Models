# Synergistic Core in LLMs

**Investigating Internal Information Processing Mechanisms and Abstraction Capabilities of Large Language Models through Integrated Information Decomposition**

Reimplementation and extension of [*A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning*](https://arxiv.org/abs/2505.17552) (Urbina-Rodriguez et al., 2025). This project applies **PhiID (Integrated Information Decomposition)** to decompose attention head interactions into *synergistic* and *redundant* components, revealing a "synergistic core" of attention heads concentrated in middle layers that drives model behavior and abstract reasoning.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Supported Models](#supported-models)
- [Getting Started](#getting-started)
- [Pipeline Details](#pipeline-details)
  - [Stage 1: Proxy Collection](#stage-1-proxy-collection)
  - [Stage 2: PhiID Pairwise Computation](#stage-2-phiid-pairwise-computation)
  - [Stage 3: Syn-Red Rank Aggregation](#stage-3-syn-red-rank-aggregation)
  - [Stage 4: Multi-Perspective Analysis](#stage-4-multi-perspective-analysis)
  - [Stage 5: Visualization & Network Analysis](#stage-5-visualization--network-analysis)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Citation](#citation)
- [License](#license)

---

## Overview

Transformers process information through layers of attention heads, but how these heads interact remains poorly understood. Using Integrated Information Decomposition (PhiID), we decompose the joint information that pairs of attention heads carry about the model's output into:

| Component | Meaning |
|-----------|---------|
| **Synergy** | Information emerging *only* from the joint state of two heads — irreducible, complementary interactions |
| **Redundancy** | Information available from *either* head alone — overlapping, duplicative processing |

Heads ranked high on synergy (the **synergistic core**) cluster in middle layers and disproportionately drive model performance. Heads ranked high on redundancy serve as distributed, robust backups.

This project provides a complete pipeline from raw model inference through PhiID computation, multi-perspective validation experiments, and publication-quality visualization.

```
                         Synergistic Core Discovery Pipeline

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        Stage 1: Proxy Collection                       │
  │                                                                         │
  │   GSM8K / MMLU / ARC ──► Model Inference ──► Forward Hooks ──► L2 Norm │
  │                                                   Time Series (CSV)    │
  │                           3_proxy_collection.py                         │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    Stage 2: PhiID Pairwise Computation                  │
  │                                                                         │
  │            L2 Norm Time Series ──► calc_PhiID() ──► Pairwise Syn/Red    │
  │                  (integrated-info-decomp)              (CSV)            │
  │              compute_al_syn_red_pairwise_mp.py                          │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                   Stage 3: Rank Aggregation                             │
  │                                                                         │
  │          Pairwise Syn/Red ──► Sum/Aggregate ──► Syn_Red_Rank per Head  │
  │                                                          (CSV)         │
  │                   compute_syn_red_rank.py                               │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐──────────────┐────────────┐
                    ▼            ▼            ▼              ▼            ▼
  ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐
  │   Stage 4: Multi-Perspective Analysis & Validation                    │
  │                                                                        │
  │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
  │  │ Cosine Sim &     │  │  CKA Similarity  │  │  Abstraction Probe   │ │
  │  │ Residual Stream  │  │  (ABA / RBF)     │  │  (Rule vs Token)     │ │
  │  │ Analysis         │  │                  │  │                      │ │
  │  │ layer_proxy_     │  │  cka_aba_rbf.py  │  │  probe_              │ │
  │  │ collection.py    │  │                  │  │  abstraction.py      │ │
  │  └─────────────────┘  └──────────────────┘  └──────────────────────┘ │
  │                                                                        │
  │  ┌─────────────────────┐  ┌────────────────────────────────────────┐ │
  │  │ Intrinsic Dimension │  │  Layer Relative Change (4 Experiments)  │ │
  │  │ under Ablation      │  │  Full Ablation / Future Prediction /   │ │
  │  │                     │  │  Circuit Localization (All / Future)    │ │
  │  │ intrinsic_dimension │  │                                        │ │
  │  │ _ablation.py        │  │  layer_relative_change.py              │ │
  │  └─────────────────────┘  └────────────────────────────────────────┘ │
  │                                                                        │
  │  ┌──────────────────────────────────────────────────────────────────┐ │
  │  │  Head Ablation Study (Synergistic-first vs Redundant-first)       │ │
  │  │  ablation_study.py                                                │ │
  │  └──────────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                 Stage 5: Visualization & Network Analysis               │
  │                                                                         │
  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────────┐ │
  │  │  Undirected   │  │   Global      │  │  Publication Plots            │ │
  │  │  Network      │  │   Efficiency  │  │  (Scatter, Heatmap, Curve,    │ │
  │  │  Graphs       │  │   &           │  │   CKA, Probe, ID, Ablation)   │ │
  │  │               │  │   Modularity  │  │                               │ │
  │  │ figure3b_     │  │ figure3c_     │  │ syn_red_scatter_pairwise.py   │ │
  │  │ network.py    │  │ metrics.py    │  │ probe_abstraction_plot.py     │ │
  │  │               │  │               │  │ layer_relative_change_plot.py │ │
  │  │               │  │               │  │ intrinsic_dimension_plot.py   │ │
  │  │               │  │               │  │ head_ablation_plot.py         │ │
  │  │               │  │               │  │ cka_plot.py                   │ │
  │  └──────────────┘  └──────────────┘  └───────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Supported Models

| Model | Parameters | Layers | Heads/Layer | Location |
|-------|-----------|--------|-------------|----------|
| **Gemma3-4B-Instruct** | 4B | 34 | 8 | `Gemma-3-4B-Instruct/` |
| **Gemma3-4B-Base** | 4B | 34 | 8 | `Checkpoints/Gemma3-4B-Base/` |
| **Gemma3-12B-Base** | 12B | 48 | 16 | `Checkpoints/Gemma3-12B-Base/` |
| **Gemma3-12B-IT** | 12B | 48 | 16 | `Checkpoints/Gemma3-12B-IT/` |
| **Qwen3-8B-Base** | 8B | 36 | 32 (GQA: 8 KV) | `Qwen-3-8B-base/` |
| **Qwen3-4B-Base** | 4B | 36 | 24 | `Checkpoints/Qwen3_4B_Base/` |
| **Qwen3-14B-Base** | 14B | 40 | 40 | `Checkpoints/Qwen3_14B_Base/` |

Model weights are not included in this repository. Download from HuggingFace and place at the paths above.

---

## Getting Started

### Prerequisites

```bash
# Core dependencies
pip install torch transformers datasets numpy pandas matplotlib seaborn tqdm scipy

# PhiID computation (vendored dependency)
pip install -e integrated-info-decomp/

# Optional: for GRPO fine-tuning experiments
pip install verl
```

### Datasets

Datasets should be placed in `data/` subdirectories. Supported datasets:

- **GSM8K** — Math reasoning (`data/gsm8k/`)
- **MMLU** — Multi-task language understanding (`data/mmlu/`)
- **AI2-ARC** — Science reasoning (`data/ai2_arc/`)
- **MATH** — Competition mathematics (`data/MATH/`)

Use the conversion utilities in `utils/` to prepare data:
```bash
python utils/convert_gsm8k_to_json.py
python utils/convert_mmlu_to_json.py
python utils/convert_arc_to_json.py
python utils/convert_math_to_json.py
```

### Quick Start

```bash
# 1. Collect proxies (L2 norm time series via forward hooks)
python src/3_proxy_collection.py

# 2. Compute pairwise synergy/redundancy via PhiID
python utils/compute_al_syn_red_pairwise_mp.py

# 3. Aggregate into per-head syn_red_rank
python utils/compute_syn_red_rank.py

# 4. Run analyses and generate plots
python src/layer_proxy_collection.py          # Residual stream analysis
python src/cka_aba_rbf.py                     # CKA similarity
python src/probe_abstraction.py               # Abstraction probing
python src/intrinsic_dimension_ablation.py    # Intrinsic dimension
python src/layer_relative_change.py           # Layer-level interventions
python src/ablation_study.py                  # Head ablation study
```

---

## Pipeline Details

### Stage 1: Proxy Collection

Collect intermediate representations during model inference via forward hooks. The proxy time series capture how information evolves across layers and attention heads during generation.

**Scripts:**

| Script | Purpose | Output |
|--------|---------|--------|
| `src/3_proxy_collection.py` | Per-head attention (AL), per-layer MLP (ML), and combined (AL+ML) L2 norms | `results/<Model>/data/L2_Norm/*.csv` |
| `src/layer_proxy_collection.py` | Cosine similarity and energy ratio between each component and residual stream | `results/<Model>/data/residual_stream/*.csv` |

**Proxy types collected:**
- **AL (Attention Layer)**: L2 norm of each attention head's output per timestep — captures per-head activation dynamics
- **ML (Multi-Layer Perceptron)**: L2 norm of each layer's MLP output — captures feedforward processing
- **AL+ML (Combined)**: L2 norm of attention + MLP combined output — captures total layer contribution

```
  Proxy Collection via Forward Hooks
  ═══════════════════════════════════

  Input Token Sequence
         │
         ▼
  ┌──────────────┐     Hook 1: Split by heads → L2 norm per head → al_buffer
  │  Layer L      │     Hook 2: MLP output → L2 norm → ml_buffer
  │  ┌────────┐  │     Hook 3: Attn+MLP output → L2 norm → al_plus_ml_buffer
  │  │ Self-  │  │
  │  │ Attn   │──┼──► Per-Head L2 Norms  ──►  Time Series CSV
  │  │(H heads)│ │    [h₁, h₂, ..., hₕ]
  │  └───┬────┘  │
  │      ▼       │
  │  ┌────────┐  │
  │  │  MLP   │──┼──► Layer L2 Norm  ──►  Time Series CSV
  │  └────────┘  │
  └──────────────┘
         │
         ▼  (repeat for all layers)
```

Supports multi-model selection via `MODEL_NAME` environment variable:
```bash
MODEL_NAME=gemma3-12b-base python src/3_proxy_collection.py
MODEL_NAME=gemma3-4b-base python src/3_proxy_collection.py
```

---

### Stage 2: PhiID Pairwise Computation

The core computational stage. Uses **Integrated Information Decomposition (PhiID)** from the vendored `integrated-info-decomp/` library to compute pairwise synergy and redundancy between all attention head pairs.

**Script:** `utils/compute_al_syn_red_pairwise_mp.py`

**PhiID computation parameters:**
```python
from phyid.calculate import calc_PhiID

# Returns: (Syn_XY, Red_XY, Un_XY, Syn_XY_given_Z, Red_XY_given_Z, ...)
result = calc_PhiID(time_series_X, time_series_Y, tau=1, kind="gaussian")
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TAU` | 1 | Time lag for PhiID |
| `KIND` | "gaussian" | Assume Gaussian statistics |
| `REDUNDANCY` | "MMI" | Minimum Mutual Information redundancy measure |
| `N_WORKERS` | 100 | CPU cores for parallel computation |

**Output:** Pairwise CSV files containing synergy and redundancy for every head pair:
```csv
question_id,layer_1,head_1,layer_2,head_2,syn,red
0,0,0,1,2,0.0023,0.0156
```

> **Note:** This stage is CPU-intensive. For models with N heads, it computes N*(N-1)/2 pairs per question. Reduce `N_WORKERS` if memory is constrained.

---

### Stage 3: Syn-Red Rank Aggregation

Aggregate pairwise synergy/redundancy into per-head scores.

**Script:** `utils/compute_syn_red_rank.py`

Computes for each head:
- **Syn**: Sum of pairwise synergy values
- **Red**: Sum of pairwise redundancy values
- **Syn_Red_Rank**: Syn - Red (positive = more synergistic)

```csv
Layer,Head,Syn,Red,Syn_Red_Rank
15,3,0.4521,0.1203,0.3318
```

This rank determines which heads belong to the **synergistic core** (high rank) vs the **redundant periphery** (low rank).

---

### Stage 4: Multi-Perspective Analysis

Seven complementary analyses validate and extend the synergistic core finding:

#### 4.1 Residual Stream Analysis (Cosine Similarity & Energy Ratio)

**Scripts:** `src/layer_proxy_collection.py` → `src/residual_stream_plot.py`

Measures how much each layer component (attention, MLP, combined) aligns with the residual stream:

```
  Residual Stream Decomposition
  ═════════════════════════════

  h_l = h_{l-1} + AL_l + ML_l     (residual stream at layer l)

  Metrics:
  ├── cos(AL_l, h_l)     → How aligned is attention with the residual stream?
  ├── cos(ML_l, h_l)     → How aligned is the MLP with the residual stream?
  ├── cos(AL_l+ML_l, h_l) → How aligned is the combined output?
  └── ||component||² / ||h_l||²  → Energy ratio (relative contribution)
```

**Outputs:** Stacked bar plots of energy ratios and cosine similarities across layers.

---

#### 4.2 CKA Similarity (Abstract Rule Extraction)

**Script:** `src/cka_aba_rbf.py` → `utils/plot/cka_plot.py`

Tests whether the synergistic core extracts **abstract rules** independent of surface form. Uses two disjoint vocabularies with identical syntactic patterns:

```
  CKA Abstract Rule Extraction Test
  ══════════════════════════════════

  Vocabulary A: "cat cat dog cat"     (AABA pattern)
  Vocabulary B: "sun sun moon sun"    (AABA pattern, different tokens)

  ┌─────────────┐     ┌─────────────┐
  │  Vocab A     │     │  Vocab B     │
  │  Sequences   │     │  Sequences   │
  └──────┬──────┘     └──────┬──────┘
         │                    │
         ▼                    ▼
  ┌─────────────┐     ┌─────────────┐
  │  Hidden      │     │  Hidden      │
  │  States A    │     │  States B    │
  └──────┬──────┘     └──────┬──────┘
         │                    │
         └────────┬───────────┘
                  ▼
         CKA(HS_A, HS_B)  per layer
         ─────────────────────────
         High CKA in synergistic core layers
         → Abstract rules extracted independently of tokens
```

Supports both **Linear CKA** and **RBF CKA** kernels.

---

#### 4.3 Abstraction Probe (Rule vs Token Classification)

**Script:** `src/probe_abstraction.py` → `utils/plot/probe_abstraction_plot.py`

Trains linear probes on intermediate representations to disentangle **abstract rule encoding** from **surface token encoding**:

- **Rule Probe** (6-class): Predict which syntactic pattern (AABA, AABB, ABBA, ABAB, AAAB, ABBB) — should peak in synergistic core
- **Token Probe** (30-class): Predict specific token identity — should dip in synergistic core (abstracting away surface form)

```
  Probe Accuracy by Layer
  ══════════════════════

  Accuracy
     ▲
     │    Rule Probe ═══╗
     │                  ║╔══╗
     │             ╔════╝║  ║
     │         ╔═══╝     ║  ╚══╗
     │     ╔═══╝         ║     ║
     │ ╔═══╝             ╚══╗  ║
     │─╢                    ╚══╝── Token Probe
     │ │    Synergistic Core  │
     └─┼─────────────────────┼──────► Layer
       │   (high rule, low   │
       │    token accuracy)  │
```

---

#### 4.4 Intrinsic Dimension under Ablation

**Script:** `src/intrinsic_dimension_ablation.py` → `utils/plot/intrinsic_dimension_plot.py`

Measures the **intrinsic dimensionality** of hidden states when synergistic or redundant heads are ablated. Uses k-NN Maximum Likelihood Estimation on GPU.

```
  Intrinsic Dimension Experiment
  ══════════════════════════════

  Three conditions:
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │   Baseline       │  │  Syn Ablation   │  │  Red Ablation   │
  │   (no ablation)  │  │  (top syn heads │  │  (top red heads │
  │                  │  │   zeroed)       │  │   zeroed)       │
  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
           │                    │                     │
           ▼                    ▼                     ▼
    Collect Hidden       Collect Hidden        Collect Hidden
    States               States                States
           │                    │                     │
           └────────┬───────────┘─────────────────────┘
                    ▼
           k-NN MLE Intrinsic Dimension
           ──────────────────────────────
           Key finding: Ablating synergistic heads
           increases intrinsic dimension (more complex,
           less structured representations)
```

---

#### 4.5 Layer Relative Change (Four Intervention Experiments)

**Script:** `src/layer_relative_change.py` → `utils/plot/layer_relative_change_plot.py`

Four complementary experiments measuring how perturbing layer s affects layer l (where l > s):

| # | Experiment | Method | Question Answered |
|---|-----------|--------|-------------------|
| 1 | **Full Ablation** | Skip layer s entirely for all tokens | Which layers are essential? |
| 2 | **Future Prediction** | Skip layer s only during prefill (first pass) | Does layer s affect future token generation? |
| 3 | **Circuit Localization (All)** | Subtract layer s's contribution from all downstream layers | How much does layer s contribute to layer l? |
| 4 | **Circuit Localization (Future)** | Same as #3 but only for future positions | Which circuit connections are predictive vs. post-hoc? |

```
  Layer Intervention Schematic
  ════════════════════════════

  Layer:    0    1    2    ...   s   ...   l   ...   L
            │    │    │         │         │         │
            ▼    ▼    ▼         ▼         ▼         ▼
  Normal:  h₁ → h₂ → h₃ → ... hₛ → ... hₗ → ... hₗ

  Ablation:  h₁ → h₂ → ... → hₛ₋₁ ──✕── hₛ₊₁ → ... hₗ
                                       (skip)

  Circuit:   h₁ → h₂ → ... → hₛ → ... hₗ - contribution(s→l)
                                              (subtract s's input)
```

**Output:** Four upper-triangular heatmap matrices showing intervention impact across (source_layer, target_layer) pairs.

---

#### 4.6 Head Ablation Study

**Script:** `src/ablation_study.py` → `utils/plot/head_ablation_plot.py`

Systematically ablates attention heads in different priority orders to validate that synergistic heads are more critical:

```
  Head Ablation Protocol
  ═════════════════════

  Step 1: Rank all heads by Syn_Red_Rank (from Stage 3)
  Step 2: Ablate heads incrementally (1%, 5%, 10%, ..., 100%)
  Step 3: Evaluate on GSM8K after each ablation step

  Two ablation orders:
  ┌──────────────────────────────────────────────────────┐
  │  Synergistic-first:  Remove top-syn heads first      │
  │  ──── Accuracy drops rapidly ──────────────────►     │
  │                                                      │
  │  Redundant-first:    Remove top-red heads first      │
  │  ──── Accuracy drops slowly ──────────────────►      │
  └──────────────────────────────────────────────────────┘

  Conclusion: Synergistic heads are disproportionately
              important for model performance
```

**Key result:** Ablating 10% of synergistic heads causes more performance degradation than ablating 50% of redundant heads.

---

### Stage 5: Visualization & Network Analysis

Generate publication-quality figures and network-level analysis.

#### 5.1 Network Graphs (Undirected)

**Script:** `utils/plot/figure3b_network.py`

Creates force-directed network graphs where nodes are attention heads and edges represent pairwise synergy (or redundancy) strength:

```
  Network Graph Construction
  ══════════════════════════

  Pairwise Syn/Red Matrix     Adjacency Matrix      Force-Directed Layout
  ┌─────────────┐         ┌─────────────┐        ┌─────────────────┐
  │ h₁  h₂  h₃ │         │ 0   0.8 0.2 │        │    ○──○         │
  │ ─────────── │  ───►   │ 0.8  0  0.5 │  ───►  │   / \  \       │
  │ 0.8 ...     │         │ 0.2 0.5  0  │        │  ○   ○──○      │
  └─────────────┘         └─────────────┘        └─────────────────┘
                                                  Node color = layer depth
                                                  Edge width = syn/red weight
```

Generates separate **synergy network** and **redundancy network** graphs per model.

---

#### 5.2 Global Efficiency & Modularity

**Script:** `utils/plot/figure3c_metrics.py`

Computes two network-level metrics comparing synergistic vs redundant connectivity:

| Metric | Synergy Network | Redundancy Network | Interpretation |
|--------|----------------|-------------------|----------------|
| **Global Efficiency** | Lower | Higher | Redundancy enables distributed information flow |
| **Modularity** | Higher | Lower | Synergy forms specialized, clustered sub-circuits |

```
  Network Metrics Comparison
  ══════════════════════════

  Global Efficiency          Modularity
  ┌──────────────────┐      ┌──────────────────┐
  │  ██████ Red       │      │  ███████████ Syn  │
  │  ███     Syn      │      │  ████      Red    │
  └──────────────────┘      └──────────────────┘

  → Synergy: modular, specialized       → Redundancy: distributed, robust
```

---

#### 5.3 Additional Plotting Scripts

| Script | Output |
|--------|--------|
| `utils/plot/syn_red_scatter_pairwise.py` | Synergy vs Redundancy rank scatter with Spearman correlation |
| `utils/plot/probe_abstraction_plot.py` | Rule accuracy & token accuracy curves by layer |
| `utils/plot/layer_relative_change_plot.py` | Upper-triangular heatmaps for 4 intervention experiments |
| `utils/plot/intrinsic_dimension_plot.py` | Intrinsic dimension curves (baseline + ablation) |
| `utils/plot/head_ablation_plot.py` | Accuracy curves & drop comparison for head ablation |
| `utils/plot/cka_plot.py` | CKA similarity by layer with core region highlighted |

---

## Project Structure

```
Synergistic_Core/
├── src/                                    # Core experiment scripts
│   ├── 3_proxy_collection.py               #   L2 norm proxy collection (AL/ML/AL+ML)
│   ├── 3_proxy_collection_arc.py           #   Proxy collection for ARC dataset
│   ├── layer_proxy_collection.py           #   Residual stream cosine similarity & ratio
│   ├── layer_relative_change.py            #   4 layer intervention experiments
│   ├── probe_abstraction.py                #   Rule vs token abstraction probes
│   ├── cka_aba_rbf.py                      #   CKA similarity (linear + RBF kernel)
│   ├── intrinsic_dimension_ablation.py     #   Intrinsic dimension under head ablation
│   ├── ablation_study.py                   #   Head ablation (syn-first vs red-first)
│   ├── residual_stream_plot.py             #   Residual stream visualization (no GPU)
│   ├── activation_collection.py            #   Activation collection for PhiID
│   ├── figure4a_perturbation.py            #   KL divergence perturbation analysis
│   ├── figure4b_math_accuracy.py           #   GSM8K accuracy with perturbation
│   ├── figure5_finetuning.py               #   SFT vs GRPO fine-tuning
│   └── ...                                 #   Additional analysis scripts
│
├── utils/                                  # Computation & plotting utilities
│   ├── compute_al_syn_red_pairwise_mp.py   #   PhiID pairwise computation (multi-core)
│   ├── compute_syn_red_rank.py             #   Per-head syn_red_rank aggregation
│   ├── convert_gsm8k_to_json.py            #   Dataset converters
│   ├── plot/                               #   Publication-quality plotting
│   │   ├── figure3b_network.py             #     Undirected network graphs
│   │   ├── figure3c_metrics.py             #     Global efficiency & modularity
│   │   ├── syn_red_scatter_pairwise.py     #     Syn vs Red scatter plots
│   │   ├── probe_abstraction_plot.py       #     Abstraction probe plots
│   │   ├── layer_relative_change_plot.py   #     Layer intervention heatmaps
│   │   ├── intrinsic_dimension_plot.py     #     Intrinsic dimension plots
│   │   ├── head_ablation_plot.py           #     Head ablation result plots
│   │   ├── cka_plot.py                     #     CKA similarity plots
│   │   └── ...                             #     Additional plot scripts
│   └── ...                                 #   Additional utilities
│
├── integrated-info-decomp/                 # Vendored PhiID library
│   ├── phyid/
│   │   ├── calculate.py                    #   calc_PhiID() main entry point
│   │   ├── measures.py                     #   PhiID measure implementations
│   │   └── utils.py
│   └── setup.py
│
├── data/                                   # Datasets (not tracked)
│   ├── gsm8k/
│   ├── mmlu/
│   ├── ai2_arc/
│   └── MATH/
│
├── results/                                # Experiment outputs (not tracked)
│   ├── Gemma3-4B-Instruct/                 #   Per-model result directories
│   ├── Gemma3-12B-Base/
│   ├── Qwen-3-8B-base/
│   └── ...
│
├── Checkpoints/                            # Model weights (not tracked)
├── Gemma-3-4B-Instruct/                    # Primary model weights
├── Qwen-3-8B-base/                         # Secondary model weights
│
├── CLAUDE.md                               # AI assistant instructions
├── LICENSE                                 # Apache 2.0
└── README.md                               # This file
```

---

## Key Findings

1. **Synergistic core exists in middle layers** — Attention heads with high synergy-to-redundancy ratios cluster in the middle third of transformer layers across all tested models.

2. **Synergistic heads are disproportionately important** — Ablating 10% of synergistic heads degrades performance more than ablating 50% of redundant heads.

3. **Abstract rules are extracted in the core** — CKA similarity and probing experiments show the synergistic core encodes abstract structural patterns independently of surface tokens.

4. **Core heads reduce representational complexity** — Ablating synergistic heads increases the intrinsic dimensionality of hidden states, suggesting they enforce structured, low-dimensional representations.

5. **Network structure differs fundamentally** — Synergy networks are modular (specialized sub-circuits), while redundancy networks have high global efficiency (distributed, robust processing).

6. **Core influences downstream processing** — Layer intervention experiments confirm that synergistic core layers have disproportionate impact on all downstream layers.

---

## Citation

This project reimplements and extends the following paper:

```bibtex
@article{urbina2025synergistic,
  title={A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning},
  author={Urbina-Rodriguez, Adrian and Quax, Rick},
  journal={arXiv preprint arXiv:2505.17552},
  year={2025}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
