# TRACER: Early Failure Detection for Task-Oriented Dialogue
---

## Repository Structure

```
TRACER/
├── tracer/                     # Core Python package
│   ├── data/                   # Dataset loaders and unified format
│   │   ├── unified.py          # Unified dialogue schema + PyTorch datasets
│   │   ├── mwoz_loader.py      # MultiWOZ 2.4 loader and success labeling
│   │   ├── sgd_loader.py       # Schema-Guided Dialogue loader
│   │   ├── abcd_loader.py      # ABCD loader
│   │   └── domain_slots.py     # Slot definitions per domain
│   ├── features/               # Trajectory feature computation
│   │   ├── trajectory.py       # 5-dim feature extraction (MultiWOZ/SGD)
│   │   ├── conflict_detector.py# Semantic conflict detector (sentence-transformers)
│   │   └── abcd_features.py    # ABCD-adapted proxy features
│   ├── models/                 # Neural architectures
│   │   ├── dual_stream.py      # TRACERPredictor, ablation variants, EarlyPredictionLoss
│   │   ├── temporal_transformer.py  # Stream A: temporal transformer
│   │   ├── belief_encoder.py   # Stream B: RoBERTa belief-state encoder
│   │   └── baselines.py        # Heuristic baselines (B1–B4)
│   ├── evaluation/             # Metrics and error analysis
│   │   ├── metrics.py          # AUC-ROC, F1, EDS, BSS
│   │   └── error_analysis.py   # FP/FN analysis, case study selection
│   ├── recovery/               # Recovery utterance generation
│   │   ├── threshold.py        # Pareto-optimal threshold search
│   │   ├── template_recovery.py# Rule-based template recovery
│   │   ├── t5_recovery.py      # Fine-tuned T5 recovery generator
│   │   └── llm_recovery.py     # LLM-prompted recovery (Llama-3.1-8B)
│   └── utils/
│       └── training_utils.py   # Seeding, scheduling, checkpointing
├── scripts/                    # Runnable experiment scripts
│   ├── preprocess_all.py       # Step 1: Preprocess datasets to unified format
│   ├── compute_features.py     # Step 2: Compute trajectory features
│   ├── train_predictor.py      # Step 3: Train full TRACER model
│   ├── train_features_only.py  # Ablation: Stream A only
│   ├── train_text_only.py      # Ablation: Stream B only
│   ├── train_t5_recovery.py    # Train T5 recovery generator
│   ├── evaluate_full.py        # Full evaluation pipeline
│   ├── empirical_analysis.py   # Feature analysis and paper tables
│   ├── prefix_length_analysis.py      # Fixed-prefix evaluation (25/50/75/100%)
│   ├── threshold_tradeoff_study.py    # Threshold sweep study
│   ├── export_paper_artifacts.py      # Export CSVs for paper tables
│   ├── generate_prompted_belief_states.py  # LLM-generated belief states
│   ├── evaluate_prompting_baseline.py      # Zero-shot LLM baseline
│   ├── evaluate_fewshot_prompting_baseline.py  # Few-shot LLM baseline
│   └── llm_assisted_taxonomy_annotation.py     # LLM taxonomy annotation
├── configs/
│   └── default.yaml            # Central configuration file
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/TRACER.git
cd TRACER
pip install -r requirements.txt
```

PyTorch must be installed separately to match your CUDA version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Data Setup

Download the datasets and place them as follows (update `configs/default.yaml` with your actual paths):

| Dataset | Source | Config key |
|---------|--------|-----------|
| MultiWOZ 2.4 | [MultiWOZ 2.4](https://github.com/smartyfh/MultiWOZ2.4) | `data.mwoz_processed_dir`, `data.mwoz_raw_path` |
| Schema-Guided Dialogue | [SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | `data.sgd_dir` |
| ABCD | [ABCD](https://github.com/asappresearch/abcd) | `data.abcd_data_path`, `data.abcd_kb_path` |

Edit `configs/default.yaml` to point to your local data paths and output directories before running any scripts.

---

## Main Experiment

### Step 1 — Preprocess Datasets

Convert all datasets to the unified JSON format and save to `cache/`:

```bash
python scripts/preprocess_all.py --config configs/default.yaml \
    --datasets mwoz sgd abcd \
    --splits train dev test
```

To process only one dataset:
```bash
python scripts/preprocess_all.py --datasets mwoz --splits train dev test
```

### Step 2 — Compute Trajectory Features

Extract the 5-dimensional trajectory features (OscScore, CoverageRate, ConflictCount, FillVelocity, DomainShifts) for every turn of every dialogue:

```bash
python scripts/compute_features.py --config configs/default.yaml \
    --splits train dev test \
    --use_conflict
```

Use `--use_conflict` to enable the sentence-transformer conflict detector (requires `sentence-transformers`). Remove it for faster feature computation without semantic conflict detection.

### Step 3 — Train TRACER (Full Dual-Stream Model)

```bash
python scripts/train_predictor.py --config configs/default.yaml \
    --dataset_filter mwoz
```

- `--dataset_filter`: `mwoz` (default), `sgd`, `abcd`, or `all`
- `--resume <path>`: resume from a checkpoint

This saves the best checkpoint to `outputs/best_predictor.pt` and logs training progress to `outputs/logs/`.

### Step 4 — Run Full Evaluation

```bash
python scripts/evaluate_full.py --config configs/default.yaml \
    --checkpoint outputs/best_predictor.pt \
    --dataset mwoz
```

Key flags:
- `--skip_recovery`: skip T5/LLM recovery evaluation
- `--skip_cross_domain`: skip cross-domain zero-shot evaluation
- `--skip_ablations`: skip neural ablation comparisons
- `--output_dir <path>`: override output directory

Results are written to `outputs/evaluate_full_results.json`.

---

## Ablation Studies

### Ablation A1 — Stream A Only (Features Only)

Train the temporal transformer with only the 5 trajectory features (no RoBERTa encoder):

```bash
python scripts/train_features_only.py --config configs/default.yaml \
    --dataset_filter mwoz
```

Saves checkpoint to `outputs/features_only.pt`.

### Ablation A2 — Stream B Only (Text Only)

Train RoBERTa over belief-state text with no trajectory features:

```bash
python scripts/train_text_only.py --config configs/default.yaml \
    --dataset_filter mwoz
```

Saves checkpoint to `outputs/text_only.pt`.

Both ablations are automatically evaluated in `evaluate_full.py` using their respective checkpoints from the config.

---

## Recovery Module

### Train T5 Recovery Generator

Fine-tune T5-small on confirmation utterances extracted from MultiWOZ:

```bash
python scripts/train_t5_recovery.py --config configs/default.yaml \
    --splits train dev
```

Saves fine-tuned model to `outputs/t5_recovery/`.

The full evaluation pipeline (`evaluate_full.py`) will automatically use:
- **Template recovery** — rule-based utterances based on failure type
- **T5 recovery** — if `outputs/t5_recovery/` exists
- **LLM recovery** — if `--llm_model` is set (Llama-3.1-8B-Instruct by default)

---

## Additional Analyses

### Prefix-Length Analysis

Evaluate all models at fixed dialogue prefixes (25%, 50%, 75%, 100%):

```bash
python scripts/prefix_length_analysis.py --config configs/default.yaml \
    --checkpoint outputs/best_predictor.pt \
    --dataset mwoz
```

Produces `plots/prefix_length_analysis.pdf` and appends results to `outputs/evaluate_full_results.json`.

### Threshold Trade-off Study

Sweep the decision threshold from 0.1 to 0.9 and plot precision/recall/EDS/FPR curves:

```bash
python scripts/threshold_tradeoff_study.py --config configs/default.yaml \
    --checkpoint outputs/best_predictor.pt
```

Produces `outputs/paper_threshold_tradeoff_summary.csv` and a plot in `plots/`.

### Empirical Feature Analysis

Compute Mann-Whitney statistics, failure taxonomy, and feature distribution plots:

```bash
python scripts/empirical_analysis.py --config configs/default.yaml
```

Produces `plots/feature_evolution.pdf`, `plots/feature_distributions.pdf`, and paper table CSVs.

### LLM-Based Belief State Generation

Use vLLM to generate LLM-predicted belief states (for downstream evaluation):

```bash
python scripts/generate_prompted_belief_states.py --config configs/default.yaml \
    --split test --model meta-llama/Llama-3.1-8B-Instruct \
    --output_suffix llm_bs
```

### LLM Prompting Baselines

Zero-shot LLM failure prediction baseline:

```bash
python scripts/evaluate_prompting_baseline.py --config configs/default.yaml \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split test
```

Few-shot variant:

```bash
python scripts/evaluate_fewshot_prompting_baseline.py --config configs/default.yaml \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split test
```

### Export Paper Artifacts

Export CSVs and framing notes for paper tables:

```bash
python scripts/export_paper_artifacts.py
```

---

## Configuration Reference

All paths and hyperparameters are in `configs/default.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `data` | `mwoz_processed_dir` | — | Path to MultiWOZ 2.4 processed directory |
| `data` | `sgd_dir` | — | Path to SGD root directory |
| `data` | `abcd_data_path` | — | Path to `abcd_v1.1.json` |
| `data` | `cache_dir` | `cache` | Where to write unified and feature JSON caches |
| `model.stream_a` | `d_model` | 64 | Transformer hidden size |
| `model.stream_a` | `num_layers` | 3 | Number of transformer layers |
| `model.stream_b` | `encoder_name` | `roberta-base` | HuggingFace encoder name |
| `model.stream_b` | `freeze_layers` | 8 | Number of frozen RoBERTa layers |
| `training` | `lr` | 1e-4 | Learning rate (Stream A + fusion) |
| `training` | `encoder_lr_factor` | 0.1 | LR multiplier for Stream B |
| `training` | `early_pred_lambda` | 0.3 | Weight for early prediction reward |
| `training` | `epochs` | 30 | Max training epochs |
| `recovery` | `max_fpr` | 0.15 | Max FPR constraint for Pareto threshold |
| `features` | `conflict_threshold` | 0.3 | Cosine similarity threshold for conflict detection |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Standard area under the ROC curve over all prefix-level predictions |
| **F1** | F1 score at the Pareto-optimal threshold |
| **EDS** | Early Detection Score — average fraction of dialogue remaining when failure is first detected |
| **FPR** | False positive rate on successful dialogues |
| **BSS** | Belief State Stability Score — measures post-recovery state stabilization |