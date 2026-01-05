# HuggingFace Accelerate & Trainer FSDP Training Pipeline: Complete Guide

This document provides a comprehensive explanation of the HuggingFace Accelerate and Trainer framework for fine-tuning Large Language Models (LLMs) with FSDP. It's designed for practitioners who want to understand the complete training pipeline and adapt it for Amazon SageMaker AI.

---

## Table of Contents

1. [Overview: What This Framework Does](#overview-what-this-framework-does)
2. [Key Concepts Explained](#key-concepts-explained)
3. [The Training Pipeline: 7 Phases](#the-training-pipeline-7-phases)
4. [Phase 1: Accelerate Configuration](#phase-1-accelerate-configuration)
5. [Phase 2: Training Configuration](#phase-2-training-configuration)
6. [Phase 3: Data Preparation](#phase-3-data-preparation)
7. [Phase 4: Model Initialization](#phase-4-model-initialization)
8. [Phase 5: Trainer Setup](#phase-5-trainer-setup)
9. [Phase 6: Training Loop](#phase-6-training-loop)
10. [Phase 7: Post-Training (Conversion & Testing)](#phase-7-post-training-conversion--testing)
11. [Comparison: Accelerate vs PyTorch Lightning](#comparison-accelerate-vs-pytorch-lightning)
12. [SageMaker Adaptation Guide](#sagemaker-adaptation-guide)
13. [Quick Reference: File Responsibilities](#quick-reference-file-responsibilities)

---

## Overview: What This Framework Does

This framework fine-tunes pre-trained language models using the HuggingFace ecosystem:

| Technology | Purpose |
|------------|---------|
| **HuggingFace Accelerate** | Handles distributed training setup (FSDP, DDP, etc.) |
| **HuggingFace Trainer** | High-level training API with callbacks, logging, checkpointing |
| **FSDP** (Fully Sharded Data Parallel) | Distributes large models across multiple GPUs |
| **PEFT/LoRA** | Parameter-efficient fine-tuning |
| **Flash Attention 2** | Optimized attention computation |

**Key Difference from PyTorch Lightning**: This framework uses HuggingFace's `Trainer` class, which provides more built-in integrations with the HuggingFace ecosystem (model hub, datasets, tokenizers) but less flexibility than PyTorch Lightning's modular approach.

---

## Key Concepts Explained

### HuggingFace Accelerate vs Trainer

These are complementary tools:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HuggingFace Ecosystem                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ACCELERATE                                    │   │
│  │  • Distributed training orchestration                               │   │
│  │  • FSDP/DDP/DeepSpeed configuration                                 │   │
│  │  • Multi-GPU/Multi-node setup                                       │   │
│  │  • Mixed precision management                                       │   │
│  │  • Launch command: accelerate launch --config_file config.yaml      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          TRAINER                                     │   │
│  │  • High-level training loop                                         │   │
│  │  • Automatic gradient accumulation                                  │   │
│  │  • Built-in callbacks (early stopping, checkpointing)               │   │
│  │  • Logging to TensorBoard/Wandb                                     │   │
│  │  • Evaluation during training                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How FSDP Works with Accelerate

Accelerate abstracts FSDP configuration into a simple YAML file:

```yaml
# accelerate_config.yaml
distributed_type: FSDP           # Enable FSDP
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD    # Maximum memory savings
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP  # Auto-detect layers
```

When you run `accelerate launch`, it:
1. Reads the config file
2. Spawns processes on each GPU
3. Wraps your model with FSDP
4. Handles all distributed communication

### Trainer's Training Loop

The Trainer handles the entire training loop automatically:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[...],
)

trainer.train()  # This single call handles everything:
                 # - Data loading
                 # - Forward/backward passes
                 # - Gradient accumulation
                 # - Optimizer steps
                 # - Logging
                 # - Checkpointing
                 # - Evaluation
```

---

## The Training Pipeline: 7 Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE OVERVIEW                          │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: ACCELERATE CONFIGURATION
    accelerate_config.yaml → FSDP settings, GPU count, mixed precision
                            │
                            ▼
Phase 2: TRAINING CONFIGURATION
    CLI Arguments → TrainingConfig dataclass → Auto-generated paths
                            │
                            ▼
Phase 3: DATA PREPARATION (Rank 0 only)
    HuggingFace Dataset → Split (train/val/test) → Convert to JSONL
                            │
                            ▼
Phase 4: MODEL INITIALIZATION
    Load Base Model → Flash Attention 2 → Gradient Checkpointing → Apply LoRA
                            │
                            ▼
Phase 5: TRAINER SETUP
    TrainingArguments → DataCollator → Callbacks → Trainer instance
                            │
                            ▼
Phase 6: TRAINING LOOP (Handled by Trainer)
    ┌──────────────────────────────────────────────────────────────┐
    │  Trainer.train() handles everything:                         │
    │  • Data iteration with distributed sampler                   │
    │  • Forward pass → Loss → Backward pass                       │
    │  • Gradient accumulation & clipping                          │
    │  • Optimizer step with scheduler                             │
    │  • Evaluation at specified intervals                         │
    │  • Checkpoint saving via callbacks                           │
    │  • Early stopping if no improvement                          │
    └──────────────────────────────────────────────────────────────┘
                            │
                            ▼
Phase 7: POST-TRAINING
    Convert FSDP Checkpoint → HuggingFace Format → Test with vLLM
```

---

## Phase 1: Accelerate Configuration

**File**: `accelerate_config.yaml`

This YAML file configures how Accelerate sets up distributed training. It's read by the `accelerate launch` command.

### Complete Configuration Explained

```yaml
# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
compute_environment: LOCAL_MACHINE    # Running on local hardware (not cloud)
debug: false                          # Disable debug mode for performance

# ═══════════════════════════════════════════════════════════════════════════
# DISTRIBUTED TRAINING TYPE
# ═══════════════════════════════════════════════════════════════════════════
distributed_type: FSDP                # Use Fully Sharded Data Parallel
                                      # Options: NO, MULTI_GPU, FSDP, DEEPSPEED, TPU

# ═══════════════════════════════════════════════════════════════════════════
# PRECISION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
mixed_precision: bf16                 # BFloat16 mixed precision
                                      # Options: no, fp16, bf16

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-NODE/MULTI-GPU SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
num_machines: 1                       # Number of nodes
num_processes: 8                      # Total GPUs (8 per node × 1 node)
machine_rank: 0                       # This machine's rank (0 = main node)
main_process_ip: localhost            # IP of main node
main_process_port: 29500              # Port for distributed communication

# ═══════════════════════════════════════════════════════════════════════════
# FSDP-SPECIFIC CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
fsdp_config:
  # How to wrap model layers
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  # Options:
  #   - TRANSFORMER_BASED_WRAP: Auto-detect transformer layers (recommended)
  #   - SIZE_BASED_WRAP: Wrap based on parameter count
  #   - NO_WRAP: Don't auto-wrap (manual control)

  # Sharding strategy
  fsdp_sharding_strategy: FULL_SHARD
  # Options:
  #   - FULL_SHARD: Shard params, gradients, optimizer states (max memory savings)
  #   - SHARD_GRAD_OP: Only shard gradients and optimizer
  #   - NO_SHARD: No sharding (like DDP)

  # Communication optimizations
  fsdp_backward_prefetch: BACKWARD_PRE    # Prefetch during backward pass
  fsdp_forward_prefetch: false            # Don't prefetch during forward

  # Memory optimizations
  fsdp_cpu_ram_efficient_loading: true    # Stream model loading
  fsdp_offload_params: false              # Don't offload to CPU (slower)

  # Checkpoint settings
  fsdp_state_dict_type: SHARDED_STATE_DICT  # Save sharded checkpoints
  # Options:
  #   - FULL_STATE_DICT: Gather full model (requires more memory)
  #   - SHARDED_STATE_DICT: Save shards separately (recommended)
  #   - LOCAL_STATE_DICT: Local shards only

  # Synchronization
  fsdp_sync_module_states: true           # Broadcast weights from rank 0
  fsdp_use_orig_params: true              # Use original param names (for PEFT)

  # Activation checkpointing (handled in code instead)
  fsdp_activation_checkpointing: false
```

### Multi-Node Configuration

For training across multiple machines, modify:

```yaml
# Main node (machine 0)
num_machines: 4
num_processes: 32              # 8 GPUs × 4 nodes
machine_rank: 0
main_process_ip: 10.0.0.1      # IP of this machine
main_process_port: 29500

# Worker nodes (machines 1, 2, 3)
# Same config but with:
machine_rank: 1                # or 2, 3
main_process_ip: 10.0.0.1      # IP of main node (not this machine)
```

### Launch Command

```bash
# Single node (uses config file)
accelerate launch --config_file accelerate_config.yaml peft_accelerate.py

# Override specific settings
accelerate launch \
  --config_file accelerate_config.yaml \
  --num_processes 4 \
  peft_accelerate.py --hf_model_id "meta-llama/Llama-3-8B"
```

---

## Phase 2: Training Configuration

**File**: `peft_accelerate.py` (lines 50-154)

### TrainingConfig Dataclass

All training parameters are defined in a dataclass for type safety and auto-CLI generation:

```python
@dataclass
class TrainingConfig:
    # ═══════════════════════════════════════════════════════════════════════
    # MODEL SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    hf_model_id: str = "Qwen/Qwen3-8B"    # HuggingFace model identifier
    trust_remote_code: bool = True         # Allow custom model code

    # ═══════════════════════════════════════════════════════════════════════
    # LORA SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    full_ft: bool = False                  # False = LoRA, True = full fine-tune
    lora_rank: int = 32                    # LoRA rank (r parameter)
    lora_alpha: int = 32                   # Scaling factor
    lora_dropout: float = 0.1              # Dropout in LoRA layers
    lora_target_modules: List[str] = [     # Layers to apply LoRA
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention
        'gate_proj', 'up_proj', 'down_proj'       # Feed-forward
    ]

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    max_steps: int = 10000                        # Total training steps
    per_device_train_batch_size: int = 2          # Batch size per GPU
    per_device_eval_batch_size: int = 2           # Eval batch size per GPU
    gradient_accumulation_steps: int = 4          # Accumulate gradients
    learning_rate: float = 5e-5                   # Peak learning rate
    weight_decay: float = 0.01                    # L2 regularization
    warmup_steps: int = 100                       # LR warmup steps
    max_grad_norm: float = 1.0                    # Gradient clipping

    # ═══════════════════════════════════════════════════════════════════════
    # EARLY STOPPING
    # ═══════════════════════════════════════════════════════════════════════
    early_stopping_patience: int = 3              # Evaluations without improvement
    early_stopping_threshold: float = 0.001       # Minimum improvement

    # ═══════════════════════════════════════════════════════════════════════
    # SEQUENCE SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    max_seq_length: int = 2048                    # Maximum sequence length

    # ═══════════════════════════════════════════════════════════════════════
    # LOGGING & EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    logging_steps: int = 10                       # Log every N steps
    eval_steps: int = 100                         # Evaluate every N steps
    max_eval_samples: int = 640                   # Limit eval samples
    use_wandb: bool = False                       # Enable Weights & Biases
```

### Global Batch Size Calculation

```
Global Batch Size = per_device_batch × num_gpus × gradient_accumulation

Example:
  per_device_train_batch_size = 2
  num_processes = 8 (from accelerate config)
  gradient_accumulation_steps = 4

  Global Batch Size = 2 × 8 × 4 = 64 samples per optimizer step
```

### Auto-Generated Paths

The `__post_init__` method creates self-documenting directory structures:

```python
def __post_init__(self):
    if self.data_dir is None:
        # Example: datasets/cognitivecomputations_dolphin/flan1m-alpaca-uncensored/train=90%-val=5%-test=5%
        dataset_name = self.hf_dataset_config.dataset_name.replace('/', '_')
        dataset_config = self.hf_dataset_config.dataset_config or 'default'
        train_pct = int(self.hf_dataset_config.train_split_ratio * 100)
        # ... calculate val_pct, test_pct
        self.data_dir = f"datasets/{dataset_name}/{dataset_config}/train={train_pct}%-val={val_pct}%-test={test_pct}%"

    if self.output_dir is None:
        # Example: results/Qwen/Qwen3-8B
        self.output_dir = f"results/{self.hf_model_id}"
```

---

## Phase 3: Data Preparation

**File**: `dataset_module.py`

### HFDatasetConfig

Configuration for loading any HuggingFace dataset:

```python
@dataclass
class HFDatasetConfig:
    dataset_name: str                          # e.g., "cognitivecomputations/dolphin"
    dataset_config: Optional[str] = None       # Subset/configuration name
    split: str = "train"                       # Which split to load

    # Splitting ratios
    train_split_ratio: float = 0.9             # 90% for training
    val_test_split_ratio: float = 0.5          # Split remaining 50/50

    # Template formatting
    input_template: str = "### Instruction:\n{instruction}\n### Input:\n{input}\n"
    output_template: str = "### Response:\n{output}"
    field_mapping: Optional[Dict[str, str]] = None  # Map placeholders to columns

    num_proc: int = 8                          # Parallel workers
    custom_converter: Optional[Callable] = None  # Custom conversion function
```

### Data Preparation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

HuggingFace Hub: "cognitivecomputations/dolphin"
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  load_dataset()                                                             │
│  ├─ Downloads from HuggingFace Hub                                          │
│  ├─ Caches locally (~/.cache/huggingface)                                   │
│  └─ Returns Dataset object                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  train_test_split() - First Split                                           │
│  ├─ train_split_ratio = 0.9                                                 │
│  └─ Result: train (90%), remaining (10%)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  train_test_split() - Second Split                                          │
│  ├─ val_test_split_ratio = 0.5                                              │
│  └─ Result: val (5%), test (5%)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Template Conversion                                                        │
│                                                                             │
│  Original:                                                                  │
│  {                                                                          │
│    "instruction": "Explain gravity",                                        │
│    "input": "",                                                             │
│    "output": "Gravity is a fundamental force..."                            │
│  }                                                                          │
│                           │                                                 │
│                           ▼                                                 │
│  Converted:                                                                 │
│  {                                                                          │
│    "input": "### Instruction:\nExplain gravity\n### Input:\n\n",            │
│    "output": "### Response:\nGravity is a fundamental force..."             │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Save as JSONL Files                                                        │
│  ├─ training.jsonl    (900,000 samples)                                     │
│  ├─ validation.jsonl  (50,000 samples)                                      │
│  ├─ test.jsonl        (50,000 samples)                                      │
│  └─ .data_ready       (marker file)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Distributed Synchronization

Only rank 0 prepares data; other ranks wait:

```python
global_rank = int(os.environ.get("RANK", 0))
marker_file = Path(config.data_dir) / ".data_ready"

if global_rank == 0:
    prepare_datasets(config.hf_dataset_config, config.data_dir)
    marker_file.touch()
else:
    while not marker_file.exists():
        print(f"Rank {global_rank} waiting for data preparation...")
        time.sleep(10)
```

### SFTDataset: Tokenization with Label Masking

The key insight: **Only output tokens contribute to the loss**.

```python
class SFTDataset(Dataset):
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Step 1: Tokenize full text (input + output)
        full_text = sample['input'] + sample['output']
        tokenized = self.tokenizer(full_text, truncation=True, max_length=2048)

        input_ids = tokenized['input_ids']
        labels = list(input_ids)  # Copy for modification

        # Step 2: Tokenize input separately to find its length
        input_tokenized = self.tokenizer(sample['input'])
        input_length = len(input_tokenized['input_ids'])

        # Step 3: Mask input portion in labels
        for i in range(input_length):
            labels[i] = -100  # -100 is ignored by CrossEntropyLoss

        return {
            'input_ids': input_ids,
            'labels': labels,           # Input masked, output preserved
            'attention_mask': attention_mask
        }
```

**Visual Representation**:

```
full_text: "### Instruction:\nExplain gravity\n### Response:\nGravity is..."
           ├────────── Input ──────────┤├────────── Output ──────────┤

input_ids: [1, 1234, 5678, 91011, 1213, | 42, 43, 44, 45, 46, 47, 48]
                                       │
labels:    [-100, -100, -100, -100, -100,| 42, 43, 44, 45, 46, 47, 48]
           ├──── Ignored in loss ─────┤├─── Used for loss ─────────┤
```

---

## Phase 4: Model Initialization

**File**: `peft_accelerate.py` (lines 219-247)

### Loading the Base Model

```python
model = AutoModelForCausalLM.from_pretrained(
    config.hf_model_id,
    trust_remote_code=True,           # Allow custom model architectures
    dtype=torch.bfloat16,             # Use BF16 for memory efficiency
    attn_implementation="flash_attention_2",  # Fast attention
    use_cache=False,                  # Disable KV cache (not needed for training)
)

# Enable gradient checkpointing for memory savings
model.gradient_checkpointing_enable()
```

### Memory Optimizations Explained

| Optimization | Effect | Trade-off |
|--------------|--------|-----------|
| `dtype=torch.bfloat16` | 50% memory reduction | Slight precision loss (negligible) |
| `flash_attention_2` | 2-4x faster attention, O(n) memory | Requires compatible GPU |
| `use_cache=False` | Saves KV cache memory | N/A for training |
| `gradient_checkpointing_enable()` | ~30% memory reduction | ~30% slower |

### Applying LoRA

```python
if not config.full_ft:
    # Prepare model for efficient training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        r=config.lora_rank,              # Rank of adaptation matrices
        lora_alpha=config.lora_alpha,    # Scaling factor
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",                     # Don't train biases
        task_type="CAUSAL_LM",
    )

    # Wrap model with LoRA adapters
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # Output: trainable params: 41,943,040 || all params: 8,030,000,000 || trainable%: 0.52%
```

### LoRA Target Modules

The default targets all major weight matrices:

```
Attention Layers:          Feed-Forward Layers:
├─ q_proj (Query)          ├─ gate_proj (Gate)
├─ k_proj (Key)            ├─ up_proj (Up projection)
├─ v_proj (Value)          └─ down_proj (Down projection)
└─ o_proj (Output)
```

---

## Phase 5: Trainer Setup

**File**: `peft_accelerate.py` (lines 249-298)

### TrainingArguments

HuggingFace's `TrainingArguments` encapsulates all training configuration:

```python
training_args = TrainingArguments(
    # ═══════════════════════════════════════════════════════════════════════
    # OUTPUT & CHECKPOINTING
    # ═══════════════════════════════════════════════════════════════════════
    output_dir=config.output_dir,         # Where to save checkpoints
    save_strategy="no",                   # Disable auto-save (use callback)
    save_total_limit=2,                   # Keep only 2 checkpoints

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING DURATION
    # ═══════════════════════════════════════════════════════════════════════
    max_steps=config.max_steps,           # Total training steps

    # ═══════════════════════════════════════════════════════════════════════
    # BATCH SIZE & ACCUMULATION
    # ═══════════════════════════════════════════════════════════════════════
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,

    # ═══════════════════════════════════════════════════════════════════════
    # OPTIMIZER SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    learning_rate=config.learning_rate,   # Peak learning rate
    weight_decay=config.weight_decay,     # L2 regularization
    warmup_steps=config.warmup_steps,     # LR warmup period
    max_grad_norm=config.max_grad_norm,   # Gradient clipping

    # ═══════════════════════════════════════════════════════════════════════
    # PRECISION
    # ═══════════════════════════════════════════════════════════════════════
    bf16=True,                            # Use BFloat16

    # ═══════════════════════════════════════════════════════════════════════
    # LOGGING & EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    logging_steps=config.logging_steps,
    logging_dir=f"{config.output_dir}/logs",
    report_to=["tensorboard"],            # or ["wandb", "tensorboard"]

    eval_strategy="steps",
    eval_steps=config.eval_steps,
    metric_for_best_model="eval_loss",
    greater_is_better=False,              # Lower loss = better

    # ═══════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════
    dataloader_num_workers=config.num_workers,
    remove_unused_columns=False,          # Keep all columns
    dataloader_drop_last=False,           # Use all data

    # ═══════════════════════════════════════════════════════════════════════
    # REPRODUCIBILITY
    # ═══════════════════════════════════════════════════════════════════════
    seed=config.seed,
)
```

### DataCollatorForSeq2Seq

The data collator handles batching with proper padding:

```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,                     # Pad to longest in batch
    pad_to_multiple_of=8,             # Align for tensor core efficiency
    label_pad_token_id=-100           # Padding in labels is ignored
)
```

**What it does**:

```
Input batch (variable lengths):
  Sample 1: [1, 2, 3, 4, 5]           (length 5)
  Sample 2: [1, 2, 3, 4, 5, 6, 7, 8]  (length 8)
  Sample 3: [1, 2, 3]                 (length 3)

After collation (padded to multiple of 8):
  Sample 1: [1, 2, 3, 4, 5, PAD, PAD, PAD]
  Sample 2: [1, 2, 3, 4, 5, 6, 7, 8]
  Sample 3: [1, 2, 3, PAD, PAD, PAD, PAD, PAD]

Labels (padding masked):
  Sample 1: [-100, -100, ..., 4, 5, -100, -100, -100]
  Sample 2: [-100, -100, ..., 6, 7, 8]
  Sample 3: [-100, -100, 3, -100, -100, -100, -100, -100]
```

### Callbacks

Two callbacks control training flow:

```python
callbacks = [
    SaveOnBestMetricCallback(),       # Save only when metric improves
    EarlyStoppingCallback(
        early_stopping_patience=3,    # Stop after 3 evals without improvement
        early_stopping_threshold=0.001  # Minimum improvement required
    ),
]
```

**SaveOnBestMetricCallback** (custom):

```python
class SaveOnBestMetricCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_value = metrics.get(args.metric_for_best_model)

        if state.best_metric is None:
            control.should_save = True
        elif not args.greater_is_better:
            if metric_value < state.best_metric:
                control.should_save = True

        return control
```

---

## Phase 6: Training Loop

**File**: `peft_accelerate.py` (lines 300-311)

### The Training Call

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=callbacks,
)

trainer.train()  # Everything happens here
```

### What Trainer.train() Does Internally

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINER.TRAIN() INTERNALS                           │
└─────────────────────────────────────────────────────────────────────────────┘

For each training step:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING                                                             │
│    ├─ DistributedSampler splits data across GPUs                            │
│    ├─ DataLoader yields micro-batches                                       │
│    └─ DataCollator pads and stacks                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. FORWARD PASS                                                             │
│    ├─ Model computes logits                                                 │
│    ├─ CrossEntropyLoss computed (ignoring -100 labels)                      │
│    └─ Loss returned                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. BACKWARD PASS                                                            │
│    ├─ loss.backward() computes gradients                                    │
│    ├─ FSDP handles gradient synchronization across GPUs                     │
│    └─ Gradients accumulated if gradient_accumulation_steps > 1              │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. OPTIMIZER STEP (every gradient_accumulation_steps)                       │
│    ├─ Gradient clipping (max_grad_norm=1.0)                                 │
│    ├─ AdamW optimizer step                                                  │
│    ├─ Learning rate scheduler step                                          │
│    └─ Zero gradients                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. LOGGING (every logging_steps)                                            │
│    ├─ Log loss to TensorBoard/Wandb                                         │
│    ├─ Log learning rate                                                     │
│    └─ Log throughput metrics                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. EVALUATION (every eval_steps)                                            │
│    ├─ Switch to eval mode                                                   │
│    ├─ Run eval_dataset through model                                        │
│    ├─ Compute eval_loss                                                     │
│    ├─ Call SaveOnBestMetricCallback                                         │
│    │   └─ Save checkpoint if improved                                       │
│    ├─ Call EarlyStoppingCallback                                            │
│    │   └─ Stop training if no improvement for patience evals                │
│    └─ Switch back to train mode                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Learning Rate Schedule

By default, Trainer uses a linear decay schedule:

```
Learning Rate
    ↑
5e-5│    ╱────────────────────────────────╲
    │   ╱                                  ╲
    │  ╱                                    ╲
    │ ╱                                      ╲
    │╱                                        ╲
  0 └────┬────────────────────────────────────┬───→ Steps
         100                                10000
         ↑                                    ↑
      Warmup                            Final step
```

---

## Phase 7: Post-Training (Conversion & Testing)

### Checkpoint Format: FSDP Sharded State Dict

Accelerate with FSDP saves checkpoints in a sharded format:

```
results/Qwen/Qwen3-8B/
├── checkpoint-1000/
│   └── pytorch_model_fsdp_0/
│       ├── .0_0.pt                   # Shard 0 (GPU 0's portion)
│       ├── .0_1.pt                   # Shard 1 (GPU 1's portion)
│       ├── .0_2.pt                   # Shard 2
│       ├── .0_3.pt                   # Shard 3
│       ├── .0_4.pt                   # Shard 4
│       ├── .0_5.pt                   # Shard 5
│       ├── .0_6.pt                   # Shard 6
│       ├── .0_7.pt                   # Shard 7
│       └── __metadata__              # Metadata for reconstruction
├── final/                            # Final merged model
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer files
└── logs/                             # TensorBoard logs
```

### Checkpoint Conversion

**File**: `convert_checkpoint_to_hf.py`

Converts FSDP sharded checkpoints to standard HuggingFace format:

```python
import torch.distributed.checkpoint as dcp

# Load sharded checkpoint
fsdp_checkpoint_dir = Path(checkpoint_path) / "pytorch_model_fsdp_0"
state_dict = {}
dcp.load(state_dict, checkpoint_id=str(fsdp_checkpoint_dir))

# Load into model
model.load_state_dict(state_dict, strict=False)

# Merge LoRA weights (if applicable)
if not config.full_ft:
    model = model.merge_and_unload()

# Save as standard HuggingFace model
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
```

**Usage**:

```bash
# Merge LoRA and save (recommended for deployment)
python convert_checkpoint_to_hf.py --base_model "Qwen/Qwen3-8B"
# Output: checkpoint-1000.hf_model/

# Save as LoRA adapter (smaller, requires PEFT to load)
python convert_checkpoint_to_hf.py --base_model "Qwen/Qwen3-8B" --no_merge
# Output: checkpoint-1000.hf_peft/
```

### Testing with vLLM

**File**: `test_checkpoint.py`

High-performance inference using vLLM:

```python
from vllm import LLM, SamplingParams

# Load merged model into vLLM
llm = LLM(
    model=temp_model_path,
    tensor_parallel_size=8,           # Distribute across 8 GPUs
    gpu_memory_utilization=0.9,       # Use 90% of GPU memory
    max_model_len=8192,               # Max context length
    dtype="bfloat16",
)

# Configure generation
sampling_params = SamplingParams(
    temperature=0.1,                  # Low for deterministic outputs
    top_p=0.95,                       # Nucleus sampling
    max_tokens=512,                   # Max output length
)

# Batch generation
outputs = llm.generate(prompts, sampling_params)
```

**Evaluation**:

```python
import evaluate

bertscore = evaluate.load('bertscore')
scores = bertscore.compute(
    predictions=predictions,
    references=references,
    lang='en'
)
print(f"BERTScore F1: {scores['f1'].mean():.4f}")
```

---

## Comparison: Accelerate vs PyTorch Lightning

| Aspect | HuggingFace Accelerate/Trainer | PyTorch Lightning |
|--------|--------------------------------|-------------------|
| **Configuration** | YAML file + TrainingArguments | Python code (FSDPStrategy) |
| **FSDP Setup** | Automatic via config | Manual auto_wrap_policy |
| **Decoder Layer** | Auto-detected | Must specify (e.g., `Qwen3DecoderLayer`) |
| **Training Loop** | Built-in (Trainer.train()) | Define `training_step()` |
| **Callbacks** | TrainerCallback | LightningCallback |
| **Checkpoint Format** | FSDP sharded or full | PyTorch Lightning .ckpt |
| **Gradient Clipping** | Automatic (max_grad_norm) | Manual or automatic |
| **Learning Rate** | Linear decay default | Must configure scheduler |
| **Flexibility** | Less (higher-level API) | More (lower-level control) |
| **HF Integration** | Native | Requires adaptation |

### When to Use Each

**Use Accelerate/Trainer when**:
- You want simpler configuration
- You're using standard HuggingFace models
- You want automatic handling of distributed training details
- You prefer YAML-based configuration

**Use PyTorch Lightning when**:
- You need custom training logic
- You want more control over the training loop
- You're building a complex training pipeline
- You prefer Python-based configuration

---

## SageMaker Adaptation Guide

### Key Changes Needed

#### 1. Entry Point Script

Create a SageMaker-compatible entry point:

```python
# train.py (SageMaker entry point)
import os
import json

def main():
    # SageMaker provides hyperparameters via environment variable
    sm_hp = os.environ.get('SM_HPS', '{}')
    hyperparameters = json.loads(sm_hp)

    # SageMaker paths
    training_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    # Override config
    config = TrainingConfig(
        data_dir=training_dir,
        output_dir=model_dir,
        **hyperparameters
    )

    train(config)

if __name__ == '__main__':
    main()
```

#### 2. SageMaker Estimator Configuration

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='./accelerate',
    role=role,
    instance_count=1,
    instance_type='ml.p4d.24xlarge',  # 8x A100 GPUs
    framework_version='2.3.0',
    py_version='py311',

    # Enable distributed training
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },

    hyperparameters={
        'hf_model_id': 'Qwen/Qwen3-8B',
        'max_steps': 10000,
        'per_device_train_batch_size': 2,
        'gradient_accumulation_steps': 4,
    },

    environment={
        'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
    },
)

estimator.fit({
    'training': 's3://your-bucket/datasets/dolphin/'
})
```

#### 3. Accelerate Config for SageMaker

Create a SageMaker-specific config:

```yaml
# accelerate_config_sagemaker.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: 8

# SageMaker handles these automatically
main_process_ip: null
main_process_port: null

fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_state_dict_type: FULL_STATE_DICT  # Easier for SageMaker model saving
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
```

#### 4. Multi-Node Training on SageMaker

```python
estimator = PyTorch(
    # ...
    instance_count=4,              # 4 nodes
    instance_type='ml.p4d.24xlarge',  # 8 GPUs each = 32 total

    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },
)
```

Update accelerate config:

```yaml
num_machines: 4
num_processes: 32  # 8 × 4
```

#### 5. Model Saving for SageMaker Endpoints

```python
# At end of training, save in HuggingFace format
if trainer.is_world_process_zero():
    # Convert FSDP checkpoint to HF format
    final_model = trainer.model

    # Merge LoRA if used
    if hasattr(final_model, 'merge_and_unload'):
        final_model = final_model.merge_and_unload()

    # Save to SageMaker model directory
    final_model.save_pretrained(
        os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
        safe_serialization=True
    )
    tokenizer.save_pretrained(os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
```

---

## Quick Reference: File Responsibilities

| File | Lines | Purpose |
|------|-------|---------|
| `accelerate_config.yaml` | 22 | FSDP distributed training configuration |
| `peft_accelerate.py` | 395 | Main training script with Trainer |
| `dataset_module.py` | 191 | Dataset loading, conversion, tokenization |
| `test_checkpoint.py` | 374 | vLLM inference and BERTScore evaluation |
| `convert_checkpoint_to_hf.py` | 213 | FSDP checkpoint to HuggingFace format |

### Key Functions Quick Reference

| Function/Class | File:Line | Purpose |
|----------------|-----------|---------|
| `TrainingConfig` | peft_accelerate.py:50 | All training hyperparameters |
| `SaveOnBestMetricCallback` | peft_accelerate.py:29 | Save only on metric improvement |
| `train()` | peft_accelerate.py:156 | Main training function |
| `HFDatasetConfig` | dataset_module.py:12 | Dataset configuration |
| `SFTDataset` | dataset_module.py:29 | Tokenization with label masking |
| `prepare_datasets()` | dataset_module.py:89 | Download and convert datasets |

### CLI Quick Reference

```bash
# Basic training
accelerate launch --config_file accelerate_config.yaml peft_accelerate.py

# Custom model
accelerate launch --config_file accelerate_config.yaml peft_accelerate.py \
  --hf_model_id "meta-llama/Llama-3-8B"

# Full fine-tuning
accelerate launch --config_file accelerate_config.yaml peft_accelerate.py \
  --full_ft

# Custom dataset
accelerate launch --config_file accelerate_config.yaml peft_accelerate.py \
  --hfdc_dataset_name "databricks/databricks-dolly-15k" \
  --hfdc_input_template "Q: {instruction}\n" \
  --hfdc_output_template "A: {response}"

# Convert checkpoint
python convert_checkpoint_to_hf.py --base_model "Qwen/Qwen3-8B"

# Test checkpoint
python test_checkpoint.py --base_model "Qwen/Qwen3-8B" --max_samples 1024
```

---

## Summary

This framework provides a production-ready pipeline for fine-tuning LLMs using the HuggingFace ecosystem:

1. **Simple Configuration**: YAML-based FSDP setup with Accelerate
2. **High-Level Training**: Trainer handles the entire training loop
3. **Memory Efficiency**: FSDP + LoRA + Gradient Checkpointing + Flash Attention
4. **Flexible Datasets**: Support for any HuggingFace dataset via templates
5. **Built-in Features**: Early stopping, best-metric checkpointing, logging

For SageMaker adaptation:
- Use SageMaker's distributed training distribution
- Map paths to SageMaker conventions (`/opt/ml/...`)
- Consider `FULL_STATE_DICT` for easier model deployment
- Save merged models in HuggingFace format for endpoint inference
