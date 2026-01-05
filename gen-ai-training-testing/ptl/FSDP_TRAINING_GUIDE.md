# PyTorch Lightning FSDP Training Pipeline: Complete Guide

This document provides a comprehensive explanation of the PyTorch Lightning FSDP training framework for fine-tuning Large Language Models (LLMs). It's designed for practitioners who want to understand the complete training pipeline and adapt it for Amazon SageMaker AI.

---

## Table of Contents

1. [Overview: What This Framework Does](#overview-what-this-framework-does)
2. [Key Concepts Explained](#key-concepts-explained)
3. [The Training Pipeline: 7 Phases](#the-training-pipeline-7-phases)
4. [Phase 1: Configuration](#phase-1-configuration)
5. [Phase 2: Data Preparation](#phase-2-data-preparation)
6. [Phase 3: Dataset Setup & Tokenization](#phase-3-dataset-setup--tokenization)
7. [Phase 4: Model Initialization](#phase-4-model-initialization)
8. [Phase 5: FSDP Strategy Configuration](#phase-5-fsdp-strategy-configuration)
9. [Phase 6: Training Loop](#phase-6-training-loop)
10. [Phase 7: Post-Training (Conversion & Testing)](#phase-7-post-training-conversion--testing)
11. [SageMaker Adaptation Guide](#sagemaker-adaptation-guide)
12. [Quick Reference: File Responsibilities](#quick-reference-file-responsibilities)

---

## Overview: What This Framework Does

This framework fine-tunes pre-trained language models (like Qwen, Llama, Mistral) using:

| Technology | Purpose |
|------------|---------|
| **PyTorch Lightning** | Simplifies training code and handles boilerplate |
| **FSDP** (Fully Sharded Data Parallel) | Distributes large models across multiple GPUs |
| **LoRA** (Low-Rank Adaptation) | Trains only a small subset of parameters efficiently |
| **Flash Attention 2** | Speeds up attention computation with less memory |

**The result**: You can fine-tune an 8B parameter model on 8 GPUs that would otherwise require 16+ GPUs with naive approaches.

---

## Key Concepts Explained

### What is FSDP?

**Problem**: A 7B parameter model in FP32 needs ~28GB just to store weights. Add gradients and optimizer states, and you need 100GB+ per GPU.

**Solution**: FSDP "shards" (splits) the model across multiple GPUs:

```
Traditional Data Parallel:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │
│ Full Model  │  │ Full Model  │  │ Full Model  │  │ Full Model  │
│ + Gradients │  │ + Gradients │  │ + Gradients │  │ + Gradients │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
     ❌ Each GPU holds everything - wastes memory

FSDP (Fully Sharded):
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │
│  Shard 1    │  │  Shard 2    │  │  Shard 3    │  │  Shard 4    │
│ (1/4 model) │  │ (1/4 model) │  │ (1/4 model) │  │ (1/4 model) │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
     ✅ Each GPU holds only its portion - 4x memory savings
```

During forward/backward passes, FSDP temporarily gathers the full layer weights when needed, then discards them.

### What is LoRA?

Instead of training all parameters, LoRA adds small "adapter" matrices to specific layers:

```
Original Weight Matrix W: [4096 x 4096] = 16M parameters
                                          ↓
LoRA adapters: A: [4096 x 32] + B: [32 x 4096] = 262K parameters

Savings: 98.4% fewer trainable parameters!
```

The framework applies LoRA to attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feed-forward layers (`gate_proj`, `up_proj`, `down_proj`).

### What is Activation Checkpointing?

During forward pass, intermediate activations are stored for the backward pass. For large models, this uses enormous memory.

**Solution**: Don't store activations. Recompute them during backward pass.

```
Without Checkpointing:
Forward:  Layer1 → [save] → Layer2 → [save] → Layer3 → [save] → Output
Memory:   ████████████████████████████████████████████████████████

With Checkpointing:
Forward:  Layer1 → Layer2 → Layer3 → Output  (activations discarded)
Backward: [recompute Layer1] → [recompute Layer2] → [recompute Layer3]
Memory:   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

Trade-off: ~30% more compute time, but significantly less memory.

---

## The Training Pipeline: 7 Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE OVERVIEW                          │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: CONFIGURATION
    CLI Arguments → Config Dataclass → Auto-generated paths
                            │
                            ▼
Phase 2: DATA PREPARATION (Rank 0 only)
    HuggingFace Dataset → Split (train/val/test) → Convert to JSONL
                            │
                            ▼
Phase 3: DATASET SETUP & TOKENIZATION
    Load Tokenizer → Create SFTDataset instances → DataLoaders ready
                            │
                            ▼
Phase 4: MODEL INITIALIZATION
    Load Base Model → Apply LoRA adapters → Print trainable params
                            │
                            ▼
Phase 5: FSDP STRATEGY CONFIGURATION
    Auto-wrap Policy → Sharding Strategy → Mixed Precision → Activation Ckpt
                            │
                            ▼
Phase 6: TRAINING LOOP
    ┌──────────────────────────────────────────────────────────┐
    │  For each batch:                                          │
    │  1. Tokenize & Pad  →  2. Forward Pass  →  3. Compute Loss│
    │  4. Backward Pass   →  5. Gradient Clip →  6. Optimizer   │
    │                                                           │
    │  Every N steps: Validation → Checkpoint Best → Early Stop │
    └──────────────────────────────────────────────────────────┘
                            │
                            ▼
Phase 7: POST-TRAINING
    Convert Checkpoint to HF Format → Test with vLLM → Evaluate
```

---

## Phase 1: Configuration

**File**: `peft_hf.py` (lines 31-149)

### What Happens

1. Parse command-line arguments
2. Create a `Config` dataclass with all training parameters
3. Auto-generate data and results directory paths

### Key Configuration Groups

```python
@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════
    # MODEL CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    hf_model_id: str = "Qwen/Qwen3-8B"    # HuggingFace model identifier

    # ═══════════════════════════════════════════════════════════════════════
    # DISTRIBUTED TRAINING
    # ═══════════════════════════════════════════════════════════════════════
    num_nodes: int = 1                     # Number of compute nodes
    gpus_per_node: int = 8                 # GPUs per node (typically 8)

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    max_steps: int = 10000                 # Total training steps
    micro_batch_size: int = 2              # Samples per GPU per step
    accumulate_grad_batches: int = 4       # Gradient accumulation steps
    # → Global batch size = 2 × 8 × 4 = 64 samples per update

    # ═══════════════════════════════════════════════════════════════════════
    # LORA CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    lora_rank: int = 32                    # Rank of LoRA matrices (higher = more params)
    lora_alpha: int = 32                   # Scaling factor (typically = rank)
    lora_dropout: float = 0.1              # Dropout for regularization
    lora_target_modules: List[str] = [     # Which layers to apply LoRA
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention layers
        'gate_proj', 'up_proj', 'down_proj'       # Feed-forward layers
    ]
```

### Auto-Generated Paths

The framework creates self-documenting directory structures:

```
datasets/
└── cognitivecomputations_dolphin/
    └── flan1m-alpaca-uncensored/
        └── train=90%-val=5%-test=5%/      ← Encodes split ratios in path
            ├── training.jsonl
            ├── validation.jsonl
            ├── test.jsonl
            └── .data_ready                 ← Marker file for distributed sync

results/
└── Qwen/
    └── Qwen3-8B/
        ├── checkpoints/
        │   └── model-peft-lora-00-1200.ckpt
        └── tb_logs/
```

### SageMaker Consideration

For SageMaker, you'll want to:
- Set `data_dir` to `/opt/ml/input/data/training`
- Set `results_dir` to `/opt/ml/model`
- Pass hyperparameters via SageMaker's hyperparameter mechanism

---

## Phase 2: Data Preparation

**File**: `dataset_module.py` (lines 402-516)

### What Happens

1. **Only Rank 0** downloads and processes the dataset
2. Load from HuggingFace Hub
3. Split into train/validation/test sets
4. Convert each sample to a standard format
5. Save as JSONL files
6. Create a `.data_ready` marker file
7. Other ranks wait for the marker file

### Data Flow Diagram

```
HuggingFace Hub
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  load_dataset("cognitivecomputations/dolphin",      │
│               "flan1m-alpaca-uncensored")           │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Original Dataset Fields:                           │
│  {                                                  │
│    "instruction": "Explain photosynthesis",         │
│    "input": "",                                     │
│    "output": "Photosynthesis is the process..."     │
│  }                                                  │
└─────────────────────────────────────────────────────┘
      │
      │  Apply Templates
      ▼
┌─────────────────────────────────────────────────────┐
│  Converted Format:                                  │
│  {                                                  │
│    "input": "### Instruction:\n                     │
│              Explain photosynthesis\n               │
│              ### Input:\n\n",                       │
│    "output": "### Response:\n                       │
│              Photosynthesis is the process..."      │
│  }                                                  │
└─────────────────────────────────────────────────────┘
      │
      │  Split (90% / 5% / 5%)
      ▼
┌───────────────┬───────────────┬───────────────┐
│ training.jsonl│validation.jsonl│  test.jsonl  │
│   (900K)      │    (50K)       │    (50K)     │
└───────────────┴───────────────┴───────────────┘
```

### Template System

Templates define how to format each sample:

```python
HFDatasetConfig(
    input_template="### Instruction:\n{instruction}\n### Input:\n{input}\n",
    output_template="### Response:\n{output}",
    field_mapping={
        "instruction": "instruction",  # template field → dataset column
        "input": "input",
        "output": "output"
    }
)
```

For complex datasets, use a custom converter:

```python
def custom_converter(sample):
    messages = sample["messages"]  # Chat format dataset
    user = [m for m in messages if m["role"] == "user"][0]["content"]
    assistant = [m for m in messages if m["role"] == "assistant"][0]["content"]
    return {"input": f"User: {user}\n", "output": f"Assistant: {assistant}"}
```

### Distributed Synchronization

```
                    Rank 0                           Ranks 1, 2, 3, ...
                       │                                    │
                       ▼                                    │
              Download Dataset                              │
                       │                                    │
                       ▼                                    │
              Split & Convert                               │
                       │                                    │
                       ▼                                    ▼
            Create .data_ready ──────────────────► Poll for .data_ready
                                                   (every 5 seconds)
                       │                                    │
                       ▼                                    ▼
                  Continue                    ───── Continue ─────
```

---

## Phase 3: Dataset Setup & Tokenization

**File**: `dataset_module.py` (lines 76-250, 518-634)

### What Happens

1. Load the tokenizer from HuggingFace
2. Create `SFTDataset` instances for train/val/test
3. Each sample is tokenized on-the-fly during training
4. DataLoaders are configured with custom collation

### Tokenization Strategy

The key insight: **Input tokens are masked from the loss; only output tokens contribute to learning**.

```
Sample:
  input:  "### Instruction:\nExplain gravity\n### Response:\n"
  output: "Gravity is a fundamental force..."

After Tokenization:
┌──────────────────────────────────────────────────────────────────────┐
│ input_ids: [BOS, 1234, 5678, 91011, ..., 42, 43, 44, 45, 46, ...]   │
│            ├──── Input Tokens ────────┤├─── Output Tokens ────────┤ │
│                                                                      │
│ labels:    [-100, -100, -100, -100, ..., 42, 43, 44, 45, 46, ...]   │
│            ├─── Ignored (masked) ─────┤├─── Used for Loss ────────┤ │
└──────────────────────────────────────────────────────────────────────┘
```

`-100` is a special value that PyTorch's CrossEntropyLoss ignores.

### Truncation Strategy: Preserve Output

When a sequence is too long, the framework truncates the **input from the left**, preserving the output:

```
Max Sequence Length: 2048 tokens

Original:
  Input:  3000 tokens (too long!)
  Output: 200 tokens

After LEFT Truncation:
  Input:  1848 tokens (last 1848 tokens kept)  ← Most recent context preserved
  Output: 200 tokens  ← Fully preserved

Why left truncation? The end of the input (closest to the output) is
usually the most relevant context.
```

### Collation: Dynamic Padding

Batches have variable-length sequences. The collate function pads to the longest sequence in the batch:

```
Batch of 4 samples:
  Sample 1: 150 tokens
  Sample 2: 320 tokens  ← Longest
  Sample 3: 200 tokens
  Sample 4: 180 tokens

After Padding (to 320):
  Sample 1: 150 tokens + 170 [PAD] tokens
  Sample 2: 320 tokens
  Sample 3: 200 tokens + 120 [PAD] tokens
  Sample 4: 180 tokens + 140 [PAD] tokens

Labels padding uses -100 (ignored in loss)
Attention mask is 0 for padding positions
```

---

## Phase 4: Model Initialization

**File**: `peft_hf.py` (lines 153-211)

### What Happens

1. Load the base model from HuggingFace with Flash Attention 2
2. Apply LoRA adapters to target layers
3. Print trainable parameter statistics

### Loading the Base Model

```python
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,           # Allow custom model code
    dtype=torch.bfloat16,             # Use BF16 for memory efficiency
    attn_implementation="flash_attention_2",  # Fast attention
    low_cpu_mem_usage=True,           # Stream weights instead of full load
)
```

**Flash Attention 2** benefits:
- 2-4x faster attention computation
- Reduced memory from O(n²) to O(n) for sequence length n
- Better numerical stability

### Applying LoRA

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,                             # Rank of adaptation matrices
    lora_alpha=32,                    # Scaling factor
    lora_dropout=0.1,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                   'gate_proj', 'up_proj', 'down_proj'],
    bias="none",                      # Don't train biases
)
self.model = get_peft_model(self.model, peft_config)
```

### Parameter Statistics

For Qwen3-8B with rank=32 LoRA:
```
Base model parameters:    8,030,000,000  (frozen)
LoRA trainable params:      41,943,040  (0.52%)
Total parameters:         8,071,943,040
```

You're training less than 1% of the model!

---

## Phase 5: FSDP Strategy Configuration

**File**: `peft_hf.py` (lines 413-446)

This is the most critical phase for distributed training. It configures how the model is distributed across GPUs.

### Auto-Wrap Policy

FSDP needs to know how to split the model. The **auto-wrap policy** specifies that each **decoder layer** should be wrapped as a separate FSDP unit:

```python
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={Qwen3DecoderLayer},  # ⚠️ Model-specific!
)
```

**CRITICAL**: When switching model families, you must change this import:

| Model Family | Decoder Layer Class |
|--------------|---------------------|
| Qwen3 | `Qwen3DecoderLayer` |
| Llama 3/3.1 | `LlamaDecoderLayer` |
| Mistral | `MistralDecoderLayer` |
| Phi-3 | `Phi3DecoderLayer` |

### Sharding Strategy

```python
sharding_strategy=ShardingStrategy.FULL_SHARD
```

Options:
- `FULL_SHARD`: Shard parameters, gradients, and optimizer states (maximum memory savings)
- `SHARD_GRAD_OP`: Only shard gradients and optimizer states
- `NO_SHARD`: No sharding (like traditional DDP)

### Mixed Precision

```python
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,       # Store parameters in BF16
    reduce_dtype=torch.bfloat16,      # All-reduce in BF16
    buffer_dtype=torch.bfloat16,      # Buffers in BF16
    cast_forward_inputs=True,         # Cast inputs to BF16
)
```

**Why BFloat16?**
- Same exponent range as FP32 (no overflow/underflow issues)
- Half the memory of FP32
- Native hardware support on A100/H100 GPUs

### Activation Checkpointing

Applied in `configure_model()` after FSDP wrapping:

```python
def check_fn(submodule):
    return isinstance(submodule, Qwen3DecoderLayer)

apply_activation_checkpointing(
    self.model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=check_fn,  # Only checkpoint decoder layers
)
```

### Complete FSDP Strategy

```python
strategy = FSDPStrategy(
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch next layer
    mixed_precision=mixed_precision_policy,
    cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
    sync_module_states=True,          # Broadcast from rank 0
    limit_all_gathers=True,           # Reduce memory spikes
    use_orig_params=True,             # Required for LoRA
)
```

---

## Phase 6: Training Loop

**File**: `peft_hf.py` (lines 244-356)

### Training Step

```python
def training_step(self, batch, batch_idx):
    # 1. Forward pass
    outputs = self.model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],
        attention_mask=batch['attention_mask'],
        use_cache=False,  # Disable KV cache during training
    )

    # 2. Loss is computed automatically by the model
    loss = outputs.loss

    # 3. Log metrics (synchronized across GPUs)
    self.log('train_loss', loss, sync_dist=True)

    return loss  # PyTorch Lightning handles backward pass
```

### Gradient Clipping (Manual for FSDP + LoRA)

```python
def on_before_optimizer_step(self, optimizer):
    # Get only trainable parameters with gradients
    params = [p for p in self.model.parameters()
              if p.requires_grad and p.grad is not None]

    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

**Why manual clipping?** PyTorch Lightning's built-in gradient clipping doesn't work correctly with FSDP + PEFT combination.

### Optimizer: AdamW with Cosine Schedule

```python
# Two parameter groups: with and without weight decay
optimizer_grouped_parameters = [
    {'params': decay_params, 'weight_decay': 0.01},      # Regular weights
    {'params': no_decay_params, 'weight_decay': 0.0},    # Bias/norm layers
]

optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=1e-5,            # Max learning rate
    betas=(0.9, 0.95),  # Momentum parameters
    fused=True,         # Fused kernel for speed
)

# Learning rate schedule: warmup then cosine decay
scheduler = get_cosine_with_min_lr_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=10000,
    min_lr=1e-7,        # Don't decay below this
)
```

### Learning Rate Schedule Visualization

```
Learning Rate
    ↑
1e-5│         ╭──────────────────╮
    │        ╱                    ╲
    │       ╱                      ╲
    │      ╱                        ╲
    │     ╱                          ╲
1e-7│────╱                            ╲────────
    └────┬────────────────────────────┬───────→ Steps
         100                        10000
         ↑                            ↑
      Warmup                     Minimum LR
```

### Callbacks

```python
# Early Stopping: Stop if validation loss doesn't improve
EarlyStopping(
    monitor='val_loss',
    patience=3,           # Stop after 3 checks with no improvement
    min_delta=0.001,      # Minimum change to count as improvement
)

# Model Checkpoint: Save the best model
ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,         # Keep only the best checkpoint
    dirpath='results/Qwen/Qwen3-8B/checkpoints',
)
```

---

## Phase 7: Post-Training (Conversion & Testing)

### Checkpoint Conversion

**File**: `convert_checkpoint_to_hf.py`

PyTorch Lightning saves checkpoints with a `model.` prefix on state dict keys. This script converts to standard HuggingFace format:

```
PyTorch Lightning Checkpoint (.ckpt)
              │
              ▼
┌─────────────────────────────────────┐
│ 1. Load checkpoint state dict       │
│ 2. Detect: LoRA or Full Fine-tune   │
│ 3. Remove 'model.' prefix from keys │
│ 4. Load into base model             │
│ 5. Merge LoRA weights (optional)    │
│ 6. Save as HuggingFace format       │
└─────────────────────────────────────┘
              │
              ▼
     HuggingFace Model Directory
     ├── config.json
     ├── model.safetensors
     └── tokenizer files
```

**Merged vs Adapter**:
- `--no_merge`: Save LoRA adapter separately (smaller, requires PEFT for inference)
- Default (merged): Weights integrated into base model (works everywhere)

### Testing with vLLM

**File**: `test_checkpoint.py`

vLLM is a high-performance inference engine:

```python
# Load merged model
llm = LLM(
    model=temp_model_path,
    tensor_parallel_size=8,         # Distribute across 8 GPUs
    gpu_memory_utilization=0.9,     # Use 90% of GPU memory
    max_model_len=8192,             # Max context length
    dtype="bfloat16",
)

# Batch inference
sampling_params = SamplingParams(
    temperature=0.1,   # Low for deterministic outputs
    top_p=0.95,        # Nucleus sampling
    max_tokens=512,    # Max output length
)

outputs = llm.generate(prompts, sampling_params)
```

### Evaluation: BERTScore

```python
from bert_score import score

# Compare predictions to reference outputs
P, R, F1 = score(predictions, references, lang="en")
print(f"BERTScore F1: {F1.mean():.4f}")
```

BERTScore measures semantic similarity, not exact match.

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
    hyperparameters = json.loads(os.environ.get('SM_HP_CONFIG', '{}'))

    # SageMaker paths
    training_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    # Override config with SageMaker paths
    config = Config(
        data_dir=training_dir,
        results_dir=model_dir,
        **hyperparameters
    )

    # Run training (same as original main())
    # ...

if __name__ == '__main__':
    main()
```

#### 2. Data Location

SageMaker expects data in specific locations:

```
/opt/ml/
├── input/
│   └── data/
│       └── training/          ← Your JSONL files go here
│           ├── training.jsonl
│           ├── validation.jsonl
│           └── test.jsonl
├── model/                     ← Output directory
│   ├── checkpoints/
│   └── tb_logs/
└── code/                      ← Your scripts
    ├── peft_hf.py
    └── dataset_module.py
```

#### 3. SageMaker Estimator

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='./ptl',
    role=role,
    instance_count=1,
    instance_type='ml.p4d.24xlarge',  # 8x A100 GPUs
    framework_version='2.3.0',
    py_version='py311',

    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },

    hyperparameters={
        'hf_model_id': 'Qwen/Qwen3-8B',
        'max_steps': 10000,
        'micro_batch_size': 2,
        'accumulate_grad_batches': 4,
    },

    # Environment variables for FSDP
    environment={
        'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
        'TOKENIZERS_PARALLELISM': 'false',
    },
)

estimator.fit({
    'training': 's3://your-bucket/datasets/dolphin/'
})
```

#### 4. Multi-Node Training

For larger models, use multiple instances:

```python
estimator = PyTorch(
    # ...
    instance_count=4,  # 4 nodes × 8 GPUs = 32 GPUs
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },
)
```

Update the config:
```python
config = Config(
    num_nodes=4,
    gpus_per_node=8,
    # ...
)
```

#### 5. Dockerfile Considerations

Create a custom Docker image if needed:

```dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu22.04-sagemaker

# Install additional dependencies
RUN pip install \
    lightning==2.3.0 \
    transformers==4.44.0 \
    peft==0.12.0 \
    flash-attn==2.6.0 \
    bert-score==0.3.13

# Copy code
COPY . /opt/ml/code/
WORKDIR /opt/ml/code
```

### Model Saving for SageMaker

After training, save in HuggingFace format for SageMaker endpoints:

```python
# At end of training
if trainer.is_global_zero:
    # Convert to HuggingFace format
    model.model.save_pretrained(os.path.join(model_dir, 'model'))
    tokenizer.save_pretrained(os.path.join(model_dir, 'model'))

    # Create inference code
    with open(os.path.join(model_dir, 'code', 'inference.py'), 'w') as f:
        f.write(inference_code)
```

---

## Quick Reference: File Responsibilities

| File | Lines | Purpose |
|------|-------|---------|
| `peft_hf.py` | 590 | Main orchestration: Config, Model, FSDP, Training |
| `dataset_module.py` | 634 | Data loading, conversion, tokenization, DataLoaders |
| `test_checkpoint.py` | 350 | vLLM inference and BERTScore evaluation |
| `convert_checkpoint_to_hf.py` | 233 | Checkpoint conversion to HuggingFace format |

### Key Functions Quick Reference

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `Config` dataclass | peft_hf.py:31 | All training hyperparameters |
| `HFCausalLMModule` | peft_hf.py:153 | PyTorch Lightning model wrapper |
| `configure_strategy()` | peft_hf.py:413 | FSDP configuration |
| `HFDatasetConfig` | dataset_module.py:16 | Dataset configuration |
| `SFTDataset` | dataset_module.py:76 | Tokenization and label masking |
| `GeneralizedHFDataModule` | dataset_module.py:253 | Lightning DataModule |

---

## Summary

This framework provides a production-ready pipeline for fine-tuning LLMs with:

1. **Memory Efficiency**: FSDP + LoRA + Activation Checkpointing = Train 8B models on 8 GPUs
2. **Flexibility**: Support for any HuggingFace model and dataset
3. **Robustness**: Early stopping, gradient clipping, mixed precision
4. **Deployment Ready**: Convert to HuggingFace format for vLLM/TGI/SageMaker

For SageMaker adaptation, the main changes are:
- Update data/model paths to SageMaker conventions
- Use SageMaker's distributed training distribution
- Save models in HuggingFace format for endpoint deployment
