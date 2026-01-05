# Ray Train FSDP Training Pipeline: Complete Guide

This document provides a comprehensive explanation of the Ray Train framework for fine-tuning Large Language Models (LLMs) with FSDP. It's designed for practitioners who want to understand the complete training pipeline and adapt it for Amazon SageMaker AI.

---

## Table of Contents

1. [Overview: What This Framework Does](#overview-what-this-framework-does)
2. [Key Concepts Explained](#key-concepts-explained)
3. [The Training Pipeline: 7 Phases](#the-training-pipeline-7-phases)
4. [Phase 1: Ray Cluster Initialization](#phase-1-ray-cluster-initialization)
5. [Phase 2: Training Configuration](#phase-2-training-configuration)
6. [Phase 3: Data Preparation](#phase-3-data-preparation)
7. [Phase 4: Model Initialization](#phase-4-model-initialization)
8. [Phase 5: FSDP & Trainer Setup](#phase-5-fsdp--trainer-setup)
9. [Phase 6: Distributed Training Loop](#phase-6-distributed-training-loop)
10. [Phase 7: Post-Training (Conversion & Testing)](#phase-7-post-training-conversion--testing)
11. [Ray Train vs Other Frameworks](#ray-train-vs-other-frameworks)
12. [SageMaker Adaptation Guide](#sagemaker-adaptation-guide)
13. [Quick Reference: File Responsibilities](#quick-reference-file-responsibilities)

---

## Overview: What This Framework Does

This framework fine-tunes pre-trained language models using Ray Train for distributed orchestration:

| Technology | Purpose |
|------------|---------|
| **Ray Train** | Distributed training orchestration with fault tolerance |
| **TorchTrainer** | Ray's wrapper for PyTorch distributed training |
| **HuggingFace Trainer** | High-level training loop within each worker |
| **FSDP** (Fully Sharded Data Parallel) | Memory-efficient model sharding |
| **PEFT/LoRA** | Parameter-efficient fine-tuning |

**Key Advantage**: Ray Train provides automatic fault tolerance, elastic scaling, and seamless integration with Ray's ecosystem (Ray Serve for inference, Ray Data for preprocessing).

---

## Key Concepts Explained

### What is Ray Train?

Ray Train is a distributed training library that orchestrates training across multiple GPUs/nodes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAY CLUSTER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   Driver Node   │  ← Your main() runs here                               │
│  │  (Ray Head)     │  ← TorchTrainer.fit() orchestrates from here           │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           │  Spawns workers                                                 │
│           ▼                                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        WORKER POOL                                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │    │
│  │  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  ...      │    │
│  │  │  GPU 0   │  │  GPU 1   │  │  GPU 2   │  │  GPU 3   │           │    │
│  │  │          │  │          │  │          │  │          │           │    │
│  │  │train_func│  │train_func│  │train_func│  │train_func│           │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │    │
│  │       │              │              │              │               │    │
│  │       └──────────────┴──────────────┴──────────────┘               │    │
│  │                      │                                              │    │
│  │                      ▼                                              │    │
│  │            NCCL Communication Ring                                  │    │
│  │            (Gradient sync, FSDP sharding)                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How Ray Train Works

1. **Driver** creates a `TorchTrainer` with configuration
2. **TorchTrainer** spawns worker processes (one per GPU)
3. Each worker executes `train_func()` independently
4. Workers communicate via NCCL for gradient synchronization
5. Workers report metrics back to driver via `train.report()`
6. Ray handles fault tolerance (restarts failed workers)

### Ray Train's Key Components

```python
# ScalingConfig: How many workers and resources
scaling_config = ScalingConfig(
    num_workers=8,                    # One per GPU
    use_gpu=True,
    resources_per_worker={"CPU": 4, "GPU": 1},
    placement_strategy="SPREAD",      # Distribute across nodes
)

# TorchTrainer: The distributed trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_func,  # Function each worker runs
    train_loop_config=config.to_dict(), # Config passed to workers
    scaling_config=scaling_config,
    torch_config=TorchConfig(backend="nccl"),  # GPU communication
    run_config=RunConfig(...),          # Checkpointing, failure handling
)

# Execute training
result = trainer.fit()
```

---

## The Training Pipeline: 7 Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE OVERVIEW                          │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: RAY CLUSTER INITIALIZATION
    ray.init() → Detect GPUs → Initialize cluster
                            │
                            ▼
Phase 2: TRAINING CONFIGURATION
    CLI Arguments → Config dataclass → ScalingConfig + TorchTrainer
                            │
                            ▼
Phase 3: DATA PREPARATION (Inside workers, Rank 0 only)
    HuggingFace Dataset → Split → JSONL → Wait for other ranks
                            │
                            ▼
Phase 4: MODEL INITIALIZATION (Each worker)
    Load Base Model → Flash Attention 2 → Gradient Checkpointing → LoRA
                            │
                            ▼
Phase 5: FSDP & TRAINER SETUP (Each worker)
    FSDP Config → TrainingArguments → HF Trainer with Callbacks
                            │
                            ▼
Phase 6: DISTRIBUTED TRAINING LOOP
    ┌──────────────────────────────────────────────────────────────┐
    │  Each worker runs trainer.train():                           │
    │  • Forward/Backward with FSDP gradient sync                  │
    │  • RayTrainReportCallback sends metrics to driver            │
    │  • SaveOnBestMetricCallback checkpoints on improvement       │
    │  • EarlyStoppingCallback monitors convergence                │
    └──────────────────────────────────────────────────────────────┘
                            │
                            ▼
Phase 7: POST-TRAINING
    Convert Checkpoint → HuggingFace Format → Test with vLLM
```

---

## Phase 1: Ray Cluster Initialization

**File**: `ray_train_sft.py` (lines 420-428)

### Ray Initialization

```python
def main():
    args = parse_args()
    config = Config.from_args(args)

    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()

    # Detect available GPUs
    available_gpus = int(ray.available_resources().get("GPU", 0))
    print(f"Available GPUs: {available_gpus}")
```

### What Happens During ray.init()

1. **Local Mode**: Starts a single-node Ray cluster on your machine
2. **Cluster Mode**: Connects to existing Ray cluster (for multi-node)
3. **Resource Detection**: Discovers CPUs, GPUs, memory
4. **Object Store**: Initializes shared memory for data passing

### Multi-Node Ray Cluster

For multi-node training, start Ray on each node:

```bash
# Head node
ray start --head --port=6379

# Worker nodes
ray start --address='<head-node-ip>:6379'

# Then run your script
python ray_train_sft.py
```

---

## Phase 2: Training Configuration

**File**: `ray_train_sft.py` (lines 53-168)

### Config Dataclass

```python
@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════
    # MODEL SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    hf_model_id: str = "Qwen/Qwen3-8B"

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    max_steps: int = 10000
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03              # 3% of steps for warmup
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    # ═══════════════════════════════════════════════════════════════════════
    # LORA SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    full_ft: bool = False                   # False = LoRA, True = full fine-tune

    # ═══════════════════════════════════════════════════════════════════════
    # EARLY STOPPING
    # ═══════════════════════════════════════════════════════════════════════
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
```

### Serialization for Ray

Config must be serializable to pass to workers:

```python
def to_dict(self):
    """Convert to dictionary for Ray Train."""
    config_dict = asdict(self)
    config_dict['hf_dataset_config'] = asdict(self.hf_dataset_config)
    return config_dict

@classmethod
def from_dict(cls, config_dict: Dict) -> 'Config':
    """Reconstruct Config from dictionary."""
    hf_config_dict = config_dict.pop('hf_dataset_config', {})
    hf_config = HFDatasetConfig(**hf_config_dict)
    return cls(hf_dataset_config=hf_config, **config_dict)
```

### ScalingConfig

Defines how training scales across GPUs:

```python
scaling_config = ScalingConfig(
    num_workers=available_gpus,           # One worker per GPU
    use_gpu=True,
    resources_per_worker={
        "CPU": 4,                         # CPUs per worker (for data loading)
        "GPU": 1                          # GPUs per worker
    },
    placement_strategy="SPREAD",          # Distribute across nodes
)
```

**Placement Strategies**:
- `SPREAD`: Distribute workers across nodes (best for multi-node)
- `PACK`: Pack workers onto fewest nodes (best for single-node)
- `STRICT_SPREAD`: Fail if can't spread evenly

### TorchTrainer Setup

```python
trainer = TorchTrainer(
    train_loop_per_worker=train_func,     # Function each worker executes
    train_loop_config=config.to_dict(),   # Passed to train_func
    scaling_config=scaling_config,
    torch_config=train.torch.TorchConfig(
        backend="nccl",                   # GPU communication backend
        timeout_s=7200                    # 2-hour timeout for operations
    ),
    run_config=RunConfig(
        name=f"{config.hf_model_id.replace('/', '-')}",
        storage_path=str(Path(config.results_dir).absolute()),
        checkpoint_config=CheckpointConfig(num_to_keep=2),
        failure_config=FailureConfig(max_failures=2),  # Fault tolerance
    ),
)
```

---

## Phase 3: Data Preparation

**File**: `dataset_module.py`

### Distributed Data Loading

Only rank 0 prepares data; others wait:

```python
def load_and_prepare_datasets(config, dataset_root, tokenizer, max_seq_length, rank):
    dataset_root = Path(dataset_root).absolute()
    dataset_root.mkdir(parents=True, exist_ok=True)

    marker_file = dataset_root / ".data_ready"

    if rank == 0:
        if not marker_file.exists():
            # Load from HuggingFace
            hf_dataset = _load_and_split_dataset(config)

            # Convert to JSONL
            _convert_hf_dataset_to_jsonl(hf_dataset['train'], dataset_root / "training.jsonl", config)
            _convert_hf_dataset_to_jsonl(hf_dataset['val'], dataset_root / "validation.jsonl", config)
            _convert_hf_dataset_to_jsonl(hf_dataset['test'], dataset_root / "test.jsonl", config)

            # Signal completion
            marker_file.write_text('ready')
    else:
        # Wait for rank 0
        while not marker_file.exists():
            print(f"Rank {rank} waiting for data preparation...")
            time.sleep(10)

    # Synchronize all ranks
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Create datasets
    train_dataset = SFTDataset(dataset_root / "training.jsonl", tokenizer, max_seq_length)
    eval_dataset = SFTDataset(dataset_root / "validation.jsonl", tokenizer, max_seq_length)

    return train_dataset, eval_dataset
```

### SFTDataset with Label Masking

```python
class SFTDataset(Dataset):
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize full text
        full_text = sample['input'] + sample['output']
        tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_length)

        input_ids = tokenized['input_ids']
        labels = list(input_ids)

        # Find where input ends
        input_tokenized = self.tokenizer(sample['input'])
        input_length = len(input_tokenized['input_ids'])

        # Mask input portion (set to -100)
        for i in range(min(input_length, len(labels))):
            labels[i] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': tokenized.get('attention_mask', [1] * len(input_ids))
        }
```

---

## Phase 4: Model Initialization

**File**: `ray_train_sft.py` (lines 221-245)

### Inside train_func (Each Worker)

```python
def train_func(config_dict: Dict):
    # Get Ray Train context
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    local_rank = train.get_context().get_local_rank()

    # Set CUDA device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        config.hf_model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,                    # Memory efficient
        attn_implementation="flash_attention_2",  # Fast attention
        use_cache=False,                          # Disable for training
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
```

### Applying LoRA

```python
    if not config.full_ft:
        # Prepare for efficient training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[m.strip() for m in config.lora_target_modules.split(',')],
            bias="none",
        )

        # Apply LoRA
        model = get_peft_model(model, peft_config)

        if rank == 0:
            model.print_trainable_parameters()
            # Output: trainable params: 41,943,040 || all params: 8B || trainable%: 0.52%
```

---

## Phase 5: FSDP & Trainer Setup

**File**: `ray_train_sft.py` (lines 256-324)

### FSDP Configuration

```python
fsdp_config = {
    "fsdp_sharding_strategy": "FULL_SHARD",       # Maximum memory savings
    "fsdp_state_dict_type": "FULL_STATE_DICT",    # Full checkpoint format
    "fsdp_offload_params": False,                  # Keep on GPU (faster)
    "fsdp_backward_prefetch": "BACKWARD_PRE",     # Prefetch during backward
    "fsdp_forward_prefetch": False,
    "fsdp_use_orig_params": True,                 # Required for LoRA
    "fsdp_cpu_ram_efficient_loading": True,       # Efficient loading
    "fsdp_sync_module_states": True,              # Sync from rank 0
    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",  # Auto-detect layers
}
```

### TrainingArguments

```python
training_args = TrainingArguments(
    output_dir=config.results_dir,
    max_steps=config.max_steps,

    # Batch configuration
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,

    # Optimizer
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    warmup_ratio=config.warmup_ratio,
    lr_scheduler_type=config.lr_scheduler_type,  # "cosine"
    max_grad_norm=config.max_grad_norm,
    optim="adamw_torch_fused",                   # Fused optimizer (faster)

    # Precision
    bf16=True,

    # Gradient checkpointing
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # FSDP
    fsdp=["full_shard", "auto_wrap"],
    fsdp_config=fsdp_config,

    # Distributed settings
    ddp_find_unused_parameters=False,
    ddp_timeout=7200,                            # 2-hour timeout
    save_on_each_node=False,                     # Single checkpoint location

    # Evaluation
    eval_strategy="steps",
    eval_steps=config.eval_steps,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Saving (controlled by callback)
    save_strategy="no",
    save_total_limit=config.save_total_limit,
)
```

### Callbacks

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[
        RayTrainReportCallback(),      # Reports metrics to Ray
        SaveOnBestMetricCallback(),    # Saves on improvement
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        ),
    ],
)
```

### RayTrainReportCallback

This callback integrates HuggingFace Trainer with Ray Train:

```python
from ray.train.huggingface.transformers import RayTrainReportCallback

# Automatically:
# - Reports training metrics to Ray driver
# - Handles checkpoint synchronization
# - Enables Ray's fault tolerance for training
```

### SaveOnBestMetricCallback

Custom callback to save only when metrics improve:

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

## Phase 6: Distributed Training Loop

### Inside Each Worker

```python
def train_func(config_dict: Dict):
    # ... setup code ...

    # Execute training
    train_result = trainer.train()

    # Synchronize GPUs
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    metrics = train_result.metrics

    # Only rank 0 saves metrics to disk
    if rank == 0:
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # All ranks report to Ray (collective operation)
    train.report(metrics)

    # Final synchronization
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
```

### Training Flow Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED TRAINING LOOP                                │
└─────────────────────────────────────────────────────────────────────────────┘

For each training step:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING (each worker loads different samples)                       │
│    Worker 0: samples[0:batch_size]                                          │
│    Worker 1: samples[batch_size:2*batch_size]                               │
│    ...                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. FORWARD PASS (FSDP gathers weights as needed)                            │
│    Each worker computes loss on its samples                                 │
│    FSDP temporarily gathers full layer weights                              │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. BACKWARD PASS (FSDP handles gradient sync)                               │
│    Gradients computed locally                                               │
│    FSDP all-reduces gradients across workers                                │
│    Backward prefetch optimizes communication                                │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. OPTIMIZER STEP (after gradient_accumulation_steps)                       │
│    Gradient clipping (max_grad_norm=1.0)                                    │
│    AdamW fused optimizer step                                               │
│    Learning rate scheduler step                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. METRICS & CHECKPOINTING                                                  │
│    RayTrainReportCallback → Reports to Ray driver                           │
│    SaveOnBestMetricCallback → Saves if improved                             │
│    EarlyStoppingCallback → Checks for convergence                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fault Tolerance

Ray Train provides automatic fault tolerance:

```python
run_config=RunConfig(
    failure_config=FailureConfig(max_failures=2),  # Allow 2 worker failures
    checkpoint_config=CheckpointConfig(num_to_keep=2),  # Keep 2 checkpoints
)
```

If a worker fails:
1. Ray detects the failure
2. Restarts the worker
3. Loads from latest checkpoint
4. Training continues

---

## Phase 7: Post-Training (Conversion & Testing)

### Checkpoint Format

Ray Train saves checkpoints in a nested structure:

```
results/
└── Qwen-Qwen3-8B/
    └── TorchTrainer_*/
        └── checkpoint_*/
            └── checkpoint/
                ├── model.safetensors
                ├── adapter_config.json  (if LoRA)
                └── tokenizer files
```

### Checkpoint Conversion

**File**: `convert_checkpoint_to_hf.py`

```python
def convert_ray_train_to_hf(base_model_id, checkpoint_path, output_dir, merge_lora, config):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    if config.full_ft:
        # Full fine-tuning: load directly
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        # LoRA: load base + adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_path)

        if merge_lora:
            # Merge for deployment
            model = model.merge_and_unload()

    # Save in HuggingFace format
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
```

### Testing with vLLM

**File**: `test_checkpoint.py`

```python
def load_model_with_vllm(config):
    # Load and merge checkpoint
    model = load_checkpoint_in_memory(config)

    # Save to temp directory
    temp_dir = tempfile.mkdtemp()
    model.save_pretrained(temp_dir)

    # Initialize vLLM
    llm = LLM(
        model=temp_dir,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        dtype="bfloat16",
    )

    return llm

def generate_and_save_predictions(llm, config):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
    )

    # Batched inference
    for batch in batches:
        outputs = llm.generate(batch, sampling_params)
        # Save predictions
```

---

## Ray Train vs Other Frameworks

| Aspect | Ray Train | PyTorch Lightning | HF Accelerate |
|--------|-----------|-------------------|---------------|
| **Orchestration** | Ray cluster | Python process | accelerate launch |
| **Fault Tolerance** | Built-in (auto-restart) | Manual | Manual |
| **Elastic Scaling** | Yes (add/remove workers) | No | No |
| **Configuration** | Python + Config dict | Python code | YAML file |
| **FSDP Integration** | Via HF Trainer | Native FSDPStrategy | Via config |
| **Multi-Node** | Ray cluster (easy) | torchrun (manual) | accelerate (manual) |
| **Inference** | Ray Serve | External | External |

### When to Use Ray Train

- **Multi-node training** with easy setup
- **Fault tolerance** required (long training jobs)
- **Elastic scaling** (add GPUs during training)
- **Integration with Ray ecosystem** (Ray Serve, Ray Data)
- **Cloud-native** deployments (Kubernetes, AWS)

---

## SageMaker Adaptation Guide

### Ray on SageMaker

There are two approaches:

#### Option 1: SageMaker Training with Ray

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='ray_train_sft.py',
    source_dir='./ray_train',
    role=role,
    instance_count=2,
    instance_type='ml.p4d.24xlarge',
    framework_version='2.3.0',
    py_version='py311',

    hyperparameters={
        'hf_model_id': 'Qwen/Qwen3-8B',
        'max_steps': 10000,
    },

    environment={
        'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
    },
)
```

Then modify the script to initialize Ray properly:

```python
def main():
    # SageMaker multi-node Ray initialization
    import socket

    if 'SM_HOSTS' in os.environ:
        hosts = json.loads(os.environ['SM_HOSTS'])
        current_host = os.environ['SM_CURRENT_HOST']

        if current_host == hosts[0]:
            # Head node
            ray.init(address='auto', _node_ip_address=socket.gethostbyname(socket.gethostname()))
        else:
            # Worker node - connect to head
            head_ip = socket.gethostbyname(hosts[0])
            ray.init(address=f'{head_ip}:6379')
    else:
        ray.init()
```

#### Option 2: Amazon SageMaker HyperPod with Ray

Use SageMaker HyperPod for managed Ray clusters:

```python
# hyperpod_config.yaml
cluster:
  instance_type: ml.p4d.24xlarge
  instance_count: 4
  ray:
    head_node_type: ml.p4d.24xlarge
    worker_node_type: ml.p4d.24xlarge
```

### Data Path Adaptation

```python
# Adapt paths for SageMaker
if 'SM_CHANNEL_TRAINING' in os.environ:
    config.data_dir = os.environ['SM_CHANNEL_TRAINING']
    config.results_dir = os.environ['SM_MODEL_DIR']
```

### Model Saving for SageMaker Endpoints

```python
# After training, save in HuggingFace format
if train.get_context().get_world_rank() == 0:
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    # Merge LoRA and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
```

---

## Quick Reference: File Responsibilities

| File | Lines | Purpose |
|------|-------|---------|
| `ray_train_sft.py` | 475 | Main training with Ray Train + FSDP |
| `dataset_module.py` | 213 | Dataset loading, conversion, tokenization |
| `test_checkpoint.py` | 349 | vLLM inference and BERTScore evaluation |
| `convert_checkpoint_to_hf.py` | 219 | Checkpoint conversion to HuggingFace format |

### Key Functions Quick Reference

| Function/Class | File:Line | Purpose |
|----------------|-----------|---------|
| `Config` | ray_train_sft.py:53 | All training configuration |
| `train_func()` | ray_train_sft.py:171 | Worker training function |
| `SaveOnBestMetricCallback` | ray_train_sft.py:32 | Checkpoint on improvement |
| `HFDatasetConfig` | dataset_module.py:15 | Dataset configuration |
| `SFTDataset` | dataset_module.py:32 | Tokenization with label masking |
| `load_and_prepare_datasets()` | dataset_module.py:155 | Distributed data loading |

### CLI Quick Reference

```bash
# Basic training
python ray_train_sft.py

# Custom model
python ray_train_sft.py --hf_model_id "meta-llama/Llama-3-8B"

# Full fine-tuning
python ray_train_sft.py --full_ft

# Custom dataset
python ray_train_sft.py \
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

This framework provides a production-ready pipeline for distributed LLM fine-tuning using Ray Train:

1. **Fault Tolerance**: Automatic worker restart on failures
2. **Easy Scaling**: Add GPUs by changing `num_workers`
3. **FSDP Integration**: Memory-efficient training via HuggingFace Trainer
4. **Ray Ecosystem**: Seamless integration with Ray Serve for inference
5. **Cloud Ready**: Works with SageMaker, Kubernetes, and other cloud platforms

For SageMaker adaptation:
- Initialize Ray cluster properly for multi-node
- Map data paths to SageMaker channels
- Save models in HuggingFace format for endpoints
- Consider SageMaker HyperPod for managed Ray clusters
