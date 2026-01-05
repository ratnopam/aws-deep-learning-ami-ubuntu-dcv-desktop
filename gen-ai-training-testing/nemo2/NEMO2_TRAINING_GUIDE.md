# NeMo 2.0 Training Pipeline: Complete Guide

This document provides a comprehensive explanation of NVIDIA NeMo 2.0 framework for fine-tuning Large Language Models (LLMs). It's designed for practitioners who want to understand the complete training pipeline and adapt it for Amazon SageMaker AI.

---

## Table of Contents

1. [Overview: What This Framework Does](#overview-what-this-framework-does)
2. [Key Concepts Explained](#key-concepts-explained)
3. [The Training Pipeline: 7 Phases](#the-training-pipeline-7-phases)
4. [Phase 1: NeMo Recipe & Configuration](#phase-1-nemo-recipe--configuration)
5. [Phase 2: HuggingFace Checkpoint Import](#phase-2-huggingface-checkpoint-import)
6. [Phase 3: Data Preparation](#phase-3-data-preparation)
7. [Phase 4: Model & Strategy Configuration](#phase-4-model--strategy-configuration)
8. [Phase 5: Training Execution](#phase-5-training-execution)
9. [Phase 6: Inference & Testing](#phase-6-inference--testing)
10. [Phase 7: Checkpoint Export](#phase-7-checkpoint-export)
11. [NeMo 2 vs Other Frameworks](#nemo-2-vs-other-frameworks)
12. [SageMaker Adaptation Guide](#sagemaker-adaptation-guide)
13. [Quick Reference: File Responsibilities](#quick-reference-file-responsibilities)

---

## Overview: What This Framework Does

NeMo 2.0 is NVIDIA's framework for training large language models, built on top of Megatron-Core:

| Technology | Purpose |
|------------|---------|
| **NeMo 2.0** | High-level training framework with recipes |
| **Megatron-Core** | Optimized transformer implementation |
| **NeMo-Run** | Experiment orchestration and configuration |
| **Tensor Parallelism** | Split model layers across GPUs |
| **Pipeline Parallelism** | Split model stages across GPUs |
| **Context Parallelism** | Split attention computation across GPUs |
| **PEFT/LoRA** | Parameter-efficient fine-tuning |

**Key Advantage**: NeMo 2.0 provides optimized training for very large models (70B+) with sophisticated parallelism strategies that go beyond FSDP.

---

## Key Concepts Explained

### NeMo 2.0 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NeMo 2.0 STACK                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       YOUR TRAINING SCRIPT                           │   │
│  │  peft_megatron.py                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         NeMo-Run                                     │   │
│  │  • Experiment orchestration                                         │   │
│  │  • Configuration management (run.Config)                            │   │
│  │  • Executor abstraction (LocalExecutor, SlurmExecutor)              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        NeMo Recipes                                  │   │
│  │  • Pre-configured training recipes (qwen3_8b, llama3_8b, etc.)      │   │
│  │  • Model architecture, optimizer, scheduler                         │   │
│  │  • Parallelism configuration                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Megatron-Core                                  │   │
│  │  • Optimized transformer layers                                     │   │
│  │  • Tensor/Pipeline/Context parallelism                              │   │
│  │  • Efficient attention implementations                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     PyTorch + NCCL                                   │   │
│  │  • Distributed training primitives                                  │   │
│  │  • GPU communication                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Parallelism Strategies Explained

NeMo 2.0 supports three types of parallelism that can be combined:

#### 1. Tensor Parallelism (TP)

Splits individual layers across GPUs:

```
Single GPU (No TP):
┌──────────────────────────────────┐
│  Linear Layer: [4096 × 4096]     │
│  All weights on one GPU          │
└──────────────────────────────────┘

Tensor Parallelism (TP=4):
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ GPU 0  │ │ GPU 1  │ │ GPU 2  │ │ GPU 3  │
│[4096×  │ │[4096×  │ │[4096×  │ │[4096×  │
│ 1024]  │ │ 1024]  │ │ 1024]  │ │ 1024]  │
└────────┘ └────────┘ └────────┘ └────────┘
     ↓          ↓          ↓          ↓
     └──────────┴──────────┴──────────┘
              All-reduce result
```

**When to use**: Always (reduces per-GPU memory)

#### 2. Pipeline Parallelism (PP)

Splits model into stages across GPUs:

```
No Pipeline Parallelism:
┌────────────────────────────────────────────────┐
│ GPU 0: All 32 layers                           │
└────────────────────────────────────────────────┘

Pipeline Parallelism (PP=4):
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  GPU 0   │→ │  GPU 1   │→ │  GPU 2   │→ │  GPU 3   │
│Layers 1-8│  │Layers9-16│  │Layers17-24│ │Layers25-32│
└──────────┘  └──────────┘  └──────────┘  └──────────┘

Micro-batches flow through pipeline:
  batch1 → batch1 → batch1 → batch1
           batch2 → batch2 → batch2 → batch2
                    batch3 → batch3 → batch3
```

**When to use**: Very large models (70B+) that don't fit with TP alone

#### 3. Context Parallelism (CP)

Splits sequence length across GPUs for attention:

```
No Context Parallelism:
┌────────────────────────────────────────────────┐
│ GPU 0: Full attention [seq_len × seq_len]      │
│        Memory: O(seq_len²)                     │
└────────────────────────────────────────────────┘

Context Parallelism (CP=4):
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  GPU 0   │ │  GPU 1   │ │  GPU 2   │ │  GPU 3   │
│seq[0:512]│ │seq[512:  │ │seq[1024: │ │seq[1536: │
│          │ │   1024]  │ │   1536]  │ │   2048]  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**When to use**: Very long sequences (8K+ tokens)

### Global Batch Size Calculation

```python
# Data Parallelism Size
dp_size = (num_nodes * gpus_per_node) / (tensor_parallel_size * pipeline_parallel_size)

# Global Batch Size
global_batch_size = micro_batch_size * accumulate_grad_batches * dp_size

# Example: 8 GPUs, TP=8, PP=1
dp_size = (1 * 8) / (8 * 1) = 1
global_batch_size = 8 * 8 * 1 = 64

# Example: 32 GPUs, TP=8, PP=2
dp_size = (4 * 8) / (8 * 2) = 2
global_batch_size = 8 * 8 * 2 = 128
```

---

## The Training Pipeline: 7 Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE OVERVIEW                          │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: NEMO RECIPE & CONFIGURATION
    CLI Arguments → Config dataclass → Import NeMo recipe
                            │
                            ▼
Phase 2: HUGGINGFACE CHECKPOINT IMPORT (Rank 0 only)
    HuggingFace Model → NeMo format checkpoint → Wait for other ranks
                            │
                            ▼
Phase 3: DATA PREPARATION
    HuggingFace Dataset → Split → JSONL → GeneralizedHFDataModule
                            │
                            ▼
Phase 4: MODEL & STRATEGY CONFIGURATION
    Recipe → Configure TP/PP/CP → Configure Trainer → Configure PEFT
                            │
                            ▼
Phase 5: TRAINING EXECUTION
    ┌──────────────────────────────────────────────────────────────┐
    │  NeMo-Run Experiment:                                        │
    │  • torchrun launcher                                         │
    │  • Megatron distributed training                             │
    │  • Gradient accumulation with pipeline flush                 │
    │  • Model checkpointing (best val_loss)                       │
    │  • Early stopping                                            │
    └──────────────────────────────────────────────────────────────┘
                            │
                            ▼
Phase 6: INFERENCE & TESTING
    Load Checkpoint → Dynamic Inference Engine → BERTScore Evaluation
                            │
                            ▼
Phase 7: CHECKPOINT EXPORT
    NeMo Checkpoint → Merge LoRA → Export to HuggingFace Format
```

---

## Phase 1: NeMo Recipe & Configuration

**File**: `peft_megatron.py` (lines 21-136)

### Config Dataclass

```python
@dataclass
class Config:
    # ═══════════════════════════════════════════════════════════════════════
    # MODEL SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    hf_model_id: str = "Qwen/Qwen3-8B"       # HuggingFace model to fine-tune
    recipe_cls_name: str = "qwen3_8b"         # NeMo recipe name

    # ═══════════════════════════════════════════════════════════════════════
    # PARALLELISM CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    num_nodes: int = 1
    gpus_per_node: int = 8
    tensor_parallel_size: int = 8             # Split layers across 8 GPUs
    pipeline_parallel_size: int = 1           # No pipeline parallelism
    context_parallel_size: int = 1            # No context parallelism

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    max_steps: int = 10000
    val_check_interval: int = 800             # Validate every 800 steps
    log_every_n_steps: int = 10
    micro_batch_size: int = 8                 # Per-GPU batch size
    accumulate_grad_batches: int = 8          # Gradient accumulation
    limit_val_batches: int = 80               # Validation batches

    # ═══════════════════════════════════════════════════════════════════════
    # PEFT SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    peft_scheme: str = "lora"                 # PEFT method
    full_ft: bool = False                     # False = LoRA, True = full

    # ═══════════════════════════════════════════════════════════════════════
    # MONITORING (Optional)
    # ═══════════════════════════════════════════════════════════════════════
    enable_megatron_progress_bar: bool = False
    enable_memory_monitor: bool = False
    enable_speed_monitor: bool = False
    enable_runtime_estimator: bool = False
    enable_memory_profile: bool = False
    enable_pytorch_profiler: bool = False
    enable_nsys_callback: bool = False
    use_wandb: bool = False
```

### Recipe Import

NeMo 2.0 uses pre-defined "recipes" for each model:

```python
def import_nemo_recipe(recipe_name):
    """Dynamically import NeMo recipe module."""
    module_path = f"nemo.collections.llm.recipes.{recipe_name}"
    try:
        recipe_module = import_module(module_path)
        return recipe_module
    except ImportError as e:
        print(f"Failed to import {module_path}: {e}")
        return None
```

**Available Recipes**:
- `qwen3_8b`, `qwen3_14b`, `qwen3_70b`
- `llama3_8b`, `llama31_8b`, `llama31_70b`, `llama31_405b`
- `mistral_7b`, `mixtral_8x7b`
- `nemotron_8b`

### Global Batch Size Property

```python
@property
def global_batch_size(self) -> int:
    dp_size = (self.num_nodes * self.gpus_per_node) // self.tensor_parallel_size // self.pipeline_parallel_size
    return self.micro_batch_size * self.accumulate_grad_batches * dp_size
```

---

## Phase 2: HuggingFace Checkpoint Import

**File**: `peft_megatron.py` (lines 147-169)

NeMo uses its own checkpoint format. HuggingFace models must be converted:

```python
def import_hf_ckpt(model_config: run.Config, node_rank: int):
    """Import HuggingFace checkpoint to NeMo format."""
    context_dir = os.path.join(config.nemo_ckpt_dir, 'context')
    weights_dir = os.path.join(config.nemo_ckpt_dir, 'weights')

    if node_rank == 0:
        # Only rank 0 imports
        if not os.path.isdir(context_dir) or not os.path.isdir(weights_dir):
            hf_source = f"hf://{config.hf_model_id}"

            import_ckpt_partial = run.Partial(
                import_ckpt,
                model=model_config,
                source=hf_source,
                output_path=config.nemo_ckpt_dir,
                overwrite=False
            )

            run.run(import_ckpt_partial,
                    executor=run.LocalExecutor(),
                    name=f"{config.recipe_cls_name}_importer")
    else:
        # Other ranks wait
        while not os.path.isdir(context_dir) or not os.path.isdir(weights_dir):
            print(f"Waiting for checkpoint import from rank 0...")
            time.sleep(10)
```

### Checkpoint Structure

```
outputs/Qwen/Qwen3-8B/
└── imported_hf_ckpt/
    ├── context/              # Model configuration and metadata
    │   ├── model_config.yaml
    │   └── ...
    └── weights/              # Model weights in NeMo format
        ├── model.ckpt
        └── ...
```

---

## Phase 3: Data Preparation

**File**: `dataset_module.py`

### GeneralizedHFDataModule

A custom data module that extends NeMo's `FineTuningDataModule`:

```python
class GeneralizedHFDataModule(FineTuningDataModule):
    """Data module for HuggingFace datasets with NeMo 2.0."""

    def __init__(self, config: HFDatasetConfig, hf_model_id: str, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hf_model_id = hf_model_id

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            trust_remote_code=True
        )

    def prepare_data(self):
        """Convert HuggingFace dataset to JSONL format."""
        marker_file = Path(self.dataset_root) / ".data_ready"

        if marker_file.exists():
            print("Dataset already prepared.")
            return

        # Load and split dataset
        hf_dataset = self._load_and_split_dataset()

        # Convert to JSONL
        self._convert_hf_dataset_to_jsonl(hf_dataset['train'], self.train_path)
        self._convert_hf_dataset_to_jsonl(hf_dataset['val'], self.val_path)
        self._convert_hf_dataset_to_jsonl(hf_dataset['test'], self.test_path)

        # Create marker file
        marker_file.write_text('ready')
```

### Data Configuration in Training

```python
def configure_data():
    return run.Config(
        GeneralizedHFDataModule,
        config=config.hf_dataset_config,
        hf_model_id=config.hf_model_id,
        dataset_root=config.data_dir,
        seq_length=config.max_seq_length,
        micro_batch_size=config.micro_batch_size,
        global_batch_size=config.global_batch_size
    )
```

---

## Phase 4: Model & Strategy Configuration

**File**: `peft_megatron.py` (lines 171-294)

### Loading the Fine-tune Recipe

```python
nemo_recipe = nemo_recipe.finetune_recipe(
    dir=config.output_dir,
    name=config.recipe_cls_name,
    num_nodes=config.num_nodes,
    num_gpus_per_node=config.gpus_per_node,
    peft_scheme=None if config.full_ft else config.peft_scheme,
    packed_sequence=False,                    # Disable for fine-tuning stability
)
```

### Parallelism Strategy

```python
# Configure tensor/pipeline/context parallelism
nemo_recipe.trainer.strategy.tensor_model_parallel_size = config.tensor_parallel_size
nemo_recipe.trainer.strategy.pipeline_model_parallel_size = config.pipeline_parallel_size
nemo_recipe.trainer.strategy.context_parallel_size = config.context_parallel_size
```

### Training Parameters

```python
nemo_recipe.data = configure_data()
nemo_recipe.log = configure_logger()
nemo_recipe.resume.restore_config.path = config.nemo_ckpt_dir
nemo_recipe.trainer.max_steps = config.max_steps
nemo_recipe.trainer.num_sanity_val_steps = 1
nemo_recipe.trainer.val_check_interval = config.val_check_interval
nemo_recipe.trainer.limit_val_batches = config.limit_val_batches
nemo_recipe.trainer.accumulate_grad_batches = config.accumulate_grad_batches
nemo_recipe.trainer.callbacks.extend(configure_callbacks())
nemo_recipe.trainer.strategy.ckpt_load_strictness = False
nemo_recipe.tokenizer = "data"                # Use data module's tokenizer
```

### Logger Configuration

```python
def configure_logger():
    # TensorBoard (always enabled)
    tb_logger = run.Config(
        TensorBoardLogger,
        save_dir="tb_logs",
        name="peft_megatron",
    )

    # Weights & Biases (optional)
    wandb_logger = run.Config(WandbLogger, ...) if config.use_wandb else None

    # Model checkpoint callback
    checkpoint_callback = run.Config(
        ModelCheckpoint,
        monitor="val_loss",
        mode="min",
        save_last="link",
        save_top_k=1,
        save_weights_only=True,
    )

    return run.Config(
        nl.NeMoLogger,
        name="nemo_logs",
        tensorboard=tb_logger,
        wandb=wandb_logger,
        log_dir=config.output_dir,
        ckpt=checkpoint_callback
    )
```

### Callbacks Configuration

```python
def configure_callbacks():
    callbacks = []

    # Early stopping (always enabled)
    early_stopping_callback = run.Config(
        EarlyStopping,
        monitor='val_loss',
        min_delta=config.early_stopping_threshold,
        patience=config.early_stopping_patience,
        verbose=True,
        mode='min',
    )
    callbacks.append(early_stopping_callback)

    # Optional monitoring callbacks
    if config.enable_megatron_progress_bar:
        callbacks.append(run.Config(MegatronProgressBar, refresh_rate=config.log_every_n_steps))

    if config.enable_memory_monitor:
        callbacks.append(run.Config(MemoryMonitor))

    if config.enable_speed_monitor:
        callbacks.append(run.Config(SpeedMonitor, window_size=100))

    if config.enable_runtime_estimator:
        callbacks.append(run.Config(RuntimeEstimator))

    # Profiling (mutually exclusive)
    if config.enable_pytorch_profiler:
        callbacks.append(run.Config(PytorchProfilerCallback, start_step=0, end_step=1, trace_dir=...))

    if config.enable_nsys_callback:
        callbacks.append(run.Config(NsysCallback, start_step=0, end_step=1, ranks=[0]))

    return callbacks
```

---

## Phase 5: Training Execution

**File**: `peft_megatron.py` (lines 296-385)

### Executor Configuration

```python
def configure_executor():
    return run.LocalExecutor(
        ntasks_per_node=config.gpus_per_node,
        nodes=config.num_nodes,
        launcher="torchrun",                  # Uses PyTorch's torchrun
    )
```

### Running the Experiment

```python
def main():
    # ... setup code ...

    try:
        print(f"Starting NeMo recipe {config.recipe_cls_name} fine-tuning...")
        executor = configure_executor()
        exp_title = "full_ft" if config.full_ft else f"peft_{config.peft_scheme}"

        with Experiment(title=exp_title, executor=executor,
                        log_level=config.log_level, base_dir=config.output_dir) as exp:
            exp.add(nemo_recipe, tail_logs=True, name=config.recipe_cls_name)
            exp.run(detach=False)

        print("Fine-tuning completed successfully!")
        print(f"Outputs saved to: {config.output_dir}")

    except Exception as e:
        print(f"Error during training: {e}")
```

### Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEMO 2.0 TRAINING EXECUTION                              │
└─────────────────────────────────────────────────────────────────────────────┘

1. Experiment.run() called
   │
   ▼
2. LocalExecutor launches torchrun
   │
   ▼
3. torchrun spawns processes on each GPU
   │
   ▼
4. Each process initializes:
   ├─ NCCL communication groups (TP, PP, DP)
   ├─ Model shards (based on TP/PP config)
   └─ Data loader (based on DP rank)
   │
   ▼
5. Training loop (Megatron-style):
   ┌──────────────────────────────────────────────────────────────┐
   │ For each micro-batch:                                        │
   │   1. Load data (DP-distributed)                              │
   │   2. Forward pass (TP/PP-distributed)                        │
   │   3. Backward pass (TP/PP gradient sync)                     │
   │                                                               │
   │ After accumulate_grad_batches:                               │
   │   4. All-reduce gradients (DP)                               │
   │   5. Optimizer step                                          │
   │   6. Log metrics                                             │
   │                                                               │
   │ At val_check_interval:                                        │
   │   7. Validation loop                                         │
   │   8. Checkpoint if improved                                  │
   │   9. Early stopping check                                    │
   └──────────────────────────────────────────────────────────────┘
   │
   ▼
6. Training complete, save final checkpoint
```

---

## Phase 6: Inference & Testing

**File**: `test_checkpoint.py`

### Dynamic Inference Engine

NeMo 2.0 provides a high-performance inference engine:

```python
def setup_model_and_dynamic_inference(path, trainer, params_dtype):
    """Setup model and create DynamicInferenceEngine."""

    # Create inference context
    inference_context = DynamicInferenceContext(
        kv_cache_params=KVCacheParams(
            buffer_size_gb=config.buffer_size_gb,
            block_size_tokens=config.block_size_tokens,
        ),
        tensor_model_parallel_size=config.tensor_parallel_size,
        max_sequence_length=config.inference_max_seq_length,
    )

    # Wrap model
    inference_wrapper = GPTInferenceWrapper(model, inference_context)

    # Create controller
    controller = TextGenerationController(inference_wrapper, tokenizer)

    # Create engine
    engine = DynamicInferenceEngine(
        controller=controller,
        max_batch_size=config.max_batch_size,
        max_tokens=config.max_tokens,
    )

    return engine
```

### Dynamic Batching

```python
def generate_with_dynamic_inference(engine, prompts, sampling_params):
    """Generate with dynamic batching for efficiency."""
    results = [None] * len(prompts)

    # Add all requests
    for i, prompt in enumerate(prompts):
        try:
            engine.add_request(prompt, sampling_params, request_id=str(i))
        except ContextOverflowError:
            # Batch full, step engine first
            while engine.num_pending_requests > 0:
                completed = engine.step()
                for req_id, output in completed:
                    results[int(req_id)] = output

            # Retry adding request
            engine.add_request(prompt, sampling_params, request_id=str(i))

    # Process remaining
    while engine.num_pending_requests > 0:
        completed = engine.step()
        for req_id, output in completed:
            results[int(req_id)] = output

    return results
```

### BERTScore Evaluation

```python
def evaluate_predictions(output_path):
    """Evaluate predictions using BERTScore."""
    import evaluate

    predictions = []
    references = []

    with open(output_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data['prediction'])
            references.append(data['label'])

    bertscore = evaluate.load('bertscore')
    scores = bertscore.compute(predictions=predictions, references=references, lang='en')

    avg_f1 = sum(scores['f1']) / len(scores['f1'])
    print(f"BERTScore F1: {avg_f1:.4f}")

    return {'bertscore_f1': avg_f1}
```

---

## Phase 7: Checkpoint Export

**File**: `convert_checkpoint_to_hf.py`

### Export to HuggingFace Format

```python
def export_nemo_to_hf(config):
    """Export NeMo checkpoint to HuggingFace format."""
    checkpoint_path = config.nemo_checkpoint_path

    if config.no_merge:
        # Export LoRA adapter only
        output_dir = checkpoint_path + ".hf_peft"
        peft.export_lora(
            checkpoint_path=checkpoint_path,
            output_path=output_dir,
        )
    else:
        # Merge LoRA and export full model
        merged_path = checkpoint_path + ".merged"

        # Step 1: Merge LoRA into base model
        peft.merge_lora(
            checkpoint_path=checkpoint_path,
            output_path=merged_path,
        )

        # Step 2: Export to HuggingFace format
        output_dir = checkpoint_path + ".hf_model"
        api.export_ckpt(
            checkpoint_path=merged_path,
            output_path=output_dir,
            target="hf",
        )

    return Path(output_dir)
```

### Output Structure

```
outputs/Qwen/Qwen3-8B/
└── nemo_logs/
    └── {timestamp}/
        └── checkpoints/
            └── nemo_logs--val_loss=0.5-epoch=0-consumed_samples=10000/
                ├── context/           # NeMo checkpoint
                ├── weights/
                ├── .merged/           # After merge_lora()
                ├── .hf_model/         # After export (merged)
                │   ├── config.json
                │   ├── model.safetensors
                │   └── tokenizer files
                └── .hf_peft/          # After export (adapter only)
                    ├── adapter_config.json
                    └── adapter_model.safetensors
```

---

## NeMo 2 vs Other Frameworks

| Aspect | NeMo 2.0 | PyTorch Lightning | HF Accelerate | Ray Train |
|--------|----------|-------------------|---------------|-----------|
| **Parallelism** | TP + PP + CP | FSDP | FSDP | FSDP |
| **Max Model Size** | 1T+ parameters | ~100B | ~100B | ~100B |
| **Optimization** | Megatron-Core | Standard PyTorch | Standard PyTorch | Standard PyTorch |
| **Configuration** | NeMo-Run configs | Python code | YAML | Python dict |
| **Checkpoint Format** | NeMo (.nemo) | Lightning (.ckpt) | HuggingFace | HuggingFace |
| **Recipes** | Pre-built recipes | Manual setup | Manual setup | Manual setup |
| **Learning Curve** | Steeper | Moderate | Easy | Moderate |

### When to Use NeMo 2.0

- **Very large models** (70B+) requiring tensor/pipeline parallelism
- **NVIDIA GPU clusters** (optimized for A100/H100)
- **Production training** at scale
- **Long sequences** requiring context parallelism
- **Integration with NVIDIA ecosystem** (TensorRT-LLM, Triton)

---

## SageMaker Adaptation Guide

### Running NeMo on SageMaker

#### Option 1: SageMaker Training Job

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='peft_megatron.py',
    source_dir='./nemo2',
    role=role,
    instance_count=4,
    instance_type='ml.p4d.24xlarge',
    framework_version='2.3.0',
    py_version='py311',

    # NeMo-specific container
    image_uri='nvcr.io/nvidia/nemo:24.05',

    hyperparameters={
        'hf_model_id': 'Qwen/Qwen3-8B',
        'recipe_cls_name': 'qwen3_8b',
        'num_nodes': 4,
        'gpus_per_node': 8,
        'tensor_parallel_size': 8,
        'pipeline_parallel_size': 2,
    },
)
```

#### Option 2: SageMaker HyperPod

For large-scale NeMo training:

```yaml
# hyperpod_config.yaml
cluster:
  instance_groups:
    - name: worker
      instance_type: ml.p5.48xlarge
      instance_count: 16

training:
  framework: nemo
  script: peft_megatron.py
```

### Path Adaptation

```python
# Adapt for SageMaker paths
if 'SM_CHANNEL_TRAINING' in os.environ:
    config.data_dir = os.environ['SM_CHANNEL_TRAINING']
    config.output_dir = os.environ['SM_MODEL_DIR']
    config.nemo_ckpt_dir = os.path.join(config.output_dir, 'imported_hf_ckpt')
```

### Multi-Node Configuration

```python
# SageMaker provides these environment variables
if 'SM_HOSTS' in os.environ:
    import json
    hosts = json.loads(os.environ['SM_HOSTS'])
    config.num_nodes = len(hosts)
    config.node_rank = hosts.index(os.environ['SM_CURRENT_HOST'])
```

### Model Export for SageMaker Endpoints

```python
# After training, export to HuggingFace format
if config.node_rank == 0:
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    # Export merged model
    export_nemo_to_hf(ExportConfig(
        nemo_logs_dir=config.output_dir,
        no_merge=False,  # Merge LoRA
    ))

    # Copy to model directory
    shutil.copytree(checkpoint_path + ".hf_model", model_dir)
```

---

## Quick Reference: File Responsibilities

| File | Lines | Purpose |
|------|-------|---------|
| `peft_megatron.py` | 387 | Main training with NeMo 2.0 recipes |
| `dataset_module.py` | ~200 | Dataset loading with GeneralizedHFDataModule |
| `test_checkpoint.py` | ~350 | Dynamic inference and BERTScore evaluation |
| `convert_checkpoint_to_hf.py` | ~200 | NeMo to HuggingFace conversion |

### Key Functions Quick Reference

| Function/Class | File:Line | Purpose |
|----------------|-----------|---------|
| `Config` | peft_megatron.py:22 | All training configuration |
| `import_nemo_recipe()` | peft_megatron.py:138 | Dynamic recipe import |
| `import_hf_ckpt()` | peft_megatron.py:147 | HuggingFace checkpoint import |
| `configure_data()` | peft_megatron.py:171 | Data module configuration |
| `configure_logger()` | peft_megatron.py:183 | Logger and checkpoint setup |
| `configure_callbacks()` | peft_megatron.py:222 | Callbacks including early stopping |
| `GeneralizedHFDataModule` | dataset_module.py | Custom NeMo data module |

### CLI Quick Reference

```bash
# Basic training
python peft_megatron.py

# Custom model and recipe
python peft_megatron.py \
  --hf_model_id "meta-llama/Llama-3.1-8B" \
  --recipe_cls_name "llama31_8b"

# Multi-node with parallelism
python peft_megatron.py \
  --num_nodes 4 \
  --gpus_per_node 8 \
  --tensor_parallel_size 8 \
  --pipeline_parallel_size 2

# Full fine-tuning
python peft_megatron.py --full_ft

# With monitoring
python peft_megatron.py \
  --enable_megatron_progress_bar \
  --enable_memory_monitor \
  --enable_speed_monitor

# Convert checkpoint
python convert_checkpoint_to_hf.py

# Test checkpoint
python test_checkpoint.py --max_samples 1024
```

---

## Summary

NeMo 2.0 provides a production-ready framework for training very large language models:

1. **Advanced Parallelism**: Tensor + Pipeline + Context parallelism for models 70B+
2. **Optimized Training**: Megatron-Core provides highly optimized transformer layers
3. **Recipe-Based**: Pre-configured recipes for popular models
4. **Flexible Configuration**: NeMo-Run provides declarative configuration
5. **Production Ready**: Battle-tested on NVIDIA GPU clusters

For SageMaker adaptation:
- Use NVIDIA NeMo container images
- Configure multi-node using SageMaker environment variables
- Export to HuggingFace format for SageMaker endpoints
- Consider SageMaker HyperPod for large-scale training
