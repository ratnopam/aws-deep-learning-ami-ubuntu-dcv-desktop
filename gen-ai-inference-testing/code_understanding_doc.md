# Gen-AI Inference Testing Framework Documentation

This document provides a comprehensive understanding of the inference testing framework located in `/gen-ai-inference-testing/`. The framework is designed to perform load testing on various AI model inference endpoints using Locust, supporting both NVIDIA CUDA and AWS Neuron accelerators.

---

## Table of Contents

1. [Directory Structure Overview](#directory-structure-overview)
2. [Core Components](#core-components)
3. [Notebook Workflows](#notebook-workflows)
4. [Step-by-Step Testing Pipeline](#step-by-step-testing-pipeline)
5. [Key Concepts](#key-concepts)
6. [Prompt Generation Process (Deep Dive)](#prompt-generation-process-deep-dive)

---

## Directory Structure Overview

### `compose/`
Contains Docker Compose YAML files that orchestrate multi-container deployments. These files define how inference server containers are started, configured, and networked together. Different compose files exist for different inference server and backend combinations.

### `config/`
Houses YAML configuration files organized by model type (encoder/decoder) and inference server. These configs specify:
- Endpoint URLs
- Template formats for API requests
- Prompt generator modules to use
- Template keys for request formatting

Structure:
```
config/
├── encoder/
│   ├── embeddings/
│   ├── reranker/
│   ├── sequence_classification/
│   └── token_classification/
└── decoder/
    └── (LLM configurations)
```

### `containers/`
Contains Dockerfiles and build scripts for creating inference server containers. These include containers for:
- Triton Inference Server with Python backend
- Triton Inference Server with vLLM backend
- OpenAI-compatible servers
- Various hardware targets (CUDA, NeuronX)

### `djl_serving/`
Configuration and model files for Deep Java Library (DJL) Serving, an alternative inference server that supports large model inference with automatic model partitioning.

### `modules/`
Python modules containing prompt generators for different model types:
- **Encoder modules**: Generate prompts for embeddings, reranker, masked language models, token classification, sequence classification
- **Decoder modules**: Generate prompts for LLM text generation

Each module provides a `PromptGenerator` class that yields prompts for load testing.

### `openai_server/`
Scripts and configurations for launching OpenAI-compatible API servers, typically backed by vLLM. Supports both CUDA and NeuronX hardware.

### `scripts/`
Utility shell scripts including:
- `build-containers.sh`: Builds all Docker containers
- Hardware detection scripts
- Server management utilities

### `triton_inference_server/`
Triton-specific configurations organized by model type:
- **embeddings/**: Python backend scripts for embedding models
- **decoder/**: vLLM backend configurations for LLMs
- Model configuration files (`config.pbtxt`)
- Launch scripts for CUDA and NeuronX

---

## Core Components

### `endpoint_user.py` - Locust User Class
The main Locust test user that drives load testing:
- Dynamically loads prompt generator modules based on configuration
- Fills request templates with generated prompts
- Sends HTTP POST requests to the inference endpoint
- Measures response times and success/failure rates

Key features:
- Template-based request formatting
- Support for various API formats (Triton, OpenAI, custom)
- Configurable through environment variables

### `endpoint_utils.py` - Utility Functions
Helper functions for the testing framework:
- `wait_for_inference_ready()`: Polls endpoint until server is healthy
- `inference_request()`: Makes single inference requests for validation
- `load_prompt_generator()`: Dynamically loads prompt generator classes
- Health check implementations for different server types

### `custom_endpoint_handler.py` - LiteLLM Adapter
A SageMaker-compatible endpoint handler that adapts LiteLLM for deployment:
- Converts SageMaker request format to LiteLLM format
- Routes requests to appropriate LLM backends
- Handles response formatting

### `launch.sh` - Docker Compose Orchestrator
Central launch script that:
- Reads environment variables for configuration
- Selects appropriate Docker Compose file based on inference server/engine
- Manages container lifecycle (up/down)
- Handles hardware-specific configurations (CUDA vs NeuronX)

### `run_locust.sh` - Locust Test Runner
Executes Locust in distributed mode:
- Launches master and worker processes
- Configures user count, spawn rate, and run duration
- Outputs results to CSV files for analysis

---

## Notebook Workflows

### `locust_encoder.ipynb` - Encoder Model Testing

Tests embedding and classification models (BERT-based, cross-encoders, etc.):

**Supported encoder types:**
- `embeddings` - Vector embeddings (e.g., BGE-M3)
- `reranker` - Cross-encoder reranking (e.g., BGE-reranker)
- `sequence_classification` - Sentence pair classification
- `token_classification` - NER, POS tagging
- `masked_lm` - Masked language modeling

**Workflow:**
1. Detect hardware (CUDA/Neuron)
2. Install dependencies
3. Build Docker containers
4. Download/cache model from HuggingFace
5. Configure batch size and tensor parallelism
6. Launch Triton Inference Server with Python backend
7. Run Locust load test
8. Visualize results

### `locust_decoder.ipynb` - LLM Testing

Tests large language models for text generation:

**Supported configurations:**
- Triton Inference Server + vLLM backend
- OpenAI-compatible server + vLLM

**Key features:**
- Configurable `MAX_MODEL_LEN` for context length
- `MAX_NUM_SEQS` for batch size control
- `TENSOR_PARALLEL_SIZE` for multi-GPU inference
- Dataset-driven prompts from ShareGPT

**Workflow:**
1. Hardware detection
2. Environment setup
3. Model snapshot download to EFS
4. Configure parallelism settings
5. Launch inference server via Docker Compose
6. Validate endpoint with single request
7. Run distributed Locust test (32 users, 32 workers)
8. Generate results CSV and visualizations

### `litellm.ipynb` - LiteLLM Proxy Testing

Tests inference through LiteLLM, a unified interface for multiple LLM providers:

**Key differences from decoder notebook:**
- Uses LiteLLM as a proxy layer
- Can route to multiple backend providers
- OpenAI-compatible API format
- Useful for testing provider abstraction

---

## Step-by-Step Testing Pipeline

The complete flow from start to finish:

### Phase 1: Hardware Detection
```python
# Detect CUDA GPUs
nvidia-smi --list-gpus | wc -l

# Or detect Neuron devices
neuron-ls -j | grep neuron_device | wc -l
```

### Phase 2: Environment Setup
- Install pip packages (locust, datasets, etc.)
- Set HuggingFace token for gated models
- Configure AWS region

### Phase 3: Container Build
```bash
bash scripts/build-containers.sh
```
Builds Docker images for the selected inference server and backend.

### Phase 4: Model Preparation
```python
snapshot_download(repo_id=hf_model_id, cache_dir=cache_dir, token=hf_token)
```
- Downloads model from HuggingFace Hub
- Caches to EFS for persistence across sessions
- Creates symlinks for container volume mounts

### Phase 5: Server Configuration
Environment variables control behavior:
- `MODEL_ID`: Path to model files
- `MAX_NUM_SEQS`: Batch size
- `TENSOR_PARALLEL_SIZE`: GPU parallelism
- `MAX_MODEL_LEN`: Context length limit
- `INFERENCE_SERVER`: triton_inference_server or openai_server
- `INFERENCE_ENGINE`: python or vllm

### Phase 6: Server Launch
```bash
bash launch.sh up   # Start containers
bash launch.sh down # Stop containers
```

### Phase 7: Health Check
```python
wait_for_inference_ready(config['endpoint_url'], timeout_seconds=1800, interval=15)
```
Polls the endpoint until the server reports ready status.

### Phase 8: Validation Request
```python
response = inference_request(config=config)
assert response.status_code == 200
```
Makes a single request to verify the server is responding correctly.

### Phase 9: Load Testing
```python
os.environ["USERS"] = "32"
os.environ["WORKERS"] = "32"
os.environ["RUN_TIME"] = "120s"
os.environ["SPAWN_RATE"] = "32"
```
Locust runs with:
- 32 concurrent users
- 32 worker processes
- 120 second duration
- Immediate spawn (32/sec)

### Phase 10: Results Analysis
```python
df = pd.read_csv(results_path + "_stats.csv")
```
CSV output includes:
- Request count
- Failure count
- Response time percentiles (50th, 95th, 99th)
- Requests per second
- Failures per second

---

## Key Concepts

### Template System
The framework uses JSON templates that get filled with prompts:
```python
# Template example for Triton
{
    "inputs": [
        {"name": "text_input", "shape": [1], "datatype": "BYTES", "data": ["<PROMPT>"]}
    ]
}
```

The `<PROMPT>` placeholder gets replaced with actual prompts from the generator.

### Prompt Generators
Each model type has a specialized prompt generator:
```python
class PromptGenerator:
    def __init__(self, config):
        self.dataset = load_dataset(...)

    def __iter__(self):
        for item in self.dataset:
            yield self.format_prompt(item)
```

> **Note:** For a comprehensive deep dive into the prompt generation process, see [Prompt Generation Process (Deep Dive)](#prompt-generation-process-deep-dive) section below.

### Dynamic Module Loading
Modules are loaded at runtime based on configuration:
```python
module = importlib.import_module(module_name)
generator_class = getattr(module, generator_name)
```

### Multi-Hardware Support
The framework abstracts hardware differences:
- **CUDA**: Uses standard PyTorch/vLLM
- **NeuronX**: Uses AWS Neuron SDK with `transformers-neuronx`

Container and script selection is automatic based on detected hardware.

### Distributed Load Testing
Locust runs in distributed mode:
- **Master**: Coordinates test, collects results
- **Workers**: Generate actual load
- Workers match user count for maximum concurrency

---

## Configuration Examples

### Encoder Configuration (Reranker)
```yaml
endpoint_url: "http://localhost:8000/v2/models/encoder/infer"
module_dir: "modules"
module_name: "reranker_prompts"
prompt_generator: "PromptGenerator"
template:
  inputs:
    - name: "text_input"
      shape: [1]
      datatype: "BYTES"
      data: ["<PROMPT>"]
template_keys:
  - "inputs.0.data.0"
```

### Decoder Configuration (vLLM)
```yaml
endpoint_url: "http://localhost:8000/v1/completions"
module_dir: "modules"
module_name: "decoder_prompts"
prompt_generator: "ShareGPTPromptGenerator"
template:
  model: "model"
  prompt: "<PROMPT>"
  max_tokens: 256
template_keys:
  - "prompt"
```

---

## Common Operations

### Starting a Test Session
1. Open the appropriate notebook (encoder/decoder/litellm)
2. Set `hf_model_id` and `hf_token`
3. Run all cells sequentially
4. Results appear in `output/locust-testing/`

### Changing Models
1. Update `hf_model_id` variable
2. Set appropriate `encoder_type` or keep defaults for decoder
3. Re-run from model snapshot cell onward

### Adjusting Load
Modify these environment variables:
- `USERS`: Number of simulated users
- `WORKERS`: Number of Locust worker processes
- `RUN_TIME`: Test duration (e.g., "60s", "5m")
- `SPAWN_RATE`: Users spawned per second

### Troubleshooting
- Check `run_locust.log` for Locust errors
- Use `docker logs <container>` for server issues
- Verify model path with `ls ~/snapshots/huggingface/`
- Ensure EFS is mounted if using shared storage

---

## File Quick Reference

| File | Purpose |
|------|---------|
| `endpoint_user.py` | Locust user class for load generation |
| `endpoint_utils.py` | Health checks, request helpers |
| `launch.sh` | Docker Compose orchestration |
| `run_locust.sh` | Locust distributed test launcher |
| `locust_encoder.ipynb` | Encoder model testing notebook |
| `locust_decoder.ipynb` | LLM testing notebook |
| `litellm.ipynb` | LiteLLM proxy testing notebook |
| `config/**/*.yaml` | Endpoint and template configurations |
| `modules/*.py` | Prompt generators by model type |
| `compose/*.yaml` | Docker Compose definitions |
| `containers/` | Dockerfiles for inference servers |

---

## Prompt Generation Process (Deep Dive)

This section provides a detailed explanation of how prompts are generated, loaded, and injected into API requests during load testing.

### Overview

The prompt generation system follows a **pipeline architecture**:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  YAML Config    │───▶│  Dynamic Module  │───▶│  Prompt         │───▶│  Template        │
│  (config/*.yaml)│    │  Loading         │    │  Generator      │    │  Filling         │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                                                      │                       │
                                                      ▼                       ▼
                                               ┌─────────────────┐    ┌──────────────────┐
                                               │  HuggingFace    │    │  HTTP POST       │
                                               │  Dataset        │    │  Request         │
                                               └─────────────────┘    └──────────────────┘
```

### Step 1: Configuration Loading

The process starts with a YAML configuration file that specifies which prompt generator to use:

```yaml
# Example: config/encoder/reranker/triton_inference_server-python.yaml
endpoint_url: "http://localhost:8080/v2/models/model/infer"
module_name: "reranking_pairs"              # Python module name
module_dir: "modules/ms-marco"              # Directory containing the module
prompt_generator: "RerankerInputGenerator"  # Class name to instantiate
template: {
  "inputs": [
    {"name": "query", "shape": [1], "datatype": "BYTES", "data": []},
    {"name": "texts", "shape": [], "datatype": "BYTES", "data": []}
  ]
}
template_keys: ["inputs.[0].data", "inputs.[1].data", "inputs.[1].shape"]
```

**Key configuration fields:**
- `module_dir`: File system path to the Python module
- `module_name`: Name of the Python file (without `.py`)
- `prompt_generator`: Class name within that module
- `template`: JSON structure for the API request
- `template_keys`: Dot-notation paths indicating where to inject generated values

### Step 2: Dynamic Module Loading

The `EndpointClient` class in `endpoint_user.py` dynamically loads the prompt generator at runtime:

```python
# endpoint_user.py lines 21-30
def __init_prompt_generator(self):
    # Add module directory to Python path
    prompt_module_dir = os.getenv("PROMPT_MODULE_DIR", "")
    sys.path.append(prompt_module_dir)

    # Import the module dynamically
    prompt_module_name = os.getenv("PROMPT_MODULE_NAME", None)
    prompt_module = import_module(prompt_module_name)

    # Get the generator class and instantiate it
    prompt_generator_name = os.getenv('PROMPT_GENERATOR_NAME', None)
    prompt_generator_class = getattr(prompt_module, prompt_generator_name)

    # Create generator instance and get iterator
    # Note: double parentheses - first () creates instance, second () calls __call__
    self.text_input_generator = prompt_generator_class()()
```

**Why double parentheses?**
- `prompt_generator_class()` - Creates an instance of the class (calls `__init__`)
- `()` again - Calls the `__call__` method which returns a generator/iterator

### Step 3: Prompt Generator Classes

Each prompt generator follows a consistent pattern:

```python
class PromptGenerator:
    def __init__(self):
        # 1. Load dataset from HuggingFace
        self.dataset = datasets.load_dataset('dataset_name', split='validation')

        # 2. Preprocess/prepare data
        self._prepare_data()

    def __call__(self):
        # 3. Yield prompts one at a time (generator pattern)
        for item in self.dataset:
            yield [formatted_prompt]  # Always yields a list
```

### Step 4: Dataset-Specific Generators

The framework includes specialized generators for different model types:

#### Reranker Generator (MS-MARCO dataset)
**File:** `modules/ms-marco/reranking_pairs.py`

```python
class RerankerInputGenerator:
    def __init__(self, num_candidates_per_query=5, seed=42):
        # Load MS-MARCO passage ranking dataset
        self.dataset = datasets.load_dataset('ms_marco', 'v1.1', split='validation')
        self._build_query_passage_mapping()

    def __call__(self):
        for query in queries:
            # For each query, select candidate documents:
            # - Include relevant passages
            # - Add non-relevant passages from same query
            # - Fill remaining with random corpus passages
            # - Shuffle to randomize order
            yield [query, documents, [len(documents)]]
```

**Output format:** `[query_string, list_of_documents, [document_count]]`

#### Token Classification Generator (CoNLL-2003 dataset)
**File:** `modules/conll2003/token_classification_prompts.py`

```python
class TokenClassificationInputGenerator:
    def __init__(self, max_length=512):
        # Load CoNLL-2003 NER dataset
        self.dataset = datasets.load_dataset('conll2003', split='validation')
        self._prepare_texts()  # Join tokens into sentences

    def __call__(self):
        for idx in shuffled_indices:
            text = self.texts[idx]
            yield [text]  # Single text for NER tagging
```

**Output format:** `[text_string]`

#### Code Generation Generator (CodeAlpaca dataset)
**File:** `modules/sahil2801-codealpca20k/prompt_generator.py`

```python
class PromptGenerator:
    def __init__(self):
        # Load CodeAlpaca coding instruction dataset
        dataset = load_dataset("sahil2801/CodeAlpaca-20k")
        self.dataset = dataset['train']

    def _create_prompt(self, sample):
        # Format instruction with optional input
        if sample['input'].strip():
            prompt = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\n..."
        else:
            prompt = f"Instruction: {sample['instruction']}\n..."
        return prompt

    def __call__(self):
        for sample in self.dataset:
            yield [self._create_prompt(sample)]
```

**Output format:** `[formatted_instruction_string]`

### Step 5: Template Filling

When a request is made, the generated prompt values are injected into the JSON template:

```python
# endpoint_user.py lines 34-48
def _fill_template(self, template: dict, template_keys: list, inputs: list):
    for i, template_key in enumerate(template_keys):
        _template = template
        keys = template_key.split(".")

        # Navigate through nested structure
        for key in keys[:-1]:
            # Handle array indices like [0], [1]
            m = re.match(r'\[(\d+)\]', key)
            if m:
                key = int(m.group(1))
            _template = _template[key]

        # Set the final value
        _template[keys[-1]] = inputs[i]

    return template
```

**Template key parsing example:**

For `template_keys: ["inputs.[0].data", "inputs.[1].data", "inputs.[1].shape"]`:

| Template Key | Navigation Path | Final Assignment |
|-------------|-----------------|------------------|
| `inputs.[0].data` | template["inputs"][0]["data"] | = inputs[0] (query) |
| `inputs.[1].data` | template["inputs"][1]["data"] | = inputs[1] (documents) |
| `inputs.[1].shape` | template["inputs"][1]["shape"] | = inputs[2] (count) |

### Step 6: Request Execution Flow

```python
# endpoint_user.py lines 50-68
def __inference_request(self, request_meta):
    # 1. Get next prompt from generator
    inputs = next(self.text_input_generator)

    # 2. Load template from environment
    template = json.loads(os.getenv('TEMPLATE', "{}"))
    template_keys = json.loads(os.getenv('TEMPLATE_KEYS', "[]"))

    # 3. Optionally prepend model name if required
    if "model" in template_keys:
        inputs.insert(0, os.getenv("MODEL", ""))

    # 4. Fill template with generated values
    data = self._fill_template(template, template_keys, inputs)

    # 5. Send HTTP POST request
    body = json.dumps(data).encode("utf-8")
    response = requests.post(self.url, data=body, headers=headers)
```

### Step 7: Iterator Exhaustion Handling

When the dataset is fully consumed, the generator reinitializes:

```python
# endpoint_user.py lines 83-86
try:
    self.__inference_request(request_meta)
except StopIteration as se:
    # Dataset exhausted - reinitialize for continuous testing
    self.__init_prompt_generator()
    request_meta["exception"] = se
```

### Complete Data Flow Example

**Scenario:** Testing a reranker model with MS-MARCO data

```
1. Config loaded:
   module_dir: "modules/ms-marco"
   module_name: "reranking_pairs"
   prompt_generator: "RerankerInputGenerator"

2. Module imported:
   from modules.ms-marco.reranking_pairs import RerankerInputGenerator

3. Generator created:
   generator = RerankerInputGenerator()()

4. Prompt generated (one iteration):
   inputs = next(generator)
   # inputs = [
   #   "what is machine learning",           # query
   #   ["ML is...", "Deep learning...", ...], # 5 candidate documents
   #   [5]                                    # document count
   # ]

5. Template filled:
   {
     "inputs": [
       {"name": "query", "shape": [1], "datatype": "BYTES",
        "data": ["what is machine learning"]},
       {"name": "texts", "shape": [5], "datatype": "BYTES",
        "data": ["ML is...", "Deep learning...", ...]}
     ]
   }

6. HTTP POST sent to inference endpoint
```

### Available Prompt Generators

| Module Path | Generator Class | Dataset | Use Case |
|-------------|-----------------|---------|----------|
| `ms-marco/reranking_pairs.py` | `RerankerInputGenerator` | MS-MARCO | Reranker models |
| `conll2003/token_classification_prompts.py` | `TokenClassificationInputGenerator` | CoNLL-2003 | NER models |
| `imdb/sequence_classification_prompts.py` | (Sequence classification) | IMDB | Sentiment analysis |
| `wikitext/masked_lm_prompts.py` | (Masked LM) | WikiText | BERT-style models |
| `squad-context/squad_context.py` | (Context extraction) | SQuAD | QA context |
| `sahil2801-codealpca20k/prompt_generator.py` | `PromptGenerator` | CodeAlpaca | Code generation |
| `thudm-longbench/prompts.py` | (Long context) | LongBench | Long-context LLMs |
| `ronneldan_tinystories/tiny_prompt_generator.py` | (Story generation) | TinyStories | Small LLMs |
| `nicholasKluge-toxic-text/llama_guard3_prompt_generator.py` | (Safety testing) | Toxic-Text | Content moderation |
| `inst-semeval2017/llama3_prompt_generator.py` | (Instruction following) | SemEval | Instruction models |

### Creating a Custom Prompt Generator

To add a new prompt generator:

```python
# modules/my-dataset/my_prompts.py
from datasets import load_dataset

class MyPromptGenerator:
    def __init__(self):
        # Load your dataset
        self.dataset = load_dataset('my_dataset', split='test')

    def __call__(self):
        """Generator that yields prompts one at a time."""
        for item in self.dataset:
            # Format your prompt
            prompt = self._format(item)
            # MUST yield a list (even for single values)
            yield [prompt]

    def _format(self, item):
        return f"Question: {item['question']}\nAnswer:"
```

Then create a config file:

```yaml
# config/encoder/my_type/triton_inference_server-python.yaml
endpoint_url: "http://localhost:8080/v2/models/model/infer"
module_dir: "modules/my-dataset"
module_name: "my_prompts"
prompt_generator: "MyPromptGenerator"
template:
  inputs:
    - name: "text_input"
      shape: [1]
      datatype: "BYTES"
      data: []
template_keys:
  - "inputs.[0].data"
```
