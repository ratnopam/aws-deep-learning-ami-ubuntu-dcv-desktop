# Qwen3 on Trainium2 — vLLM + native TorchNeuron

Serve Qwen3 models on a single `trn2.48xlarge` through vLLM running on the **native TorchNeuron**
backend, and derive defensible SLOs (TTFT, ITL, TPOT, throughput).

> **Status: plan + unverified scaffolding.** Nothing here has been executed on hardware yet.
> Read [`PLAN.md`](./PLAN.md) first — it carries the full step-by-step plan, the pinned versions,
> the risk register, and open items O-1..O-6 that need resolving on first run.

## Why this is separate from `containers/openai_server-vllm-neuron`

The existing container is the **legacy** path: vLLM 0.7.2 patched, `torch-neuronx` + XLA
(`PJRT_DEVICE=NEURON`), NxD Inference for model implementations, Neuron SDK 2.28, trn1/inf2 era.

This is the **forward** path: vLLM 0.29.0 via the `vllm-neuron` plugin (no patches), TorchNeuron
native compilation (`VLLM_NEURON_BACKEND=neuron_native`), model implementations inside the plugin
(NxD Inference was dropped as a dependency at 0.21.0), Neuron SDK 2.33.0, trn2/trn3 only.

The old patch files do not apply and the compilation model is different, so this is a new example
rather than an upgrade. The legacy path is left untouched and working.

## Stack (Neuron SDK 2.33.0, 2026-10-19)

| Component | Version |
|---|---|
| TorchNeuron (Beta) | `2.14.3.0` |
| vLLM Neuron plugin (Beta) | `0.29.0.1.2.0` (vLLM `0.29.0`) |
| PyTorch | `2.14` (CPU wheel) |
| `neuronx-cc` | `2.28.7631.0` |
| Neuron Runtime | `2.35.30.0` |
| Python / OS in DLC | `3.14` / Ubuntu 26.04 |

Both headline components are **Beta** — pin exactly and record `pip freeze` with every result.

## Layout

```
containers/
  Dockerfile              primary: FROM the AWS Neuron vLLM DLC  <- use this
  Dockerfile.from-source  reference: full stack from source      <- build off-block
serve/
  download_model.sh       pinned checkpoint download + sha256 manifest
  precompile.sh           warm the compile cache, then exit
  run_qwen3_32b.sh        the published Trn2 BF16 recipe
bench/
  run_vllm_bench.sh       `vllm bench serve` sweep (source of truth)
  locustfile.py           streaming cross-validation harness
  parse_results.py        JSON -> slo_table.csv, slo_knee.md, slo_curves.png
notebooks/
  qwen3_32b_trn2_vllm.ipynb   runnable end to end over DCV
PLAN.md                   the detailed plan
```

## Quick start on a trn2.48xlarge

```bash
cd gen-ai-inference-testing/trn2-qwen3-torchneuron

docker buildx build -t openai_server-vllm_native:neuron -f containers/Dockerfile .

MODEL_ID=Qwen/Qwen3-32B DEST=$HOME/snapshots/Qwen3-32B bash serve/download_model.sh

# Precompile FIRST. Skipping this means benchmarking the compiler.
docker run --rm --privileged -v /dev:/dev \
  -v "$HOME/snapshots:/snapshots" -v "$HOME/cache:/cache" \
  -e MODEL_PATH=/snapshots/Qwen3-32B --shm-size 32g \
  openai_server-vllm_native:neuron bash /opt/program/serve/precompile.sh

docker run -d --name qwen3-trn2 --privileged -v /dev:/dev \
  -v "$HOME/snapshots:/snapshots" -v "$HOME/cache:/cache" \
  -e MODEL_PATH=/snapshots/Qwen3-32B --shm-size 32g -p 8000:8000 \
  openai_server-vllm_native:neuron bash /opt/program/serve/run_qwen3_32b.sh

bash bench/run_vllm_bench.sh
python3 bench/parse_results.py --results-dir ./results --hourly-cost <effective-$/hr>
```

Or just run the notebook, which does all of the above with the validation steps included.

## Three things that will bite you

1. **Precompile, always.** First run traces on CPU and compiles to NEFF (docs default the timeout
   to 1200s). The cache fingerprint invalidates on *any* version, compiler-flag or config change,
   so freeze your sweep matrix, then precompile once per config. On a time-boxed Capacity Block
   this is the difference between collecting data and watching a compiler.
2. **`--ignore-eos` is mandatory in benchmarks.** Without it, output length depends on the model's
   stopping behaviour and ITL comparisons across cells become meaningless.
3. **BF16 only on trn2.** The Qwen3-32B recipe advertises MXFP8, but that requires **Trn3**.

Also: do not install `torch-neuronx` into the serving image. The plugin uses
`libtorch-neuronx-lite`, which vendors the TorchNeuron compile components — the two together are a
conflict risk. Standalone TorchNeuron work gets its own venv.

## Capacity math

`trn2.48xlarge` = 16 Trainium2 devices x 8 NeuronCore-v3 = **128 physical cores**, 1,536 GiB HBM.
At `NEURON_LOGICAL_NC_CONFIG=2` that is **64 logical cores**.

Qwen3-32B at TP8/LNC2 uses 8 logical cores = **2 of 16 devices**, so **8 independent replicas fit
on one instance**. That makes a single box a real throughput-scaling testbed, and it is the number
that matters for EKS bin-packing later.

## Model coverage

| Model | Status | Notes |
|---|---|---|
| Qwen3-32B Dense | planned first | published Trn2 recipe, the control |
| Qwen3-VL-32B | planned | report text vs image TTFT separately |
| Qwen3-Embedding-8B | planned | TTFT/ITL do not apply; latency + embeddings/sec instead |

## References

- TorchNeuron — <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/index.html>
- vLLM Neuron plugin — <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html>
- Qwen3-32B tutorial — <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/tutorials/tutorial-qwen3-32b.html>
- Precompile cache — <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/how-to-precompile-cache.html>
- Plugin source — <https://github.com/vllm-project/vllm-neuron> (`release-0.29.0.1.2.0`)
