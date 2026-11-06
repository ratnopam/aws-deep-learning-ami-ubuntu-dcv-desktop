# Primary image: layer on the AWS Neuron vLLM DLC.
# Low-risk path -- the DLC already bundles Neuron SDK 2.33.0, the vllm-neuron
# plugin, neuronx-cc and libtorch-neuronx-lite at mutually compatible versions.
#
# UNVERIFIED: not yet built or run on hardware.
#
# Build:
#   docker buildx build -t openai_server-vllm_native:neuron \
#     -f containers/Dockerfile .

ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
    # NOTE: deliberately NOT setting PJRT_DEVICE=NEURON. That is the legacy
    # torch-neuronx/XLA path; the native backend is selected by
    # VLLM_NEURON_BACKEND=neuron_native at runtime instead.

# Default to the native TorchNeuron compile path. Must be set in the
# environment before Python starts -- setting it in-process has no effect.
ENV VLLM_NEURON_BACKEND=neuron_native \
    VLLM_CACHE_ROOT=/cache \
    VLLM_NEURON_COMPILATION_TIMEOUT=1200 \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
    NEURON_LOGICAL_NC_CONFIG=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        jq \
        numactl \
    && rm -rf /var/lib/apt/lists/*

# Benchmark-side deps. The DLC ships vLLM (which provides `vllm bench serve`),
# so only the cross-validation harness and parsing need adding.
RUN python -m pip install --no-cache-dir \
        "locust>=2.32" \
        "pandas>=2.2" \
        "matplotlib>=3.9"

COPY serve /opt/program/serve
COPY bench /opt/program/bench
RUN chmod -R u+x /opt/program/serve /opt/program/bench

WORKDIR /opt/program

# Record what actually got installed -- both key components are Beta, so the
# exact resolved versions are part of any benchmark result.
RUN python -m pip freeze > /opt/program/pip-freeze.txt

EXPOSE 8000
CMD ["/bin/bash"]
