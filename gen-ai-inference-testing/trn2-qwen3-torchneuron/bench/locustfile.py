#!/usr/bin/env python3
"""Locust cross-validation harness for the Qwen3-on-trn2 SLO sweep.

This is the SECOND opinion, not the source of truth -- `vllm bench serve` is.
Its job is to catch mis-instrumentation: if median TTFT and output throughput
from the two harnesses disagree by more than ~10%, one of them is wrong and
the numbers are not publishable. Treat disagreement as a blocker, not as
something to average away.

TTFT is measured as the wall time to the first SSE chunk carrying content.
ITL is measured between successive content chunks. Both therefore require
streaming; a non-streaming request cannot produce an ITL at all.

Mirrors the repo's existing run_locust.sh contract (master/worker, --csv),
so it can be driven by:

  SCRIPT=bench/locustfile.py \
  ENDPOINT_NAME=http://localhost:8000 \
  WORKERS=4 USERS=8 RUN_TIME=3m SPAWN_RATE=1 \
  RESULTS_PREFIX=./results/locust_c8 \
  bash ../run_locust.sh

Environment:
  MODEL              served model name (default qwen3-32b)
  INPUT_TOKENS       approximate prompt length in tokens (default 4096)
  OUTPUT_TOKENS      max_tokens, with ignore_eos (default 256)

UNVERIFIED: not yet run against a live server.
"""

from __future__ import annotations

import json
import os
import time

from locust import HttpUser, constant, events, task

MODEL = os.environ.get("MODEL", "qwen3-32b")
INPUT_TOKENS = int(os.environ.get("INPUT_TOKENS", "4096"))
OUTPUT_TOKENS = int(os.environ.get("OUTPUT_TOKENS", "256"))

# ~0.75 tokens per word is the usual rough ratio; close enough for a
# cross-check, and the exact prompt is identical across runs either way.
_FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank "
           "while the morning light spreads across the valley. ")


def build_prompt(target_tokens: int) -> str:
    words_needed = int(target_tokens * 0.75)
    reps = max(1, words_needed // len(_FILLER.split()) + 1)
    text = (_FILLER * reps).split()
    return " ".join(text[:words_needed])


PROMPT = build_prompt(INPUT_TOKENS)

# Collected across all users; summarised at test stop.
_ttfts: list[float] = []
_itls: list[float] = []
_output_tokens: list[int] = []


class Qwen3StreamUser(HttpUser):
    wait_time = constant(0)

    @task
    def chat_stream(self) -> None:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
            "stream": True,
            # Match vllm bench serve's --ignore-eos so output length is
            # controlled rather than observed.
            "ignore_eos": True,
        }

        start = time.perf_counter()
        ttft: float | None = None
        last = start
        itls: list[float] = []
        chunks = 0

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True,
            name=f"chat in{INPUT_TOKENS} out{OUTPUT_TOKENS}",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if not delta.get("content"):
                        continue  # role-only preamble chunk carries no token
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    else:
                        itls.append(now - last)
                    last = now
                    chunks += 1
            except Exception as exc:  # noqa: BLE001 - report, don't crash the run
                resp.failure(f"stream error: {exc}")
                return

            if ttft is None:
                resp.failure("no content chunks received")
                return

            resp.success()

        _ttfts.append(ttft * 1000.0)
        _itls.extend(i * 1000.0 for i in itls)
        _output_tokens.append(chunks)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


@events.test_stop.add_listener
def report(environment, **_kwargs) -> None:
    """Print the metrics vllm bench serve reports, for direct comparison."""
    if not _ttfts:
        print("\n[locust] no successful streamed requests -- nothing to compare\n")
        return

    runtime = max(environment.runner.stats.last_request_timestamp or 0
                  - (environment.runner.stats.start_time or 0), 1e-9)
    total_out = sum(_output_tokens)
    summary = {
        "requests": len(_ttfts),
        "median_ttft_ms": round(_pct(_ttfts, 50), 2),
        "p90_ttft_ms": round(_pct(_ttfts, 90), 2),
        "p99_ttft_ms": round(_pct(_ttfts, 99), 2),
        "median_itl_ms": round(_pct(_itls, 50), 2),
        "p99_itl_ms": round(_pct(_itls, 99), 2),
        "mean_output_chunks": round(total_out / len(_output_tokens), 1),
        "output_throughput_est": round(total_out / runtime, 2),
    }
    print("\n=== locust cross-validation summary ===")
    for k, v in summary.items():
        print(f"  {k:24}: {v}")
    print("\nCompare median_ttft_ms and output_throughput against the matching")
    print("vllm bench serve cell. Divergence >10% means one harness is wrong.")
    print("NOTE: output_throughput_est counts SSE content chunks, not tokenizer")
    print("tokens. It is an approximation -- expect a small systematic offset")
    print("versus vllm bench serve, which counts real tokens.\n")

    out = os.environ.get("LOCUST_SUMMARY_JSON")
    if out:
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {out}\n")
