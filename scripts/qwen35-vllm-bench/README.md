# Qwen3.5-4B long-trace vLLM benchmark

This kit starts a text-only Qwen3.5-4B vLLM server on one 80 GB A100 and measures
aggregate output throughput, TTFT, TPOT, end-to-end time, completion length, GPU
utilization, memory, and power. It streams and discards generated text, so a
65,536-token response does not accumulate in client memory.

## Recommended starting point

The default server configuration is:

| Setting | Initial value | Why |
| --- | ---: | --- |
| `MAX_MODEL_LEN` | 131072 | Covers a 65,536-token output plus prompt, while retaining Qwen's recommended 128K-class context cap. |
| `MAX_NUM_SEQS` | 24 | Conservative starting point for BF16 KV cache at 65K tokens per live sequence. |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | vLLM recommends values above 8192 for small models on large GPUs; this workload is decode-heavy. |
| `GPU_MEMORY_UTILIZATION` | 0.95 | Uses the dedicated 80 GB GPU aggressively while leaving some runtime headroom. |
| `DTYPE` | bfloat16 | Native, numerically robust A100 format. |
| `KV_CACHE_DTYPE` | auto (BF16) | Quality/control baseline; compare with FP8 KV separately. |
| text-only | enabled | Does not load/profile the vision path; leaves more memory for cache. |
| chunked prefill | enabled | Prevents a long prefill from monopolizing a scheduler step. |
| async scheduling | enabled | Reduces CPU scheduling gaps. |
| MTP speculation | disabled | Start without it: current vLLM guidance says MTP can lower saturated throughput even when it improves low-concurrency TPOT. |
| prefix caching | disabled | Unique short prompts do not benefit; test it only if requests share a material prefix. |

The model's hybrid layout has only eight full-attention layers, but an approximate
BF16 attention-KV upper bound is still about 32 KiB/token/sequence, or about 2 GiB
for a 65,536-token sequence. vLLM's hybrid cache manager and recurrent states make
the actual allocation model-specific, so use the server's startup cache-capacity
line and the full-length confirmation test as the authority.

## Install

Use a recent NVIDIA driver/CUDA-compatible host. The installer defaults to
`vLLM 0.19.1`, Transformers `4.57.6`, and the CUDA 12.9 PyTorch backend. This
combination supports Qwen3.5 and avoids both a current CUDA-wheel regression and
the incompatible Transformers 5.x config types that can remain after replacing
an unpinned nightly installation.

Triton compiles a small CUDA driver helper locally, so the host also needs GCC
and the development headers matching the Python interpreter. On Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-dev build-essential
```

```bash
cd qwen35-vllm-bench
./install.sh
source .venv/bin/activate
```

The final import check must print `vLLM import: OK`. To try a newer release or
nightly after that packaging issue is fixed:

```bash
VLLM_VERSION=0.25.1 ./install.sh
VLLM_CHANNEL=nightly ./install.sh
```

### Repair the `libcudart.so.13` installation error

If an earlier run installed the faulty nightly into `.venv`, simply rerun the
updated installer. The exact version constraint will replace it:

```bash
cd qwen35-vllm-bench
./install.sh
source .venv/bin/activate
python -c 'import torch, transformers, vllm; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__)'
```

Do not create a `libcudart.so.13` symlink pointing to a CUDA 12 library; they are
different ABIs. If the pinned installation still fails, collect the relevant
driver, wheel, and shared-library information with:

```bash
./diagnose.sh | tee diagnose.log
```

## Start the server

Foreground mode traps Ctrl+C and shuts down the API server and engine workers:

```bash
./serve.sh
```

For detached operation, the script manages the PID and log itself:

```bash
./serve.sh --background
./serve.sh --status
./serve.sh --logs
./serve.sh --stop
```

`--stop` sends SIGTERM to the supervisor, which forwards it to the entire vLLM
process group and waits up to 30 seconds before using SIGKILL. Override the
grace period with `VLLM_STOP_TIMEOUT`. The default state files are `server.pid`
and `server.log`; override them with `VLLM_PID_FILE` and `VLLM_LOG_FILE`.

The key startup log lines report available KV cache and maximum concurrency. If
startup OOMs, first lower `GPU_MEMORY_UTILIZATION` to `0.93`; if runtime
preemptions occur, lower `MAX_NUM_SEQS`. Do not use CPU weight/KV offload for a
4B model on this GPU when throughput is the objective.

Every important setting is an environment variable:

```bash
MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=8192 ./serve.sh
MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=8192 ./serve.sh --background
```

## Run one exact-length test

In a second terminal:

```bash
source .venv/bin/activate
python benchmark.py \
  --concurrency 16 \
  --num-requests 16 \
  --output-tokens 65536 \
  --result results/c16-o65536.json
```

During generation, the client prints completed/active requests, approximate
output-token progress, live aggregate tokens/s, elapsed time, and ETA every five
seconds. The approximation is based on non-empty streaming deltas and is replaced
with the server's exact usage count as each request finishes. Change the cadence
with `--progress-interval-s 10`, or disable it with `--progress-interval-s 0`.

The same cadence is checkpointed as newline-delimited JSON in
`results/c16-o65536.progress.jsonl` (derived from the `--result` filename). Each
line is flushed immediately and contains cumulative and interval throughput,
completed-request TTFT/TPOT/end-to-end percentiles, exact and estimated token
counts, ETA, GPU utilization/memory/power, and heuristic `signals`. This makes
the measurements collected before an interruption usable. Follow it live with:

```bash
tail -f results/c16-o65536.progress.jsonl | jq .
```

Choose another path with `--progress-log PATH`, or suppress file logging with
`--disable-progress-log`. A zero progress interval suppresses periodic records
but still writes the final snapshot.

`--ignore-eos` is on by default so every request generates exactly the target
length. This is the correct mode for a controlled throughput test, but it can
force generation beyond the model's natural answer. To measure natural,
quality-representative traces instead:

```bash
python benchmark.py --concurrency 16 --output-tokens 65536 --no-ignore-eos
```

For your own task distribution, use `--prompt-file prompts/example.txt`. One
prompt repeated at several random seeds isolates serving throughput. A real
production test should also sample your actual prompt-length distribution.

## Sweep efficiently

Do a short screening sweep first:

```bash
python sweep.py --config sweep.example.json --output-tokens 4096
```

This restarts vLLM for each enabled server configuration and writes
`results/sweep/summary.csv`. It checkpoints that CSV after every run, writes
run-level snapshots beside each result as `*.progress.jsonl`, and records sweep
lifecycle events in `results/sweep/sweep-progress.jsonl`. Override the latter
with `--sweep-log PATH`. Rank on `output_throughput_tok_s`, but reject runs
with errors, preemptions in `server.log`, unexpectedly low GPU utilization, or
failure to hit the requested token length.

The sweep reports server-startup wait time, configuration and run position,
per-run duration, the benchmark's live token progress, and an approximate ETA
based on completed runs.

The live `signals` are intentionally conservative indicators, not automatic
verdicts: `gpu_underutilized` suggests testing more concurrency or checking a
host-side bottleneck; `gpu_memory_near_capacity` warns about little allocation
headroom; `recent_throughput_below_run_average` can indicate late-run pressure;
and `high_tpot_tail` marks uneven completed-request decode latency. Confirm any
signal against `server.log` and repeated runs before changing the configuration.

Then confirm only the best two or three configurations at full length. Examples:

```bash
python sweep.py --config sweep.example.json \
  --config-name bf16-b16384-s24 \
  --output-tokens 65536 \
  --result-dir results/full-bf16

python sweep.py --config sweep.example.json \
  --config-name fp8kv-b16384-s40 \
  --output-tokens 65536 \
  --result-dir results/full-fp8kv
```

A short sweep cannot expose late-run KV pressure; the 65K confirmation is
mandatory. For full-length runs, edit `concurrencies` to a narrow band around
the short-sweep winner (for example `[12, 16, 20, 24]`) to avoid days of testing.

## Tuning order

1. **Concurrency:** Find the knee of aggregate output tokens/s. Start with
   1, 2, 4, 8, 12, 16, 20, and 24 for BF16 KV. Stop increasing when throughput
   flattens, TPOT grows sharply, or preemption appears.
2. **KV precision:** Compare `auto` with `fp8_e4m3`. FP8 doubles cache capacity,
   but A100 lacks native FP8 tensor cores, so conversion overhead and any quality
   impact must be measured. Use the same prompts/seeds and validate task quality.
3. **Token budget:** Compare 8192, 16384, and 32768. This mostly changes prefill
   behavior; long-generation steady-state throughput may move little.
4. **GPU memory fraction:** Once stable, try 0.93, 0.95, and 0.97. A higher value
   increases cache capacity but can cause startup or transient OOMs.
5. **Speculation:** Test MTP only after finding the non-speculative optimum. It is
   most likely to help low concurrency/latency; enable the disabled example and
   verify the method name against `vllm serve --help` for the installed nightly.
6. **Prefix caching:** Enable only for a shared, nontrivial prefix. It saves
   prefill work, not autoregressive decode work, and hybrid-model prefix caching
   is still called experimental in current guidance.

Use aggregate `output_throughput_tok_s` as the primary throughput objective.
`1 / TPOT` is a per-request latency view and is not aggregate server throughput.
Also inspect the whole server log for `preempt`, `recompute`, `OOM`, and graph
capture warnings before accepting a configuration.

## Notes on a 65K “reasoning trace”

The context limit is input plus output. Keep the tokenized prompt below about
65K with the 131072 cap if requesting 65536 new tokens. The default prompt is a
synthetic benchmark. For coherent traces, do not assume that forcing 65K tokens
with `ignore_eos` preserves answer quality; run both exact-length capacity tests
and natural-EOS quality tests.

Sampling defaults match Qwen's published general thinking recommendation:
temperature 1.0, top-p 0.95, top-k 20, min-p 0, and presence penalty 1.5.
