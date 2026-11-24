# Metal PoC results

On Apple M4 Metal, removing unused host logits reduced physical memory substantially. Removing Qwen's LM head also improved throughput by about 31%. These are results from the original isolated experiment on 2026-11-23, based on llama.cpp `5f55650a78f92aff4d48d671423e888fac0469ff`, not production validation of a new API.

Upstream already merged related work in [#30255](https://github.com/ggml-org/llama.cpp/pull/30255) and [#30445](https://github.com/ggml-org/llama.cpp/pull/30445). This fork preserves the older-snapshot PoC for review and reproduction; it is not a new upstream fix. Moving the harness into this example does not turn the historical measurements into measurements of a newly rebuilt binary.

## Setup and correctness

Mac mini M4, 16 GiB unified memory, Metal only. The native core, ggml and Metal backend were built from source. No zg or Rust integration was involved. One sequence per context, one thread per context, shared model weights, and unchanged batch output flags and pooling.

- `baseline`: original allocation and computation.
- `host_off`: omit host logits allocation and skip logits output copies. Qwen still computes its LM head.
- `full_off`: additionally omit Qwen's LM head. Gemma already has no LM head, so only the first two modes were measured for Gemma.

Q8_0 models, identified by SHA256:

- EmbeddingGemma 300M: `b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63`
- Qwen3-Embedding 0.6B: `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`

All 35 cases completed. All 1,345 recorded vectors had finite values and the expected dimensions. All 807 paired vectors across modes were float32 bitwise equal, with maximum absolute difference zero; token IDs, binary and native-library hashes also matched. The four baseline comparisons across passes were exact. All five short-long-short cases reproduced the initial short-input vector exactly after resizing and reusing the context. Post-run checks found the binary, five native libraries and both models unchanged.

## 256-token results

Each configuration used two passes in opposite mode order, three warmup rounds and 20 measured rounds per pass. Each round ran one job per context. Latencies exclude warmup and context creation. Reported p50/p95 are the median of the two per-pass job-latency quantiles; throughput is total measured jobs divided by total measured wall time. Physical footprint is the median of the two sampled process peaks, including loading and warmup, sampled every 20 ms. It is not RSS or a GPU allocation counter.

| Model | Contexts | Mode | Footprint MiB | p50 ms | p95 ms | Vectors/s |
|---|---:|---|---:|---:|---:|---:|
| Gemma | 1 | baseline | 374.47 | 27.87 | 28.13 | 36.51 |
| Gemma | 1 | host_off | 118.40 | 27.87 | 28.08 | 36.57 |
| Gemma | 4 | baseline | 1175.00 | 106.57 | 108.65 | 37.43 |
| Gemma | 4 | host_off | 150.61 | 106.63 | 108.40 | 37.45 |
| Qwen | 1 | baseline | 299.03 | 122.55 | 123.91 | 8.17 |
| Qwen | 1 | host_off | 150.30 | 118.78 | 120.48 | 8.42 |
| Qwen | 1 | full_off | 150.29 | 93.48 | 94.21 | 10.74 |
| Qwen | 4 | baseline | 863.13 | 477.03 | 485.26 | 8.32 |
| Qwen | 4 | host_off | 268.48 | 464.94 | 470.59 | 8.58 |
| Qwen | 4 | full_off | 268.31 | 366.87 | 371.31 | 10.88 |

Gemma throughput was essentially unchanged. Qwen `full_off` improved throughput over baseline by 31.5% at C=1 and 30.8% at C=4. These are small, synthetic two-pass comparisons; minor differences are not evidence of a stable speedup.

## More contexts and long inputs

The following are single-pass memory checks, not additional steady-state throughput measurements. C=8 used 256 tokens with three warmup rounds and one measured round. Long cases used one context and the sequence 32, maximum length, 32, without warmup.

| Case | Baseline footprint MiB | host_off MiB | full_off MiB |
|---|---:|---:|---:|
| Gemma, C=8, 256 tokens | 2242.14 | 192.74 | N/A |
| Qwen, C=8, 256 tokens | 1615.24 | 425.88 | 425.38 |
| Gemma, C=1, up to 2048 tokens | 2220.72 | 171.64 | N/A |
| Qwen, C=1, up to 8192 tokens | 5803.94 | 1061.52 | 1061.39 |

Native allocation logs confirmed the removed host logits capacity: 2048 MiB for Gemma C=8/256 or C=1/2048; 1184.914 MiB for Qwen C=8/256; and 4739.656 MiB for Qwen C=1/8192. These values follow actual output rows times vocabulary size times four bytes, not differences from another PR's benchmark.

For Qwen C=1, `full_off` reduced the nominal Metal compute-buffer capacity from 150.25 to 20.29 MiB. Physical footprint and RSS were almost unchanged between `host_off` and `full_off`. Do not add that approximately 130 MiB allocator reduction to the measured physical-memory savings.

These results cover Metal on this M4 only. They do not establish CPU or CUDA performance, production API compatibility, or zg integration behavior. Explicit context-release checks also do not validate zg's idle-time eviction policy.

## PR checkout verification, 2026-11-24

The four changed native files and the C++ harness match the original tested source byte for byte. The bundled comparison script rechecked all 807 captured vector pairs successfully. A fresh full-source Metal build of the relocated example also completed on the Mac mini.

Fresh execution validation did not complete: the new executable was terminated before inference, including when invoked with `--help`. System logs reported AMFI signature and XProtect loader rejection, despite on-disk signature verification passing. No security settings were changed. This rebuild therefore provides compilation evidence only; the measurements above remain the historical 2026-11-23 results, not a new benchmark of the PR checkout.
