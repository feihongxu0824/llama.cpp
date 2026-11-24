# Embedding logits PoC

This fork-only experiment targets commit `5f55650a78f92aff4d48d671423e888fac0469ff`, the native revision used by the existing SDK. It is not a proposed production API or a patch for current upstream master. Upstream already merged the corresponding optimizations in [#30255](https://github.com/ggml-org/llama.cpp/pull/30255) and [#30445](https://github.com/ggml-org/llama.cpp/pull/30445).

The native change is confined to four files. `POC_LOGITS_MODE` is read once when each context is created:

| Mode | Host logits buffer/copy | Qwen LM head |
| --- | --- | --- |
| `baseline` (default) | Original behavior | Original behavior |
| `host_off` | Omitted | Still computed |
| `full_off` | Omitted | Omitted |

EmbeddingGemma has no LM head, so both experimental modes leave its graph unchanged. Batch output flags, pooling and the public C ABI stay unchanged. Experimental modes require Gemma embedding or Qwen3, embeddings enabled, MEAN/LAST pooling, a default context and no backend samplers. The mode cannot be changed during the context lifetime. This deliberately narrow environment switch is for comparing fresh processes, not for general application integration.

`POC_LOGITS_MEMORY` logs the actual host output-buffer capacity when it grows. This includes embeddings and is separate from the backend's compute-buffer capacity. The host-only mode also removes the output copy; the difference between `host_off` and `full_off` isolates removing the LM-head graph.

## Reproduce on the test Mac mini

Run model inference on the Mac mini or T4 test host, not on the development computer. The recorded experiment used Mac mini M4 / Metal only. Use the same GGUF file for every mode; model hashes and measured results are in [RESULTS.md](RESULTS.md).

From the repository root on the test host:

```sh
cmake -S examples/embedding-logits-poc -B build-logits-poc \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF
cmake --build build-logits-poc --target llama-logits-poc --parallel 4

for mode in baseline host_off full_off; do
  POC_LOGITS_MODE="$mode" build-logits-poc/llama-logits-poc \
    --model /path/to/Qwen3-Embedding-0.6B-Q8_0.gguf --device metal \
    --tokens 256 --contexts 1 --warmup 3 --iterations 20 --output "$mode.json"
done
python3 examples/embedding-logits-poc/compare.py baseline.json host_off.json full_off.json
```

Keep backend/library override environment variables unset, including `GGML_BACKEND_PATH`, `POC_BACKEND_DIR` and `DYLD_LIBRARY_PATH`. The optional prebuilt-ggml CMake path is not used for the recorded Metal measurements.

For Gemma, compare `baseline` and `host_off` using `embeddinggemma-300M-Q8_0.gguf`. Repeat with `--contexts 4` or `8` for independent contexts sharing one model. For correctness and reuse, use `--lengths 32,2048,32` for Gemma or `--lengths 32,8192,32` for Qwen, with `--warmup 0 --iterations 1`. These cold long-input runs are not steady-state throughput measurements.

The C++ harness constructs exact native-token lengths without truncating special tokens, sets every batch output flag, and records raw sequence embeddings. It uses persistent worker threads and one CPU/helper thread per context. Timings include KV clearing, batch setup, decode, synchronization and copying the result, but exclude loading, context creation and JSON output. Memory is sampled every 20 ms and at lifecycle boundaries. Mac physical footprint and RSS are reported separately.

The comparison script requires identical input token IDs and checks float32 bits, including warmup outputs. A mismatch is an error, not silently accepted with a tolerance. The JSON contains the vectors for further investigation.
