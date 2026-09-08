# vLLM with NVFP4 and DFlash2 for Blackwell GPUs

This repository provides a community build of vLLM v0.27.1 for serving
Qwen3.8-27B on a single Blackwell GPU. It uses ModelOpt NVFP4 target weights,
NVFP4 DFlash2 draft weights, and an NVFP4 KV cache. The default configuration supports a
262,144-token context limit, up to four concurrent sequences, and tool
calling. An optional CPU vision sidecar converts images into embeddings
for the vLLM server.

The release is validated on an RTX 5090 (SM120, 32 GB). A build configuration
is also provided for DGX Spark / GB10 (SM121), but it has not been validated
on that hardware. This is a community project, independent of the official
vLLM and NVIDIA releases.

## Overview

- **NVFP4 weights and KV cache.** Both the target and draft models use
  NVFP4 weights and KV caches. GDN/SSM state uses BF16.
- **DFlash2 speculative decoding.** A 5-layer, 1.92B-parameter
  block-diffusion draft model proposes seven tokens per verification step
  (`K=7`). Each step uses eight target query tokens.
- **Context and concurrency.** The default profile allocates 8 GiB to the
  target KV cache, providing a measured pool of 325,139 tokens shared across
  active sequences. It enables prefix caching, priority scheduling, and
  chunked prefill.
- **Multimodal support.** Release `v0.27.1-sm12x-dflash2.3` adds support for
  Qwen3.5's three-axis M-RoPE to the fused QK-norm, RoPE, and gate Triton
  kernel. This keeps the fused decode path available when multimodal
  embeddings are enabled.
- **Optional CPU vision.** The sidecar uses an INT8-quantized ViT tower by
  default, with limits of eight CPUs and 6 GB of memory. Text serving does
  not require the sidecar.

## Getting started

You need Linux or WSL2, an RTX 5090, a working NVIDIA driver,
[Docker Engine with the Compose plugin](https://docs.docker.com/engine/install/),
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The startup script also requires `curl` and `python3`. For DGX Spark / GB10,
see the [SM121 build notes](#sm121-dgx-spark--gb10).

Allow roughly 30 GB of downloads: about 9 GB for the runtime image, 20.6 GB
for the target checkpoint, and 1.3 GB for the draft model. The startup script
warns if less than 45 GB of disk space is available.

```bash
git clone https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2.git
cd vllm-sm12x-nvfp4-dflash2
./start.sh
```

To include the CPU vision sidecar, use:

```bash
./start.sh --vision
```

On first use, `start.sh` creates `.env` from `.env.example`. It checks the
selected GPU, VRAM, Docker, and disk space; pulls the pinned runtime image;
downloads the pinned model checkpoints into Docker named volumes; and
starts vLLM. Once the server is healthy, it runs a chat completion smoke
test that checks whether `19 × 23` returns `437`.

On a host with multiple GPUs, create `.env` from `.env.example` before
starting and set `GPU_DEVICE` to the Blackwell GPU's index or UUID. The
default is `0`. The startup script checks that device, and Compose selects
it through `NVIDIA_VISIBLE_DEVICES` with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

After startup, use these endpoints and model name:

| Service | URL |
|---|---|
| OpenAI-compatible vLLM API | `http://127.0.0.1:18089/v1` |
| Vision proxy, when started with `--vision` | `http://127.0.0.1:8016/v1` |

Served model name: `qwen3.8-27b-nvfp4-dflash2`.

The first startup includes model downloads and CUDA/FlashInfer warmup.
Later starts reuse the downloaded image and model caches.

## Common operations

```bash
./status.sh                # Show containers, health, GPU, and model-cache status
./verify.sh --smoke        # Check health, model routing, and the arithmetic test
./verify.sh --full         # Also check long-decode determinism, retrieval, and tools
./verify.sh --vision       # Run the two-image fixture and vision concurrency checks
./stop.sh                  # Remove containers and keep cache volumes
./stop.sh --purge-cache    # Also remove model, draft, and vLLM cache volumes
```

Run the vision checks after starting the sidecar with `./start.sh --vision`.
The retrieval check in `--full` is a needle-in-a-haystack (NIAH) test.

## Pinned components

The default runtime image and model checkpoints use immutable digests or
revisions. Their full values are in [.env.example](.env.example).

| Component | Version or artifact |
|---|---|
| Runtime image | `ghcr.io/seanyourhighness/vllm-sm12x-nvfp4-dflash2`, pinned by digest in `.env.example` |
| Target model | [`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090/tree/69274a0d8dff5dd35bcee8290612f71e03b6e981), revision `69274a0` |
| Draft model | [`YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4`](https://huggingface.co/YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4), revision `d913b0b` |
| vLLM base | [v0.27.1, commit `6e448d0ea`](https://github.com/vllm-project/vllm/commit/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac) |
| FlashInfer | 0.6.16.post3, commit `9dc1b24`, with the [PR #4346](https://github.com/flashinfer-ai/flashinfer/pull/4346) backport for SM120 NVFP4 paged prefill |
| Main patch | [`0001-v0271-sm12x-dflash2-nvfp4.patch`](0001-v0271-sm12x-dflash2-nvfp4.patch), covering 51 Python files |
| Vision patch | [`0002-qwen3-next-fused-mrope-vision.patch`](0002-qwen3-next-fused-mrope-vision.patch), covering two production Python files and a targeted CUDA test |
| Vision image layer | [`Dockerfile.vision-mrope`](Dockerfile.vision-mrope), applied to the `.2` base image |
| Chat template | [`chat-template.jinja`](chat-template.jinja) |
| Artifact checksums | [`SHA256SUMS`](SHA256SUMS) |

Compose passes the pinned target revision to vLLM and loads the repository's
chat template with `--chat-template`. This replaces the template bundled
with the model. Model weights are downloaded separately from the runtime
image.

## Default runtime configuration

These settings were validated on the RTX 5090. The KV cache pool is shared:
four active sequences have roughly 81K cached tokens each if the pool is
divided equally.

| Setting | Default |
|---|---|
| Speculative decoding | `{"method":"dflash","model":"/models/draft","num_speculative_tokens":7,"kv_cache_dtype":"nvfp4"}` |
| Target KV cache | NVFP4; 8 GiB (`8589934592` bytes), with a measured 325,139-token pool |
| GDN/SSM state | `bfloat16` |
| Maximum context length | 262,144 tokens |
| Maximum concurrent sequences | 4 |
| Maximum batched tokens | 4,096 |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes `[8,16,24,32]` |
| Draft execution | Eager mode (`VLLM_DFLASH_FORCE_EAGER=1`) to avoid integrated XQA graph interference |
| Target XQA | Dedicated CUDA stream (`VLLM_XQA_DEDICATED_STREAM=1`) |
| FlashInfer autotuning | Enabled; cached results are loaded at startup |
| Scheduling | Priority scheduling, prefix caching, and chunked prefill; long-prefill threshold of 2,048 tokens |
| Sampling | Default temperature override of 0.6; thinking enabled with medium effort |
| Tool calling | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| Multimodal serving | `--enable-mm-embeds` with image and video limits set to zero; fused M-RoPE kernel |
| Triton JIT cache | `/home/vllm/.cache/vllm/triton`, persisted in the vLLM cache volume |

## Performance

The following results were measured on a single RTX 5090 using the published
`v0.27.1-sm12x-dflash2.3` image. The benchmark used three warmup runs and five
measured runs, with cache busting.

| Concurrent sequences | Narrative, total tokens/s | Code, total tokens/s |
|---:|---:|---:|
| 1 | ~98 | ~173 |
| 2 | ~192 | ~330 |
| 3 | ~278 | ~447 |
| 4 | ~344 | ~587 |

At four concurrent sequences, aggregate throughput was about 3.5 times the
single-sequence result. Draft acceptance was approximately 61% at `K=7`.
The reported runs had no restarts or out-of-memory errors.

See [BENCHMARKS.md](BENCHMARKS.md) for decode, prefill, and vision results
and reproduction instructions. [EVIDENCE.md](EVIDENCE.md) contains the
detailed measurement record.

## How the vision patch works

The CPU sidecar encodes images and sends their embeddings to vLLM. The GPU
server therefore runs with `--enable-mm-embeds`, even though the image and
video limits are zero to exclude the GPU vision tower.

In the source used for release `.2`, the fused QK-norm, RoPE, and gate
decoder kernel did not support Qwen3.5's three-axis M-RoPE. Enabling
multimodal embeddings selected a slower eager path, including for text
requests. The idle CPU sidecar was not the cause of the slowdown.

Release `.3` adds temporal, height, and width (T/H/W) position selection to
the existing Triton kernel, supporting both contiguous and interleaved
M-RoPE sections. Qwen3Next passes all three position axes to the kernel.
Triton compiles the new variants on first use and saves them in the cache;
the incremental image layer does not require a native CUDA rebuild.

In the RTX 5090 comparison, the older embedding-capable path produced
61.6 narrative and 105.7 code tokens/s. The language-only path produced
116.2 and 202.4 tokens/s, respectively. The `.3` patch restored multimodal
support with decode performance comparable to the language-only
configuration across one to four concurrent sequences. See
[EVIDENCE.md](EVIDENCE.md) for the matched measurements and the passing
SM120 CUDA and two-image fixture checks.

Related upstream work is documented in
[vLLM #49744](https://github.com/vllm-project/vllm/pull/49744) and
[vLLM #43056](https://github.com/vllm-project/vllm/pull/43056). This release
uses a smaller backport for its pinned vLLM version.

### Upgrading from `.2`

Update the repository and stop the existing containers:

```bash
git pull --ff-only
docker compose down --remove-orphans
```

Set `IMAGE` in your existing `.env` to the published digest in the updated
`.env.example`. The startup script preserves an existing `.env`, so pulling
the repository alone does not update that setting. Then start the server:

```bash
./start.sh
```

Use `./start.sh --vision` to include the sidecar, then run
`./verify.sh --vision` once it is healthy. The first multimodal request
compiles the new Triton kernel variants.

The `.3` image adds a small layer to the `.2` base and uses the same model
artifacts. To build that layer locally, use
[`Dockerfile.vision-mrope`](Dockerfile.vision-mrope) and set `IMAGE` to the
resulting local image.

## Release verification

Check the local artifacts against their recorded checksums, then run the
server checks:

```bash
sha256sum --check SHA256SUMS
./verify.sh --full
./verify.sh --vision        # Requires the running vision sidecar
```

The `release-integrity` workflow checks release integrity on every push.
Benchmark results and reproduction instructions are in
[BENCHMARKS.md](BENCHMARKS.md); validation details are in
[EVIDENCE.md](EVIDENCE.md).

## SM121: DGX Spark / GB10

The build script supports `linux/arm64` with `torch_cuda_arch_list=12.1`
using the same base source and main patch. Run it natively on the Spark:

```bash
SPARK=1 ./build.sh
```

SM121 has not been validated on native hardware. The published validation
results, including the `.3` vision measurements, apply to SM120. The
multi-architecture tag will be promoted after the SM121 build passes
greedy determinism, NIAH retrieval, tool calling, and a four-sequence soak
test on GB10 hardware.

## Rollback and coexistence

This project can be installed alongside the MTP release,
`vllm-sm120-nvfp4-mtp`. It uses a separate repository, image, Compose project
(`qwen38-dflash2`), container names, and model cache volumes. Its default
ports are 18089 and 8016; the MTP release uses 18079 and 8006.

Run only one 27B GPU server at a time on a single 32 GiB GPU. To switch back
to MTP, run `./stop.sh`, then start the pinned MTP project.
