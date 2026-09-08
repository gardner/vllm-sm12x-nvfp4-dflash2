# vLLM with NVFP4 and DFlash2 for Blackwell GPUs

This repository builds **vLLM v0.28.0** to serve Qwen3.8-27B on a single
Blackwell GPU. The target model, DFlash2 draft model, and KV cache use
NVFP4. An optional CPU vision sidecar converts images into embeddings for
the vLLM server.

The v0.28.0 update is a source-build candidate. Patch application and CPU
regression checks pass; the Docker build, GPU serving, and performance
still need validation. The published RTX 5090 results in this repository
apply to the previous v0.27.1 release. No v0.28.0 runtime image has been
published by this project.

## Why a custom build is still needed

The [official v0.28.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)
already includes DFlash2. We use that implementation and maintain two
patches for the remaining requirements:

- **SM12x NVFP4 support:** attention dispatch and cache scales for RTX 50-series
  and GB10 GPUs, quantized draft KV projection, and draft execution fixes.
- **Fused M-RoPE:** an upstream fix that keeps the fused decode path available
  with the pinned model's multimodal configuration. It is absent from the
  v0.28.0 release tree.

The patches now change nine production files, down from 53 across the old
main and vision patches. One change is in the native CUDA cache writer,
so the build recompiles vLLM using its official Dockerfile. The build uses
the official FlashInfer dependency without a separate FlashInfer patch.
See [patches/README.md](patches/README.md) for each change, its upstream
status, and when it can be removed.

## Getting started

Use Linux or WSL2 with an RTX 5090, a working NVIDIA driver,
[Docker Engine and Compose](https://docs.docker.com/engine/install/), and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The scripts also need Git, `sha256sum`, `curl`, and Python 3. Building vLLM
requires substantial CPU memory and disk space; allow additional space for
the build cache beyond the roughly 22 GB of model downloads.

```bash
git clone https://github.com/gardner/vllm-sm12x-nvfp4-dflash2.git
cd vllm-sm12x-nvfp4-dflash2
cp .env.example .env
./build.sh
./start.sh
```

`build.sh` produces `vllm-sm12x-nvfp4-dflash2:local-v0280-dflash2-1`, matching
`.env.example`. It checks artifact hashes, clones the pinned upstream tag,
verifies its commit, applies both patches, and builds the non-root runtime.
Set `MAX_JOBS` and `NVCC_THREADS` to control build concurrency; their defaults
are 3 and 2.

On a host with multiple GPUs, set `GPU_DEVICE` in `.env` to the Blackwell
GPU's index or UUID before starting. The default is `0`.

`start.sh` creates `.env` if needed, checks the GPU and Docker setup, verifies
that the image matches this checkout, downloads the pinned checkpoints into
Docker volumes, and starts vLLM. Once healthy, it runs a chat completion
smoke test that checks whether `19 × 23` returns `437`.

To include the CPU vision sidecar:

```bash
./start.sh --vision
```

The sidecar uses an INT8-quantized vision tower by default, with limits of
eight CPUs and 6 GB of memory. Set `SIDECAR_INT8=0` for its FP32 fallback.

| Service | URL |
|---|---|
| OpenAI-compatible API | `http://127.0.0.1:18089/v1` |
| Optional vision proxy | `http://127.0.0.1:8016/v1` |

Served model name: `qwen3.8-27b-nvfp4-dflash2`.

## Upgrading from v0.27.1

An existing `.env` is preserved. Update its `IMAGE` setting to
`vllm-sm12x-nvfp4-dflash2:local-v0280-dflash2-1` and remove `DYNAMIC_SCHEDULE`
if present. The new profile uses fixed K7 and no longer supports the custom
dynamic-K schedule.

```bash
git pull --ff-only
./stop.sh
./build.sh
./start.sh --vision        # Omit --vision if the sidecar is not needed
./verify.sh --full
./verify.sh --vision       # Requires the running sidecar
```

Keep your GPU selection, ports, and model settings. Model revisions are
unchanged, so their downloaded volumes can be reused. Startup rejects an
old runtime image with the new configuration. The old metadata and vision
layer Dockerfiles have been removed; both patches are included in the full
build, and `VISION_MROPE=1` is no longer needed.

## Pinned components

Source pins are in [release.env](release.env); model revisions and serving
settings are in [.env.example](.env.example).

| Component | Version or artifact |
|---|---|
| vLLM | [v0.28.0, commit `2cf0a6915`](https://github.com/vllm-project/vllm/commit/2cf0a6915ce544dc493a0990f2ea38d81601128a) |
| CUDA | 13.0.3 |
| FlashInfer | 0.6.16.post3, installed by the official vLLM Dockerfile |
| Target | [`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090/tree/69274a0d8dff5dd35bcee8290612f71e03b6e981), revision `69274a0` |
| Draft | [`YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4`](https://huggingface.co/YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4/tree/d913b0b5603a67c26f3edaf7e42a9f8cf89886be), revision `d913b0b` |
| Patch order | [patches/series](patches/series) |
| Artifact checksums | [SHA256SUMS](SHA256SUMS) |
| Chat template | [chat-template.jinja](chat-template.jinja), mounted by Compose |

The locally built image tag is mutable. A published deployment should use
its registry digest. To build and push to a registry you control, set
`GHCR_OWNER` and `GHCR_PAT`, then run `./build.sh --push`.

## Default serving profile

These settings are carried forward from the previous RTX 5090 profile.
Available KV capacity and performance must be measured again on v0.28.0.

| Setting | Default |
|---|---|
| Speculative decoding | DFlash2, fixed K7: seven draft tokens and one anchor token |
| Target and draft KV cache | NVFP4 |
| Target KV allocation | 8 GiB (`8589934592` bytes) |
| GDN/SSM state | BF16 |
| Maximum context length | 262,144 tokens |
| Maximum concurrent sequences | 4 |
| Maximum batched tokens | 4,096 |
| Target CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes `[8,16,24,32]` |
| Draft execution | `enforce_eager: true` in the speculative config |
| Attention | FlashInfer; FA2 prefill and masked XQA decode on SM12x |
| XQA stream | `VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM=1` |
| Scheduling | Priority, prefix caching, and chunked prefill |
| Sampling | Temperature 0.6; thinking enabled with medium effort |
| Tool calling | `qwen3_coder` parser with automatic tool choice |
| Multimodal input | `--enable-mm-embeds`; image/video limits zero to exclude the GPU vision tower |

The KV pool is shared across active sequences. The maximum context setting
does not reserve that many tokens for each sequence. The prior release's
measured 325,139-token pool is historical, not a v0.28.0 capacity guarantee.

## Verification and operations

```bash
./scripts/check-release.sh # Checksums, configuration, and syntax; Compose if available
./status.sh                # Container, health, GPU, and cache status
./verify.sh --smoke        # Health, model routing, and arithmetic
./verify.sh --full         # Also long-decode determinism, retrieval, and tools
./verify.sh --vision       # Image fixtures and vision concurrency
./stop.sh                  # Remove containers; keep cache volumes
./stop.sh --purge-cache    # Also remove model, draft, and vLLM cache volumes
```

The CI workflow also applies the patches to a fresh upstream checkout and
runs CPU regressions. See the [patch validation instructions](patches/README.md#validation)
for local CPU tests and the native SM12x writer/attention gate. Image builds,
model loading, CUDA graph execution, and vision kernel correctness require
GPU validation before release promotion.

[BENCHMARKS.md](BENCHMARKS.md) and [EVIDENCE.md](EVIDENCE.md) retain the
v0.27.1 performance and validation records. There are no v0.28.0 benchmark
results yet.

## SM121: DGX Spark / GB10

Build natively on the Spark with:

```bash
SPARK=1 ./build.sh
```

This selects `linux/arm64` and `TORCH_CUDA_ARCH_LIST=12.1`. SM121 remains
unvalidated on native hardware. Validate both the kernels and full serving
profile on GB10 before using or publishing that image.

## Rollback and coexistence

For rollback, use the previous repository revision with its matching image
and `.env`. The new startup checks require matching source and image
provenance.

This project uses the Compose project `qwen38-dflash2`, ports 18089 and
8016, and its own model cache volumes. It can coexist on disk with the MTP
project, `vllm-sm120-nvfp4-mtp`. Run only one 27B GPU server at a time on a
single 32 GiB GPU.
