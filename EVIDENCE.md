# Evidence — all-NVFP4 DFlash2 release (RTX 5090 / SM120)

> Historical v0.27.1 results. The current v0.28.0 source build has not yet
> been built or benchmarked on a GPU. These measurements do not validate it.

## v0.28.0 source checks (2026-09-08)

The maintained patches were applied to a fresh clone of upstream v0.28.0,
commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`.

- `CHECK_PATCH_APPLY=1 ./scripts/check-release.sh`: passed artifact hashes,
  shell syntax, default configuration checks, patch application, and Python
  compilation. All 27 CPU source regressions passed with Python 3.12.
- Both YAML files parsed. Docker Compose rendering was skipped because the
  Docker CLI is unavailable in this environment.
- SM120 and SM121 build scripts passed a command-level check with Docker
  mocked and a real local upstream clone. The check verified every build
  argument against the upstream Dockerfile, both patches, the non-root
  stage, image labels, and rejection of stale images at startup.
- No Docker image was built or published. CUDA kernels, model loading,
  serving correctness, and performance have not been validated on this
  update. The new GPU gate is `tests/test_sm12x_nvfp4.py`.

The remaining sections record the previous v0.27.1 release.

Compiled 2026-08-25 from the validated production runs. Source of record:
GBrain `vllm/2026-08-21-qwen38-dflash2-nvfp4-release-plan`,
`sessions/2026-08-24-dflash2-capacity-k7-production-promotion`,
`sessions/2026-08-24-dflash2-drafter-parity-radixark`,
`bench/2026-08-25-coding-benchmark-c1-vs-c4-dflash2-k7`.

## `.3` fused M-RoPE candidate (SM120)

The `.2` regression was reproduced with the CPU sidecar idle: enabling the
multimodal/embedding-capable GPU path disabled the fused QK-norm + RoPE + gate
decoder path. The incremental `.3` patch extends that existing Triton kernel
for Qwen3.5's T/H/W M-RoPE (contiguous and interleaved sections), keeping the
base image and all native CUDA artifacts unchanged. The minimal Docker layer
contains the two production Python files and is recorded as 12,600 bytes;
Triton JIT compiles the new variants on first use into the persisted cache.

The reviewed CUDA correctness matrix passed **9/9** cases. The exact vision
fixture passed **2/2** images (both color/shape/label checks). These are SM120
(RTX 5090) results; SM121 (DGX Spark/GB10) remains unvalidated.

### Matched decode matrix

Narrative/code tok/s, measured with the same DFlash2 K7 profile and workload
shape. The candidate keeps the multimodal server configuration active; the
control uses `.2`'s `--language-model-only` path.

| Concurrency | `.3` candidate narrative / code | `.2` language-only narrative / code |
|---:|---:|---:|
| c1 | 107.72 / 206.63 | 106.26 / 192.76 |
| c2 | 215.87 / 352.09 | 213.30 / 370.32 |
| c3 | 307.44 / 453.91 | 316.07 / 455.56 |
| c4 | 403.77 / 616.33 | 405.54 / 592.75 |

The prior multimodal control was 61.6 narrative / 105.7 code tok/s, versus
116.2 / 202.4 in language-only mode. The candidate removes that roughly
50-percent decode cliff without requiring a native rebuild.

### Upstream duplicate-work check

The implementation area is already covered by open upstream efforts
[vLLM #49744](https://github.com/vllm-project/vllm/pull/49744) and
[vLLM #43056](https://github.com/vllm-project/vllm/pull/43056). The release
keeps this as a narrow pinned-vLLM backport and does not open a duplicate PR.

## What makes this build unique

No public container, repo, or blog ships a working **NVFP4 weights + NVFP4
draft + NVFP4 KV + DFlash2 + concurrency** stack on a 5090. The universal
pattern is NVFP4 weights + FP8 KV; the DFlash2 draft is usually BF16 or
W8A16. This release is the all-NVFP4 4/4 artifact, with the drafter
itself NVFP4 (W4A16, group-16, compressed-tensors `nvfp4-pack-quantized`).

## Overlay contents (51 files, Python-only, +2826/−217 vs v0.27.1)

| Area | Files | What it does |
|---|---|---|
| DFlash2 core | `vllm/model_executor/models/qwen3_dflash2.py`, `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` | block-diffusion drafter (5 Qwen3-style layers, local conv, candidate path selector), K7 verification |
| FlashInfer NVFP4 | `vllm/v1/attention/backends/flashinfer.py` (+599) | non-causal NVFP4 paged-prefill via fa2 backend, XQA dedicated stream, #4346 integration |
| GDN/ReplaySSM | `vllm/model_executor/layers/mamba/*`, `vllm/v1/attention/backends/gdn_attn.py` | GDN speculative ReplaySSM half, active-width fix, page-padding guard |
| Mixed cache layout | `vllm/v1/kv_cache_interface.py`, `vllm/config/cache.py` | per-attention-group KV dtype/layout (NVFP4 target + NVFP4 draft) |
| Runtime-K | `vllm/config/speculative.py`, `vllm/v1/worker/gpu/spec_decode/*` | dynamic-K schedule, exact-shape candidate selection, per-K graph dispatch |
| Fused-KV dequant | `vllm/model_executor/models/qwen3_dflash.py` | quantized-drafter fused-KV fallback (#51581 class) |

## Verified measurements (RTX 5090, SM120, K7, maxseq4, 8 GiB NVFP4 KV)

| Item | Value | Source |
|---|---|---|
| KV pool @ 8 GiB pin, 262K boot | 325,139 tokens | production promotion (2026-08-24) |
| c1 decode (techprose, 1024 tok) | 107.86 tok/s (autotune-on) | `evidence/runtime-k/mtp-no-rebuild-20260824/` |
| c4 aggregate (techprose) | 405.53 tok/s | same |
| c4 coding workload (3 legacy prompts + C++ lane) | 349.2 aggregate median, 2.31x vs c1 151.2 | `bench/2026-08-25-coding-benchmark-c1-vs-c4-dflash2-k7` |
| 4-lane 4,096-token soak | 368.51 aggregate, all streams completed, Running=4 Waiting=0 | production promotion |
| Draft acceptance (K7) | 5.2–5.8 mean accepted length, ~61% accept rate | coding benchmark |
| NIAH | 184,024-token needle recovered exactly | production promotion |
| Greedy determinism | PASS (canary `19×23 → 437`) | production promotion |
| Tools / structured output | 10/10, JSON-schema PASS | 2026-08-21 acceptance gates |
| Language-only narrative decode | 116.15 tok/s mean, CV 1.6% (5 runs) | 2026-08-25 Club 3090 `bench.sh`, prefill disabled |
| Language-only code decode | 202.43 tok/s mean, CV 11.2% (5 runs) | same |
| Vision-capable control | narrative 61.6, code 105.7 tok/s | same image/config except multimodal serving mode |
| Restarts / OOM | 0 / 0 | production promotion |

## Gate results — release candidate

- Focused source tests: 23/23 PASS (2026-08-21); 60/60 spec-decode tests
  (runtime-K, 2026-08-23).
- Greedy determinism PASS; canary 437.
- NIAH PASS at 30,683 and 184,024 prompt tokens.
- Tools 10/10; JSON-schema structured output PASS.
- Language-only startup PASS with no multimodal warmup; Compose explicitly
  passes `--language-model-only`.
- NVFP4 target/draft cache and FlashInfer XQA confirmed in live logs.

## Vision disposition

The historical CPU-sidecar correctness gates passed, but that design kept the
GPU server embedding-capable. The CPU process was idle during text-only tests;
the regression was caused by the GPU model path. In this pinned vLLM source,
Qwen3.5's `use_fused_qk_norm_rope_gate` is enabled only when
`multimodal_config.language_model_only` is true. Limiting image/video counts to
zero prunes the vision tower but does not enable the fused decoder path.

Release `.2` disabled vision and multimodal embeddings to protect decode
throughput. Release `.3` restores the optional profile after the fused M-RoPE
kernel extension: the GPU server stays on the multimodal-capable path, while
the CPU sidecar remains bounded at 4 CPUs, 6 GB, and 256 processes. Start it
with `./start.sh --vision` and run `./verify.sh --vision` for the exact fixture.

## CPU vision sidecar speedups (2026-08-26, host-side, no image rebuild)

The sidecar is bind-mounted from the host (`./sidecar.py` →
`/opt/vllm-release/sidecar.py`), so both levers below are host-file changes
that take effect on `docker compose --profile vision up -d --force-recreate
vision`; the `.3` runtime image and model artifacts are unchanged.

### Lever 1 — core cap 4 → 8

Measured ~1.5–1.6× faster encode, not 2×: encode is partially
memory-bandwidth-bound under WSL2, so scaling is sub-linear. 1 MP image
(1280×960): 16.97 s → 10.66 s. `torch.compile` was probed and root-caused as
a dead end (~0%: the ViT block loop graph-breaks and falls back to eager; the
fixing flags crash on a data-dependent `torch.split`), so it was reverted.

### Lever 2 — INT8 ViT tower (default on, `SIDECAR_INT8=1`)

No ONNX/OpenVINO in the image, so `torch.ao.quantization.quantize_dynamic`
quantizes the 110 ViT Linear layers (INT8 weights, fp32 activations):

| Input | 8-core eager | 8-core INT8 | Speedup |
|---|---:|---:|---:|
| 640×480 | 2.88 s | 1.72 s | 1.67× |
| 1280×960 | 10.66 s | 7.18 s | 1.48× |
| 1920×1080 | 10.67 s | 6.84 s | 1.56× |
| 2560×1440 | 10.52 s | 6.68 s | 1.58× |
| text-only | 0.37 s | 0.40 s | ~1.0× |

**Accuracy cost:** pooler-embedding cosine similarity vs fp32 ≈ 0.90
(0.906 noise / 0.897 structured / 0.906 solid) — real but moderate; opt-out
via `SIDECAR_INT8=0`. End-to-end 1 MP image: ~16.9 s (original 4-core eager)
→ ~6.8 s (current 8-core INT8) ≈ 2.5×.

Known follow-up: `quantize_dynamic` is deprecated in torch 2.13; migrate to
`torchao.quantization.quantize_` on the next torch bump.

## Rollback

Stop the DFlash2 Compose project (`./stop.sh`) and start the unchanged
pinned MTP project (`vllm-sm120-nvfp4-mtp` on :18079). Both repos, images,
model caches, and configs remain installed; only one 27B GPU server runs
at a time on a 32 GiB card.
