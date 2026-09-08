# Patches against vLLM v0.28.0

Base: [`2cf0a6915ce544dc493a0990f2ea38d81601128a`](https://github.com/vllm-project/vllm/commit/2cf0a6915ce544dc493a0990f2ea38d81601128a).
Reviewed against the official release on 2026-09-08. `series` defines the
application order; `../SHA256SUMS` pins the files. Both patches are applied
by every build, including text-only deployments that enable multimodal embeddings.

The previous overlay changed 51 files, plus three in the vision patch.
This series changes **11 upstream files: nine production files and two test
files**. The production changes are eight Python files and one CUDA file.
Repository tests add CPU checks and an SM12x GPU gate without expanding the
upstream overlay.

## 0001: SM12x NVFP4 and quantized DFlash

| Change | Why it remains | Upstream reference / removal condition |
|---|---|---|
| Enable NVFP4 KV on SM120/SM121; select HND storage, BF16 queries/output, and FA2 prefill, including non-causal draft attention | v0.28.0 restricts NVFP4 KV to SM100 and rejects non-causal NVFP4 prefill | Related open [#54772](https://github.com/vllm-project/vllm/pull/54772) and [#53979](https://github.com/vllm-project/vllm/pull/53979). Remove when a release supports this complete path. |
| Write linear V block scales on SM12x | FA2/XQA consume linear scales; the stock CUDA writer swizzles them for SM100 | Keep until upstream's writer matches the SM12x reader layout. Preserve the SM100 branch. |
| Pass NVFP4 data, scales, and speculative masks to XQA; optionally isolate its CUDA stream | The release's dedicated SM12x XQA call excludes NVFP4 | Adapted from open [#53543](https://github.com/vllm-project/vllm/pull/53543). Remove when released and the GPU gate passes. |
| Configure the draft's quantization mappings and project context KV through its quantized layers | The pinned compressed-tensors draft has packed weights and scales, rather than sliceable floating-point QKV weights | Mapping issue: [#53122](https://github.com/vllm-project/vllm/pull/53122). The local fallback preserves the existing unquantized fused path. Remove when released support handles this checkpoint. |
| Embed draft inputs outside the compiled forward; keep warmup selector indices in bounds | Avoids draft compile/warmup failures | Adapted from open [#53978](https://github.com/vllm-project/vllm/pull/53978). |
| Honor `speculative_config.enforce_eager` in the V2 DFlash graph manager | Keeps the draft eager while retaining target CUDA graphs | Remove when the upstream DFlash graph manager honors this flag. |
| Include draft width and eager mode in the compilation hash | Prevents reuse of incompatible draft graphs when fixed K is changed between starts | Related open [#53292](https://github.com/vllm-project/vllm/pull/53292). |

Open PRs are references, not claims of upstream approval. The compatibility
patch adapts only the required parts to the release API; it is not a bulk
cherry-pick of these branches. Full GPU serving validation remains pending.

## 0002: upstream fused M-RoPE fix

This is the source diff of upstream commit
[`07ef21bc69842665cf5056371ea15a6d014f3b05`](https://github.com/vllm-project/vllm/commit/07ef21bc69842665cf5056371ea15a6d014f3b05),
[#52676](https://github.com/vllm-project/vllm/pull/52676), including its kernel
tests. It merged on August 25 but is absent from the v0.28.0 release tree.

It keeps the fused QK-norm/RoPE/gate path for the pinned model's interleaved
M-RoPE configuration. It replaces our broader custom vision implementation.
Contiguous M-RoPE layouts retain upstream's fallback behavior. Remove the
patch once a stable release contains this fix.

## Removed changes

v0.28.0 already includes DFlash2's model, convolution, candidate selector,
V2 runner selection, loader indirection, explicit draft causality, target
RoPE inheritance, and multiple draft KV groups. It also includes the SM12x
XQA path and uniform-query checks, GDN gate alignment, and packed KV cache
allocation. These no longer need local backports.

The default profile uses fixed K7. The dynamic-K schedule, sliding-window
overrides, ReplaySSM experiments, benchmark instrumentation, and unrelated
parser/scheduler modifications were removed. Tool calling uses upstream
vLLM; the existing tool gate must pass before promoting a runtime image.

FlashInfer remains the official Dockerfile's **0.6.16.post3** dependency.
That release exposes FA2 NVFP4 KV support. We do not apply a FlashInfer
source patch or claim that [#4346](https://github.com/flashinfer-ai/flashinfer/pull/4346)
is part of this build. Older performance results may include a different
FlashInfer build and must be measured again.

## Validation

With Python 3.12, `pytest`, and CPU PyTorch installed:

```bash
CHECK_PATCH_APPLY=1 ./scripts/check-release.sh
```

This clones the pinned release, applies the complete series, compiles its
Python sources, and runs CPU regressions against methods extracted from
that checkout. These cover attention dispatch/dtypes, packed cache views,
quantized context projections, eager mode, compilation hashes, warmup,
and CUDA stream ordering with mocked streams. They do not execute CUDA or
load the model checkpoints.

After building on an SM12x host, run the native writer/attention gate in
the new image before starting the model server:

```bash
docker run --rm --gpus all --user root \
  -v "$PWD/tests:/tests:ro" --entrypoint bash \
  vllm-sm12x-nvfp4-dflash2:local-v0280-dflash2-1 \
  -lc 'python3 -m pip install pytest && python3 -m pytest -q /tests/test_sm12x_nvfp4.py'
./start.sh --vision
./verify.sh --full
./verify.sh --vision
```

Also run the upstream fused M-RoPE kernel tests from a prepared source tree
against the built runtime, then repeat the concurrency/capacity benchmarks.
Record the image digest, GPU, source commit, and patch checksums with the
results. SM120 and SM121 need separate hardware validation.
