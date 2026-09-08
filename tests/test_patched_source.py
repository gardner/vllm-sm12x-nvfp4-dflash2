"""CPU regression tests for methods in the actual patched upstream checkout.

AST extraction removes module imports and compile decorators that require CUDA.
The method bodies under test are compiled unchanged; no vLLM code is copied here.
"""

import ast
import hashlib
import os
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn
from torch.nn import functional as F

SOURCE = Path(os.environ["VLLM_SOURCE_DIR"])
ATTENTION = "vllm/v1/attention/backends/flashinfer.py"
DRAFT = "vllm/model_executor/models/qwen3_dflash.py"


def extract(path, name, namespace=None, methods=None):
    tree = ast.parse((SOURCE / path).read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    if methods is not None:
        node.bases = []
        node.keywords = []
        node.body = [n for n in node.body if getattr(n, "name", None) in methods]
    for item in ast.walk(node):
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)):
            item.decorator_list = [
                d
                for d in item.decorator_list
                if isinstance(d, ast.Name)
                and d.id in ("classmethod", "staticmethod", "contextmanager")
            ]
    scope = {"torch": torch, "nn": nn, "F": F, **(namespace or {})}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / path), "exec"), scope)  # noqa: S102 -- test verified source
    return scope[name]


def platform(sm):
    return NS(
        is_device_capability=lambda cap: sm == cap,
        is_device_capability_family=lambda cap: sm // 10 == cap // 10,
        get_device_capability=lambda: NS(major=sm // 10),
    )


@pytest.mark.parametrize(
    "sm,allowed", [(90, False), (100, True), (120, True), (121, True)]
)
def test_nvfp4_capability_and_layout(sm, allowed):
    backend = extract(
        ATTENTION,
        "FlashInferBackend",
        {
            "current_platform": platform(sm),
            "supports_trtllm_attention": lambda **kw: sm == 100,
        },
        ["supports_kv_cache_dtype", "get_required_kv_cache_layout"],
    )
    assert backend.supports_kv_cache_dtype("nvfp4") is allowed
    assert backend.get_required_kv_cache_layout() == ("HND" if sm >= 100 else None)


@pytest.mark.parametrize(
    "sm,dtype",
    [(100, torch.float8_e4m3fn), (120, torch.bfloat16), (121, torch.bfloat16)],
)
@pytest.mark.parametrize("is_prefill", [True, False])
def test_nvfp4_query_dtype(sm, dtype, is_prefill):
    backend = extract(ATTENTION, "FlashInferBackend", {}, ["get_dtype_for_flashinfer"])
    builder_cls = extract(
        ATTENTION,
        "FlashInferMetadataBuilder",
        {
            "current_platform": platform(sm),
            "FlashInferBackend": backend,
            "force_use_trtllm_attention": lambda: None,
        },
        ["get_q_data_type"],
    )
    builder = builder_cls()
    builder.cache_dtype = "nvfp4"
    builder.model_config = NS(dtype=torch.bfloat16)
    builder.vllm_config = NS(
        attention_config=NS(disable_flashinfer_q_quantization=False)
    )
    assert builder.get_q_data_type(is_prefill=is_prefill) == dtype


@pytest.mark.parametrize("sm", [100, 120, 121])
@pytest.mark.parametrize("causal", [True, False])
def test_nvfp4_prefill_backend(sm, causal):
    builder_cls = extract(
        ATTENTION,
        "FlashInferMetadataBuilder",
        {
            "BatchPrefillWithPagedKVCacheWrapper": lambda *args, **kwargs: NS(**kwargs),
            "get_kv_cache_layout": lambda: "HND",
            "current_platform": platform(sm),
        },
        ["_get_prefill_wrapper"],
    )
    builder = builder_cls()
    builder.use_dcp = builder.has_sinks = False
    builder.is_kvcache_nvfp4 = True
    builder.use_fa2_nvfp4_kv = sm in (120, 121)
    builder._prefill_wrapper = builder._noncausal_prefill_wrapper = None
    builder._get_workspace_buffer = lambda: None
    if sm == 100 and not causal:
        with pytest.raises(NotImplementedError):
            builder._get_prefill_wrapper(causal)
    else:
        assert builder._get_prefill_wrapper(causal).backend == (
            "trtllm-gen" if sm == 100 else "fa2"
        )


class Unquantized:
    pass


class PackedProjection:
    """No raw .weight: emulate a packed matrix and distinct per-row scales."""

    def __init__(self, packed, scale, bias):
        self.packed = packed
        self.scale = scale
        self.bias = bias
        self.quant_method = object()
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return F.linear(x, self.packed.float() * self.scale[:, None], self.bias), None


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_context_projection_preserves_scales_and_kv_order(quantized, has_bias):
    torch.manual_seed(7)

    def rms_norm(out, x, weight, eps):
        out.copy_(F.rms_norm(x, (x.shape[-1],), weight, eps))

    model_cls = extract(
        DRAFT,
        "DFlashQwen3Model",
        {
            "UnquantizedLinearMethod": Unquantized,
            "ops": NS(rms_norm=rms_norm),
        },
        ["_build_context_kv_buffers", "_project_context_kv"],
    )
    model = model_cls()
    model.hidden_norm = NS(weight=torch.rand(12))
    model._rms_norm_eps = 1e-6
    layers, matrices, biases = [], [], []
    # Three layers, two KV heads, head size four, Q width eight.
    for _ in range(3):
        packed = torch.randint(-8, 8, (24, 12), dtype=torch.int8)
        scale = torch.rand(24) + 0.1
        weight = packed.float() * scale[:, None]
        bias = torch.randn(24) if has_bias else None
        proj = (
            PackedProjection(packed, scale, bias)
            if quantized
            else NS(weight=weight, bias=bias, quant_method=Unquantized())
        )
        layers.append(NS(qkv_proj=proj, q_size=8, k_norm=NS(weight=torch.ones(4))))
        matrices.append(weight)
        biases.append(bias)
    model._build_context_kv_buffers(layers, has_bias)
    context = torch.randn(5, 12)
    k, v = model._project_context_kv(context, 5, 3, 2, 4)
    normed = F.rms_norm(context, (12,), model.hidden_norm.weight, 1e-6)
    for i, (weight, bias) in enumerate(zip(matrices, biases)):
        reference = F.linear(normed, weight, bias)
        torch.testing.assert_close(k[i], reference[:, 8:16].reshape(5, 2, 4))
        torch.testing.assert_close(v[i], reference[:, 16:24].reshape(5, 2, 4))
        if quantized:
            assert layers[i].qkv_proj.calls == 1
    assert k.is_contiguous() and v.is_contiguous()


@pytest.mark.parametrize("heads,head_dim", [(4, 256), (8, 128)])
def test_backend_nvfp4_storage_views(heads, head_dim):
    split = extract("vllm/utils/torch_utils.py", "nvfp4_split_data_scale")
    backend = extract(
        ATTENTION,
        "FlashInferBackend",
        {
            "nvfp4_kv_cache_full_dim": lambda d: d // 2 + d // 16,
        },
        ["get_kv_cache_shape"],
    )
    shape = backend.get_kv_cache_shape(3, 16, heads, head_dim, "nvfp4")
    cache = torch.zeros(shape, dtype=torch.uint8)
    kd, ks = split(cache[:, :heads])
    vd, vs = split(cache[:, heads:])
    kd.fill_(1)
    ks.view(torch.uint8).fill_(2)
    vd.fill_(3)
    vs.view(torch.uint8).fill_(4)
    for page in cache:
        physical = page.flatten()
        dsize, ssize = heads * 16 * head_dim // 2, heads * 16 * head_dim // 16
        expected = torch.cat(
            [
                torch.full((n,), val, dtype=torch.uint8)
                for n, val in [(dsize, 1), (ssize, 2), (dsize, 3), (ssize, 4)]
            ]
        )
        torch.testing.assert_close(physical, expected)


class Mode(Enum):
    NONE = 0
    FULL = 1
    FULL_DECODE_ONLY = 2

    def decode_mode(self):
        return self


@pytest.mark.parametrize(
    "eager,expected", [(True, Mode.NONE), (False, Mode.FULL_DECODE_ONLY)]
)
def test_draft_eager_flag_preserves_target_mode(eager, expected):
    cls = extract(
        "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
        "DFlashSpeculator",
        {
            "CUDAGraphMode": Mode,
            "AttentionCGSupport": NS(UNIFORM_BATCH=NS(value=1)),
            "DFlashCudaGraphManager": lambda config, device, mode, **kw: NS(mode=mode),
        },
        ["init_cudagraph_manager"],
    )
    draft = cls()
    draft.speculative_config = NS(enforce_eager=eager)
    draft.attn_cg_support = NS(min_cg_support=NS(value=1))
    draft.vllm_config, draft.device, draft.num_query_per_req = NS(), "cpu", 8
    draft.init_cudagraph_manager(Mode.FULL)
    assert draft.query_cudagraph_manager.mode == expected


def test_draft_width_changes_compile_cache_hash():
    cls = extract(
        "vllm/config/speculative.py",
        "SpeculativeConfig",
        {
            "safe_hash": hashlib.sha256,
        },
        ["compute_hash"],
    )
    config = cls()
    config.method, config.enforce_eager = "dflash", True
    config.draft_model_config = NS(compute_hash=lambda: "same-model", hf_config=NS())
    config.num_speculative_tokens = 7
    first = config.compute_hash()
    config.num_speculative_tokens = 3
    assert config.compute_hash() != first


def test_selector_warmup_sentinels_are_safe():
    score = extract("vllm/model_executor/models/qwen3_dflash2.py", "_score_edges")
    # Valid candidate IDs, sentinel anchor IDs as used by dummy warmup.
    args = [
        torch.randn(12, 3),
        torch.randn(12, 3),
        torch.tensor([[[2, 4], [3, 5]]]),
        torch.randn(1, 2, 2),
        torch.randn(1, 2, 3),
        torch.tensor([-1]),
        2,
    ]
    # Use the upstream signature to avoid changing the order as it evolves.
    import inspect

    names = list(inspect.signature(score).parameters)
    values = dict(
        zip(
            [
                "successor_table",
                "predecessor_table",
                "candidate_ids",
                "unary_logits",
                "hidden",
                "anchor_token_ids",
                "top_k",
            ],
            args,
        )
    )
    out = score(**{name: values[name] for name in names})
    assert out.shape == (1, 2, 2, 2) and torch.isfinite(out).all()


def test_isolated_stream_joins_even_when_attention_raises():
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append((self.name, other.name))

    current, side = Stream("current"), Stream("side")

    @contextmanager
    def stream_context(_):
        yield

    scope = extract(
        ATTENTION,
        "_xqa_isolated_stream_scope",
        {
            "contextmanager": contextmanager,
            "envs": NS(VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM=True),
            "torch": NS(cuda=NS(current_stream=lambda: current, stream=stream_context)),
            "_get_xqa_isolated_stream": lambda: side,
        },
    )
    with pytest.raises(RuntimeError, match="attention failed"), scope(True):
        raise RuntimeError("attention failed")
    assert events == [("side", "current"), ("current", "side")]
