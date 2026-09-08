"""GPU gate: vLLM's NVFP4 writer -> FlashInfer FA2 and masked XQA on SM12x.

Run inside the freshly built image, before starting the model server. These
checks exercise native code; the CPU source tests cannot validate this path.
"""

from types import SimpleNamespace as NS

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires an SM12x GPU", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("requires an SM12x GPU", allow_module_level=True)

import flashinfer
from vllm.utils.torch_utils import nvfp4_split_data_scale
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    _make_xqa_draft_block_mask,
)


def dequantize(data, scales, global_scale):
    # E2M1: low nibble precedes high nibble. Scales are linear on SM12x.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).long()
    return lut[codes] * scales.float().repeat_interleave(16, dim=-1) * global_scale


@pytest.mark.parametrize("heads,kv_heads,head_dim", [(24, 4, 256), (32, 8, 128)])
@pytest.mark.parametrize("causal", [True, False])
@torch.inference_mode()
def test_writer_fa2_and_masked_xqa(heads, kv_heads, head_dim, causal):
    torch.manual_seed(17)
    device = "cuda"
    page_size, pages, q_len, batch = 16, 24, 8, 3
    lengths = [27, 55, 88]
    shape = FlashInferBackend.get_kv_cache_shape(
        pages, page_size, kv_heads, head_dim, "nvfp4"
    )
    cache = torch.zeros(shape, dtype=torch.uint8, device=device)
    # Vary scales within each page to detect the incorrect SM100 V-scale swizzle.
    amplitude = torch.logspace(-1, 1, pages * page_size, device=device)[:, None, None]
    key = (
        torch.randn(
            pages * page_size, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * amplitude
    )
    value = torch.randn_like(key) * amplitude.flip(0)
    key, value = key.bfloat16(), value.bfloat16()
    layer = NS(
        _k_scale=torch.tensor(1.0, device=device),
        _v_scale=torch.tensor(1.0, device=device),
    )
    impl = NS(
        is_kvcache_nvfp4=True,
        num_kv_heads=kv_heads,
        cache_dtype="nvfp4",
        kv_sharing_target_layer_name=None,
    )
    FlashInferImpl.do_kv_cache_update(
        impl,
        layer,
        key,
        value,
        cache,
        torch.arange(pages * page_size, device=device, dtype=torch.int64),
    )
    kd, ks = nvfp4_split_data_scale(cache[:, :kv_heads])
    vd, vs = nvfp4_split_data_scale(cache[:, kv_heads:])
    deq_k, deq_v = dequantize(kd, ks, 1.0), dequantize(vd, vs, 1.0)
    for restored, original in [(deq_k, key), (deq_v, value)]:
        original = (
            original.view(pages, page_size, kv_heads, head_dim).transpose(1, 2).float()
        )
        relative_rms = (
            restored - original
        ).square().mean().sqrt() / original.square().mean().sqrt()
        assert relative_rms < 0.2

    table = torch.arange(pages, dtype=torch.int32, device=device).reshape(batch, -1)
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    counts = [(n + page_size - 1) // page_size for n in lengths]
    indptr = torch.tensor(
        [0, counts[0], sum(counts[:2]), sum(counts)], dtype=torch.int32, device=device
    )
    indices = torch.cat([table[i, :n] for i, n in enumerate(counts)])
    last_page = torch.tensor(
        [(n - 1) % page_size + 1 for n in lengths], dtype=torch.int32, device=device
    )
    qptr = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=device)
    query = torch.randn(
        batch * q_len, heads, head_dim, dtype=torch.bfloat16, device=device
    )
    scale = head_dim**-0.5
    # Reference attention over the exact values read from the quantized cache.
    references = []
    for i, length in enumerate(lengths):
        k = deq_k[table[i, : counts[i]].long()].transpose(1, 2).flatten(0, 1)[:length]
        v = deq_v[table[i, : counts[i]].long()].transpose(1, 2).flatten(0, 1)[:length]
        k, v = (
            k.repeat_interleave(heads // kv_heads, dim=1),
            v.repeat_interleave(heads // kv_heads, dim=1),
        )
        q = query[i * q_len : (i + 1) * q_len].float()
        logits = torch.einsum("qhd,khd->hqk", q, k) * scale
        if causal:
            mask = (
                torch.arange(length, device=device)[None, :]
                <= length - q_len + torch.arange(q_len, device=device)[:, None]
            )
            logits.masked_fill_(~mask[None], float("-inf"))
        references.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), v))
    reference = torch.cat(references).bfloat16()
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="fa2"
    )
    wrapper.plan(
        qptr,
        indptr,
        indices,
        last_page,
        heads,
        kv_heads,
        head_dim,
        page_size,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type="nvfp4",
    )
    prefill = wrapper.run(
        query, (kd, vd), kv_cache_sf=(ks, vs), k_scale=1.0, v_scale=1.0
    )
    torch.testing.assert_close(prefill, reference, atol=0.15, rtol=0.05)
    mask = _make_xqa_draft_block_mask(q_len, causal, torch.device(device))
    mask = mask[None].expand(batch, -1, -1).contiguous()
    output = torch.empty_like(query)
    workspace.zero_()
    flashinfer.decode.xqa_batch_decode_with_kv_cache(
        query=query,
        kv_cache=(kd, vd),
        kv_cache_sf=(ks, vs),
        workspace_buffer=workspace,
        block_tables=table,
        seq_lens=seq_lens,
        max_seq_len=max(lengths),
        bmm1_scale=scale,
        bmm2_scale=1.0,
        out=output,
        kv_layout="HND",
        q_len_per_req=q_len,
        mask=mask,
    )
    torch.testing.assert_close(output, reference, atol=0.15, rtol=0.05)
