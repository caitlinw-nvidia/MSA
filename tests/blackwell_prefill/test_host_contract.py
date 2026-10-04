"""CPU-only integration checks; these do not compile or execute a CUDA kernel.

Run directly with Python to avoid the GPU test suite's package/conftest imports.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]


class TensorMetadata:
    def __init__(self, shape, dtype=torch.bfloat16, contiguous=True, address=16):
        self.shape, self.dtype = shape, dtype
        self.ndim, self.device, self.is_cuda = len(shape), "cuda:0", True
        self.contiguous, self.address = contiguous, address

    def is_contiguous(self):
        return self.contiguous

    def stride(self, dim):
        return 1

    def data_ptr(self):
        return self.address


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


# Load the private host helpers without importing the existing CUDA-dependent
# sparse module on this CPU-only test host. The production definitions are used.
_namespace = {"torch": torch, "os": os}
load_functions("python/fmha_sm100/sparse_fmha_adapter.py", {
    "_blackwell_prefill_enabled", "_supports_blackwell_prefill",
    "_can_run_blackwell_prefill", "_prepare_blackwell_metadata",
    "_run_blackwell_prefill",
}, _namespace)
ADAPTER = types.SimpleNamespace(
    can_run=_namespace["_can_run_blackwell_prefill"],
    prepare=_namespace["_prepare_blackwell_metadata"],
)


class DispatchContract(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"FMHA_SM100_BLACKWELL_PREFILL": "1"})
        self.env.start()
        self.cap = patch.object(torch.cuda, "get_device_capability", return_value=(10, 3))
        self.cap.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.cap.stop)
        self.q = TensorMetadata((8192, 64, 128))
        self.k = TensorMetadata((64, 4, 128, 128))
        self.v = TensorMetadata(self.k.shape)
        self.table = TensorMetadata((1, 64), torch.int32)
        self.plan = dict(kv_block_num=16, page_size=128, causal=True,
                         blackwell_default_causal_offset=True, usable_SM_count=-1)

    def accepts(self, **kwargs):
        return ADAPTER.can_run(self.q, self.k, self.v, self.plan, self.table, **kwargs)

    def test_blackwell_native_dtypes(self):
        for capability in ((10, 0), (10, 3)):
            with patch.object(torch.cuda, "get_device_capability", return_value=capability):
                for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                    self.q.dtype = self.k.dtype = self.v.dtype = dtype
                    self.assertTrue(self.accepts())

    def test_rubin_and_other_architectures_keep_existing_route(self):
        for capability in ((10, 7), (9, 0), (12, 0)):
            with patch.object(torch.cuda, "get_device_capability", return_value=capability):
                self.assertFalse(self.accepts())

    def test_unsupported_plans_and_custom_offsets(self):
        for key, value in (("kv_block_num", 8), ("page_size", 64), ("causal", False),
                           ("usable_SM_count", 16), ("blackwell_default_causal_offset", False)):
            with patch.dict(self.plan, {key: value}):
                self.assertFalse(self.accepts(), (key, value))
        self.assertFalse(self.accepts(q_offset_override=0))
        self.assertFalse(self.accepts(k_scale=object()))

    def test_layout_dtype_and_output_guards(self):
        self.v.dtype = torch.float8_e4m3fn
        self.assertFalse(self.accepts())
        self.v.dtype = torch.bfloat16
        self.q.contiguous = False
        self.assertFalse(self.accepts())
        self.q.contiguous = True
        self.assertFalse(self.accepts(out=TensorMetadata(self.q.shape, torch.float32)))
        self.q.shape = (8192, 16, 128)  # BF16 GQA4 unsupported by the donor.
        self.assertFalse(self.accepts())

    def test_explicit_legacy_switch(self):
        with patch.dict(os.environ, {"FMHA_SM100_BLACKWELL_PREFILL": "0"}):
            self.assertFalse(self.accepts())

    def test_metadata_is_rebuilt_from_each_layers_topk(self):
        seen = []

        def prepare(q2k, *args, **kwargs):
            seen.append((q2k, kwargs))
            return q2k, None, None

        shape = dict(total_k=8192, total_rows=64, max_seqlen_q=8192, max_seqlen_k=8192)
        with patch.dict(_namespace, {"build_k2q_csr": prepare}):
            first, second = object(), object()
            self.assertIs(ADAPTER.prepare(first, None, None, **shape)[0], first)
            self.assertIs(ADAPTER.prepare(second, None, None, **shape)[0], second)
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(x[1]["qhead_per_kv"] == 16 for x in seen))
        self.assertTrue(all(x[1]["return_schedule"] for x in seen))


class BenchmarkPacking(unittest.TestCase):
    def test_repacking_preserves_codes_and_each_block_scale(self):
        ns = {"torch": torch, "_SCALE_GROUPS": 8}
        load_functions("python/fmha_sm100/decode_q8kv4/interface.py", {"interleave_v_scales"}, ns)
        load_functions("python/fmha_sm100/cute/quantize.py", {"_round_up", "nvfp4_scale_128x4_offset"}, ns)
        load_functions("benchmarks/bench_sparse_attention_ops.py", {"_nvfp4_head_slot_cache"}, ns)
        prefill = types.ModuleType("fmha_sm100.prefill_q8kv4")
        prefill.interleave_v_scales = ns["interleave_v_scales"]
        quantize = types.ModuleType("quantize")
        quantize.nvfp4_scale_128x4_offset = ns["nvfp4_scale_128x4_offset"]
        b, length, heads = 2, 256, 4
        rows = b * length * heads
        inputs, originals = [], []
        for seed in (13, 29):
            generator = torch.Generator().manual_seed(seed)
            codes = torch.randint(0, 256, (b * length, heads, 64), dtype=torch.uint8, generator=generator)
            scales = torch.randint(0, 127, (rows, 8), dtype=torch.uint8, generator=generator)
            offsets = ns["nvfp4_scale_128x4_offset"](torch.arange(rows)[:, None], torch.arange(8)[None, :], 8)
            swizzled = torch.empty_like(scales).flatten()
            swizzled[offsets.flatten()] = scales.flatten()
            inputs.append(types.SimpleNamespace(data=codes, logical_scale_shape=(rows, 8), scale_128x4=swizzled))
            originals.append((codes, scales))
        with patch.dict(sys.modules, {prefill.__name__: prefill, quantize.__name__: quantize}):
            packed = ns["_nvfp4_head_slot_cache"](*inputs, b, length, heads)
        for side, slots in enumerate(packed):
            raw = slots.flatten(2)
            for batch in range(b):
                for token in range(length):
                    page, local = batch * (length // 128) + token // 128, token % 128
                    for head in range(heads):
                        row = (batch * length + token) * heads + head
                        self.assertTrue(torch.equal(raw[page, head, local * 64:(local + 1) * 64], originals[side][0][batch * length + token, head]))
                        for group in range(8):
                            offset = local * 8 + group if side == 0 else (local // 4) * 32 + group * 4 + local % 4
                            self.assertEqual(int(raw[page, head, 128 * 64 + offset]), int(originals[side][1][row, group]))


if __name__ == "__main__":
    unittest.main()
