# Blackwell sparse-prefill port

Working branch: `v1_MSA_blackwell` on `caitlinw-nvidia/MSA`.

- Base: vllm-project/MSA dev `5545effeafc567bee5ae664966fb2327292da37b`.
- Donor: MiniMax-AI/MSA nv_dev `3b9fed061dd68d9c082c1874a9ac6ba966724f67`.
- Targets: SM100 (B200/GB200) and SM103 (B300). Correctness validated on GB200; B300 validation remains pending.

## Implementation

`python/fmha_sm100/cute/src/blackwell_prefill/` contains the specialized
Blackwell forward and combine implementation. Dev’s existing CSR builder and schedule type are reused. Its manifest records
the donor commit, original source hashes, and selected changes applied to shared
helpers. The donor's duplicated `dsl/` package is removed.

Existing dev files are extended with selected Blackwell hunks rather than replaced:

| Existing file in `cute/src/common/` | Selected change |
| --- | --- |
| `softmax.py` | Optional P448 bias and selection of donor exp2 arithmetic |
| `utils.py` | Optional packed-add/FMA subtraction in exp2 emulation |
| `pipeline.py` | Optional explicit transaction-count arrival for Q-TMA |
| `tma_utils.py` | Optional generic-pointer descriptor prefetch |
| `cute_dsl_utils.py` | New `exit_thread_if` and `compile_with_timing` functions |

All new options default to dev's previous behavior; only the imported forward
opts into them. The other common helpers are reused unchanged, including sequence
metadata, masking, paging, copy primitives, MMA descriptors/helpers, barriers,
and GQA combine handling. `ParamsBase` and fake-tensor creation use the same Quack
dependency already used by dev. Unused donor-only functions are not imported.
Dev-only helpers with retained callers remain: for example, `warp_prefix_sum`
is still used by `common/tile_scheduler.py`, and the cubin/SASS dump helper is
still used by dev's compile wrapper.

These are selective donor hunks with their provenance recorded in the manifest,
not whole-file replacements or whole donor commits. The imported kernels retain
the SM100/SM103 allowlist and contain no Rubin helpers, options, or optimization
branches. Existing Rubin code used by retained dev paths is not part of this port.

`sparse_fmha_adapter.py` owns both the existing sparse-prefill entry points and
the private Blackwell routing/setup helpers; there is no separate Blackwell
adapter module. It routes compatible public FMHA calls to the imported stack:
causal, paged, head dimension/page size 128, TopK16, native FP8 or BF16 Q/K/V,
default bottom-right causal alignment, and no SM limit. FP8 supports GQA
1/2/4/8/16 and contiguous KV; BF16 supports GQA8/16 and unit-inner-stride KV.
Custom offsets, explicit quantization/output scales, other TopK values, mixed
dtypes and unsupported layouts retain dev's existing implementation.

`_prepare_blackwell_metadata()` calls dev's existing `build_k2q_csr` with
`return_schedule=True`. The donor CSR builders and scheduler module are omitted.
The serving plan can be reused across layers; metadata is rebuilt from each
layer's current TopK. The forward uses dev's existing `SparseAttentionSchedule`.
For GQA1/2/4, the launcher creates and passes the current Q tensor's gather4
map using dev's existing `create_q_gather4_tma_desc` helper.
The FP8 path retains the donor's P448/LSE convention and Blackwell exp2 selection;
this is not an attempt to preserve bitwise equality with old dev FP8 outputs.

NVFP4 already has its C++ port in dev (`18d1f6d`). Its C++ code, global scales,
block-scale shift, cache strides and other supported TopK values are preserved.
The public adapter and standalone wrapper retain dev's metadata preparation.
The C++ forward uses the ported combine for Blackwell TopK16.
SM-limited calls keep the existing schedule builder. The old CuTe NVFP4 fallback
is retained.

The imported kernels call `src.common.aot_cache` directly, using
`blackwell_prefill_` keys and dev's existing source/toolchain validation. New code cannot pick up a legacy combine
object just because a key happens to match. Existing native warmup remains valid;
the repository's sparse AOT warmup reaches the new eligible FP8/BF16 calls through
the public adapter.

## Benchmark changes

`benchmarks/bench_sparse_attention_ops.py` now supports:

- `--include-plan`: rebuild plans inside the existing CUDA-event callback.
  Without it, the existing run-only timing default is preserved. Sparse per-layer
  CSR/schedule preparation remains in run in the public FMHA API either way.
- `--nvfp4-backend auto|q8kv4|cute_dsl`: auto exercises the public NVFP4 route on
  Blackwell. `q8kv4` forces the C++ route and raises if unsupported. `cute_dsl`
  reproduces the old flat CuTe benchmark path. Other GPUs keep the old path under
  auto. No benchmark shapes, warmup/repeat durations or timing helper were changed.

The new NVFP4 benchmark repacks the existing TE-quantized BF16 inputs into dev's
per-head slots, including V-scale interleaving and the original global scales.
Quantization/repacking stays outside timing. It does not use the old comparison
adapter's randomly generated NVFP4 codes. The C++ path uses dev's existing
block-scale shift 3 for TE scales, so its E4M3 intermediate rounding can differ
from the old CuTe path.

`FMHA_SM100_BLACKWELL_PREFILL=0` disables the new FP8/BF16/combine
dispatch. For the exact old NVFP4 benchmark path, also pass
`--nvfp4-backend cute_dsl`; the environment switch alone retains dev's existing
C++ NVFP4 backend.

## Validation

After removing donor metadata, the final source passed the unchanged repository
sparse-attention script (73/73), the unchanged Q8KV4 correctness file (13/13),
and all seven host checks again on GB200, with fresh compilation caches.

The gather4-fixed port passed 73/73 cases in the unchanged repository
`tests/regression/test_sparse_attn.py`, and 13/13 tests in the unchanged
`tests/q8kv4_prefill/test_correctness.py` (including the full cases), on GB200.
The targeted matrix passed 35 native FP8/BF16 configurations and five C++ NVFP4
configurations with both dev and donor metadata builders. Cross-builder outputs
were byte-identical; the tests establish that donor metadata is unnecessary for
correctness in these cases. The cleanup therefore selects the tested dev builder.

Native public-API CUDA graph capture still encounters the pre-existing host
`.tolist()` in page-table construction. Targeted native graph checks prebuilt
that table; NVFP4 public-API graph checks passed. These results do not establish
full native public-API graph support.

Host routing and packing checks remain in
`tests/blackwell_prefill/test_host_contract.py`. The metadata check now verifies
per-layer calls to the existing dev builder.

Performance, serving TTFT, and SM103 validation remain pending. Reusing dev
metadata may change latency relative to the donor builder; no speedup claim is
made for this trimmed commit. For future benchmark comparisons, use fresh caches,
repository warmup, and the required same-container GPU-idle preflight.

Example measurement commands after GPU approval and environment setup:

```bash
python benchmarks/bench_sparse_attention_ops.py --dtype fp8 \
  --sections prefill,paged_prefill,sparse_prefill --include-plan -o fp8.tsv
python benchmarks/bench_sparse_attention_ops.py --dtype bf16 \
  --sections prefill,paged_prefill,sparse_prefill --include-plan -o bf16.tsv
python benchmarks/bench_sparse_attention_ops.py --dtype nvfp4 \
  --sections sparse_prefill --include-plan --nvfp4-backend q8kv4 -o nvfp4.tsv
```

The October 2 cluster run and its source snapshots were not modified by this
local port. The canonical editable source is now the persistent cluster workspace
`/home/scratch.caitlinw_coreai/msa-blackwell-port-correctness-20261004/MSA`.

## Decode indexer port (`v1_MSA_blackwell_decode`)

One commit on top of the prefill port ports nv_dev's Blackwell decode-indexer
optimizations, pinned to MiniMax-AI/MSA nv_dev `9ae7751`. Provenance and
adaptations are in `python/fmha_sm100/csrc/include/q8kv4_indexer/vendor_manifest.json`.
The decode attention kernels are unchanged: dev's Q8KV4 decode attention is
already ahead of nv_dev's, and nv_dev's FP8/BF16 decode attention is FlashInfer.

| Change | Where | Effect |
| --- | --- | --- |
| Single-head Q8KV4 indexer on the SM100 TMEM-source FP8 MMA (tcgen05), one persistent CTA per SM | `csrc/include/q8kv4_indexer/*`, `csrc/q8kv4_indexer_decode.cu` | Replaces the `mma.sync` kernel; two and four heads keep dev's CuTe DSL tcgen05 kernel |
| `BatchDecodeIndexerPlan`: one kernel builds a step's page prefix, per-SM work split, length snapshot and per-row candidate counts | `cute/src/sm100/decode_indexer_plan.py`, `cute/q8_indexer_interface.py` | Replaces dev's torch-op / CUB planning; graph-capturable; shareable by every decode wrapper of a step |
| Compact TopK grid, `ceil(rows / 4)` CTAs for the warp family | `csrc/indexer_topk_select.cu` | Q8KV4 decode wrappers; bit-identical output |

API: decode wrappers keep their constructors, `plan()` and `run()`. `plan()` gains
an optional `shared_plan=`, `workspace_size()` gains an optional `device=` (the
plan size depends on the SM count), and `BatchDecodeIndexerPlan` is exported from
`fmha_sm100` and `fmha_sm100.sparse`. Scores, valid-page counts and TopK output
stay token-major, as before.

Not ported: query lengths other than eight, the BF16 decode indexer, and nv_dev's
own Q8KV8/BF16 CuTe decode GEMM.

Canonical editable source for this branch (Compute Lab, Santa Clara scratch):
`/home/scratch.fkhoubsirat_coreai/caitlin-files/msa-v1-blackwell-decode-20261005/MSA`.

Validation status: syntax checks only. No CUDA compilation, GPU correctness run
or benchmark has been performed for this commit. Run
`pytest -q python/fmha_sm100/cute/test_q8_indexer.py` on SM100/SM103; it covers the
single-head kernel against the reference, shared plans across formats, a captured
`plan.update()`, and compact-grid equivalence.
