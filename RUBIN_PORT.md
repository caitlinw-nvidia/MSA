# Rubin sparse-prefill extension

This branch extends Blackwell port 384bf3cc8dfc529adc540a7a51aef5bc2f5b457f.
Donor: MiniMax-AI/MSA nv_dev 9ae7751cf6b28f1c551a284de90dddf9971d621e.
The attention/combine source at that revision matches the Blackwell donor
3b9fed061dd68d9c082c1874a9ac6ba966724f67. Indexer changes come from
e1639ac5f115a259de71e0a813becbffdde50389.

## Attention and combine

The existing imported kernel module now also supports SM107. Blackwell keeps
its Q3/Q2 buffers, generic MMA instructions, and two-stage combine. Rubin uses
Q4 for native FP8 and Q5 for BF16, SM107 FP8 K64 MMA, and four-stage combine.
FP32 softmax is the default, with P448 enabled for FP8 probabilities.
Packed-FP16 softmax and its TMEM load/max reduction are opt-in through the
direct run_pagekv launcher's enable_fp16_softmax=True parameter. Native Rubin
FP8 retains K64 MMA by default (enable_2x_fp8=None). Both options reject
explicit True for unsupported architecture/dtype and enter the compile key.
The public adapter uses these defaults; the legacy dev interface is unchanged.

Existing dev rubin_helpers.py and rubin_softmax_helpers.py exactly match the
donor and are reused unchanged. The existing softmax helper receives an explicit
fp8_probability_scale=448; its default remains unchanged for other callers.
This is not a bitwise-equivalence claim against old FP32 softmax.

Dev's metadata builder, public FMHA API, gather4 descriptor fix, opt-in TMA
transaction-count handling, generic-pointer descriptor prefetch, Quack imports,
and source-aware AOT cache remain in place. No new adapter or metadata module.
The existing blackwell_prefill module name is retained to avoid renaming the
Blackwell port; its capabilities now include Rubin.

FMHA_SM100_RUBIN_PREFILL=0 disables this attention/combine route on Rubin.
FMHA_SM100_BLACKWELL_PREFILL continues to control Blackwell independently.
The native route retains the Blackwell port's shape/layout/scale restrictions,
including matching Q/K/V storage dtypes. Pending mixed-dtype vLLM integration
edits in the separate Blackwell serving workspace are not included.

## Q8KV8 prefill indexer

Rubin-specific donor hunks are applied to dev's existing multi-head indexer:
SM107 K64 FP8 MMA, two Q buffers, 168 registers per warp role, TMEM load/max,
lookahead of physical page indices, and a host-computed page-chunk cap that
splits short-query work across resident clusters.

Dev's multi-head Q-row packing, five-word task descriptors, per-head output
shape, public wrapper, and source-aware cache are retained. The page-chunk cap
uses dev's multi-head row count and is passed as a runtime planner argument.
Blackwell keeps its previous schedule cap and kernel choices.

TopK has a separate SM107 native build identity. Ordinary offline prebuilds
continue to use their previous targets; the Rubin module is built on demand.
Decode eligibility and kernels are not changed.

## Validation

Validated on Rubin SM107 with the image's built-in CuTe compiler:
vllm-rubin-py3-devel-20260928.sqsh, Torch
2.14.0a0+b2c75dd062.nvinternal.rubin.

- Repository attention regression: 73/73 passed.
- Repository Q8 prefill indexer/TopK: 36 passed, 76 unrelated tests deselected.
  Only the test fixture's architecture gate was extended; assertions unchanged.
- CPU dispatch/options contracts: 10/10 passed, including FP32-by-default,
  explicit FP16 opt-in, and independent Blackwell/Rubin route controls.
- FP32 softmax with P448 aligned: 25 FP8 and 10 BF16 cases were byte-identical
  to retained dev in partial outputs, LSE, and final output. The dev diagnostic
  copy enables P448 and its matching LSE correction; unmodified dev leaves
  P448 off. BF16 is an unchanged control.
- FP16 softmax explicitly enabled in both: 25/25 FP8 cases matched dev
  byte-for-byte with P448 enabled in both (also 25/25 with P448 off in both).
- Unchanged repository sparse-prefill benchmark: all five default sequence
  lengths passed for FP8 and BF16, using both softmax settings. FP32-default
  latencies in ms for 8K/16K/32K/64K/128K were
  FP8: 0.7135/1.2595/2.3024/4.1753/8.6600;
  BF16: 0.9206/1.6628/3.0025/5.8733/11.7498.
  CUDA-event timing with default cold-L2 behavior; plan construction excluded.
  This is not an old-dev-versus-port speedup measurement.

A separate custom reference harness showed discrepancies for fresh-prefill
FP8 GQA 4/8/16. Its FP32 reference differs from the packed-FP16 path and it
does not establish a port-specific correctness failure: those configurations
match dev byte-for-byte when arithmetic settings match. Repository correctness
and bitwise equality do not establish end-to-end model accuracy.

Detailed raw logs, diagnostic harnesses, source hashes and caches are persisted
under msa-rubin-port-20261005 on ComputeLab scratch and Hecate Lustre.
Diagnostic kernel copies and benchmark results are not package source.
No Rubin speedup or end-to-end serving claim is made by this commit.
