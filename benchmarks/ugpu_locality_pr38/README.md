# K-localized Q8KV4 decode indexer with PR #38's locality domains (experiment)

Rubin has two memory locality domains. Ordinary allocations interleave across both, so about
half of every read crosses the link between them. This experiment puts the index-K pool in
known domains and has each CTA score the K pages that live in its own SM's domain, using the
locality support from vllm-project/MSA#38 (`prefill_q8kv4/csrc/api/locality_domains.cu`):

- **Pool:** `alloc_locality_striped`, whose 2 MB locality pages alternate between the domains.
  A 9216-byte K page lives in the domain of its locality page; pages that straddle two locality
  pages are left unused.
- **Placement:** logical page `j` of every request lives in domain `j % 2`.
- **SM -> domain:** `sm_locality_domains` (the driver's `cuDevSmResourceSplit`), passed to the
  kernel as a table indexed by `%smid`. On the tested GPU the split covered 200 of 212 SMs; the
  12 it leaves unassigned are filled from `(%smid >> 1) & 1`, which the harness first checks
  against every SM the table does cover.
- **Steering:** per-domain prefix sums and one claim per CTA before the PDL wait; a CTA whose
  domain has no slots left takes one from the other domain. Per-page math is unchanged, so
  scores are bitwise identical to `Q8KV4DecodeIndexerSm100` and top-k is unchanged.

`locality_domains.cu` is compiled unchanged from this tree with a small binding
(`pr38_locality_binding.cpp`); the prefill attention kernel is not built.

| File | Contents |
| --- | --- |
| `python/fmha_sm100/cute/src/sm100/q8kv4_indexer_decode_localized.py` | localized FP16-K kernel (#41, the kernel on this branch's `dev` base) |
| `python/fmha_sm100/cute/src/sm100/q8kv4_indexer_decode_localized_fp8.py` | localized FP8 kernel (from #40, `a64ff16`), for reference |
| `benchmarks/ugpu_locality_pr38/bench_pr38_cases.py` | 28 repository decode cases, plain kernel on an ordinary pool (A) vs localized kernel on the striped pool (C) |

```bash
MSA_REPO=$(pwd) PYTHONPATH=benchmarks/ugpu_locality_pr38:python:python/fmha_sm100/cute \
  python benchmarks/ugpu_locality_pr38/bench_pr38_cases.py --heads 4 2 --out results.json
```

The benchmark uses the production weights of the 28 decode cases, cold L2 (layer rotation with
reuse distance at least 2 x L2), CUDA graphs, 3 rounds, score kernel only, and checks every
valid score bitwise and every top-k row. To reproduce the FP8 rows, replace
`q8kv4_indexer_decode.py` with `a64ff16`'s version and `q8kv4_indexer_decode_localized.py`
with `q8kv4_indexer_decode_localized_fp8.py`.

## Results (2026-10-19, Hecate TS6 Rubin, `hecate0042`, driver 620.62)

Weighted mean score-kernel latency, 28 cases, 0 score and 0 top-k mismatches:

| kernel | heads | plain | localized | change |
| --- | --- | --- | --- | --- |
| FP16-K (#41) | 4 | 23.82 us | 22.46 us | -5.7 % |
| FP16-K (#41) | 2 | 21.61 us | 20.18 us | -6.6 % |
| FP8 (#40) | 4 | 26.44 us | 24.52 us | -7.3 % |
| FP8 (#40) | 2 | 24.49 us | 23.03 us | -6.0 % |

- 64 x 200K (4 heads): FP16-K 110.3 -> 102.5 us, FP8 129.3 -> 119.6 us. Under Nsight Compute the
  memory waits (long scoreboard) drop about 20 % for FP16-K, and both kernels stay at 7-9 TB/s,
  well under the 13.4 TB/s interleaved read ceiling: the gain comes from shorter read latency,
  not bandwidth.
- Small cases (8 x 1K) are 2-15 % slower from the claim and split overhead; a production version
  would gate on batch x context.

**Node type matters.** The earlier `rubin-indexer-decode-local` branch, run unchanged on the same
TS6 node, also gains (-7.4 % at 4 heads, -4.4 % at 2 heads), while its DLCluster TS4B runs
(driver 620.47) and an earlier TS4B run of this design showed about 0 %. The cause on TS4B is not
yet known (board/firmware, driver, or software stack).

Tested against `dc762bd`; this branch's base adds #34, which changes `q8_indexer_interface.py`
(one-head dispatch). The two-/four-head path the harness uses was not rerun on this base.

## Repository-style benchmark (not yet run)

`benchmarks/bench_q8kv4_decode_indexer_locality.py` is `benchmarks/bench_q8kv4_decode_indexer.py`
with a localized arm added, so `diff` of the two files shows only the locality changes. Both arms
are timed exactly as the repository benchmark times `run()` (scores + TopK, cold-L2 slot rotation,
CUDA graphs, CV gate), and every valid score and TopK entry of the localized arm is compared
bitwise with the plain arm. It takes PR #38's locality functions from the Q8KV4 prefill
extension itself (`fmha_sm100.prefill_q8kv4.jit.load_extension`), so its first run builds that
extension. Two and four index heads, query length 8:

```bash
python benchmarks/bench_q8kv4_decode_indexer_locality.py --suite full --num-index-heads 4 --out h4.json
```

This script compiles but has not been run yet; the results above come from `bench_pr38_cases.py`
(score kernel only).
