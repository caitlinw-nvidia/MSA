# uGPU-localized Q8KV4 decode indexer (experiment)

Rubin GPUs have two memory locality domains (uGPUs). Ordinary allocations
interleave across both, so about half of every kernel's reads cross the link
between them. This experiment assumes the index-K pool is already localized:
logical page `j` of every request lives in domain `j % 2` (page-interleaved, like
DCP with `cp_kv_cache_interleave_size` equal to the block size), and each CTA
scores the pages that live in its own SM's domain. Scores are bitwise identical
to the ordinary kernels and top-k is unchanged.

| heads | baseline | localized kernel |
| --- | --- | --- |
| 1 | `q8kv4_indexer_decode_sm107` (vllm-project/MSA#34, merged here) | `csrc/q8kv4_indexer_decode_sm107_parity.cu`: per-domain tile lists and counters, each CTA claims tiles of its own domain and steals the other's once its own run out |
| 2, 4 | `Q8KV4DecodeIndexerSm100` | `src/sm100/q8kv4_indexer_decode_local.py`: per-domain slots, one claim per CTA before the PDL wait, falling back to the other domain |

Both find a CTA's domain from `%smid` in an SM -> domain table that the host
builds from the driver's SM split (`cuDevSmResourceSplit`, CUDA 13.4 driver) and
passes as compile-time constants (H=2/4) or kernel parameters (H=1); there is
no memory load. `locality_ext.cu` (adapted from vllm-project/MSA#38's
`locality_domains.cu`, MIT) reports the split and allocates the striped pool.

```bash
python benchmarks/ugpu_locality/bench_decode_indexer_locality.py --heads 4 2 1 --out results.json
```

The benchmark runs the 28 repository decode cases with their production
weights, cold L2 (layer rotation with reuse distance at least 2 x L2), times the
score kernel only (CUDA graph, 7 rounds alternating the arm order, 100 replays
each), and checks every valid score bitwise and every top-k row.

## Results (2026-10-13)

Weighted mean score-kernel latency, all 28 cases, 0 score and 0 top-k mismatches:

| GPU | heads | baseline | localized | change |
| --- | --- | --- | --- | --- |
| VR200 TS4B (DLCluster) | 4 | 25.556 us | 25.504 us | +0.20 % |
| VR200 TS4B (DLCluster) | 2 | 24.148 us | 24.107 us | +0.17 % |
| VR200 TS4B (DLCluster) | 1 | 16.817 us | 17.407 us | -3.51 % |

On TS4B, mid-size batches gain 2-4 %, b8 at 4K-10K loses 4-15 % (the claim and
split cost more than they save on a few pages) and 50K-200K is within about
1 %. The two- and four-head kernel reaches only about 7.6-8.1 TB/s at the largest
cases, below the 13.4 TB/s cap of interleaved reads, so locality has little to
remove; the one-head kernel reaches about 12.2 TB/s but its parity scheduling
costs about 4 % and locality itself was worth about +0.8 %. An earlier Hecate
TS6 run of the four-/two-head design (CTA domain from a hardware register, CuTe
DSL of that image) measured +6.5 % / +3.4 %; the `%smid` version has not been
rerun on TS6 yet.
