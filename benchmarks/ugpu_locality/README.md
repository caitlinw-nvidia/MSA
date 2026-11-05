# uGPU-localized Q8KV4 decode indexer (experiment)

Rubin GPUs have two memory locality domains (uGPUs). Ordinary allocations
interleave across both, so about half of every kernel's reads cross the link
between them. This experiment assumes the index-K pool is already localized:
logical page `j` of every request lives in domain `j % 2` (page-interleaved, like
DCP with `cp_kv_cache_interleave_size` equal to the block size), and each CTA
scores the pages that live in its own SM's domain. Scores are bitwise identical
to the ordinary kernel and top-k is unchanged.

`src/sm100/q8kv4_indexer_decode_local.py` (`Q8KV4DecodeIndexerLocalSm100`, two or
four heads) subclasses `Q8KV4DecodeIndexerSm100` and changes only the work split:
per-domain prefix sums of each request's pages in that domain, one claim per CTA
of a slot of its own domain before the PDL wait (the other domain's once its own
run out), and a stride-2 walk over the page table. A CTA finds its domain from
`%smid` in an SM -> domain table that the host builds from the driver's SM split
(`cuDevSmResourceSplit`, CUDA 13.4 driver) and compiles into the kernel; there is
no memory load. `locality_ext.cu` (adapted from vllm-project/MSA#38's
`locality_domains.cu`, MIT) reports the split and allocates the striped pool.

| benchmark | cases |
| --- | --- |
| `bench_decode_indexer_locality.py --heads 4 2` | the 28 repository decode cases with their production weights |
| `bench_decode_indexer_steal.py` | one DEP rank of MiniMax-M3 decode: 4 heads, batch 16 and 20, lengths log-uniform in [4K, 200K], two 200K among 8K, all 64K |

Both use cold L2 (layer rotation with reuse distance at least 2 x L2), time the
score kernel only (CUDA graph, 7 rounds alternating the arm order, 100 replays
each), and check every valid score bitwise and every top-k row.

## Results

DEP-rank cases (`bench_decode_indexer_steal.py`), mean score-kernel latency over the
8 cases, 0 score and 0 top-k mismatches:

| GPU | image (CuTe DSL, CUDA) | plain | localized | change |
| --- | --- | --- | --- | --- |
| VR200 TS6 (Hecate) | trtllm-rubin-sm107 (4.5, 13.4) | 14.506 us | 13.329 us | -8.1 % |
| VR200 TS6 (Hecate) | vllm rubin-py3-devel (4.8, 13.5) | 13.930 us | 13.830 us | -0.75 % |
| VR200 TS4B (DLCluster) | vllm rubin-py3-devel (4.8, 13.5) | 13.708 us | 13.589 us | -0.9 % |

With the same image, TS6 and TS4B agree within about 1.5 % and localization is worth
about 1 % on both. The larger TS6 gain seen with the older image comes from code
generation: that compiler makes the plain kernel slower (branch-resolution and
instruction-fetch stalls) and the localized one faster. The 28 repository cases on
TS4B (2026-10-13) were 0.20 % faster with four heads (25.556 -> 25.504 us, weighted)
and 0.17 % with two (24.148 -> 24.107 us); the kernel
reaches only about 7.6-8.1 TB/s at the largest cases, below the 13.4 TB/s cap of
interleaved reads, so locality has little to remove.
