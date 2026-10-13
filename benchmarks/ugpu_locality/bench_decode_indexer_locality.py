#!/usr/bin/env python3
"""Q8KV4 decode indexer: ordinary pool (A) vs uGPU-localized pool + localized kernel (C).

Experiment for GPUs with two memory locality domains (Rubin uGPUs). Arm C
allocates the index-K pool as one virtual range whose stripes alternate between
the domains (locality_ext.alloc_striped) and builds block tables that place
logical page j of every request in domain j % 2; arm A runs the same tables on
an ordinary pool holding identical bytes.

  H=1: A = BatchDecodeIndexerQ8KV4Wrapper(num_heads=1) (SM107 one-head C++ kernel),
       C = q8kv4_indexer_decode_sm107_parity (same kernel, per-domain tile lists).
  H=2/4: A = BatchDecodeIndexerQ8KV4Wrapper (CuTe Q8KV4DecodeIndexerSm100),
       C = Q8KV4DecodeIndexerLocalSm100.

C kernels find their CTA's domain from %smid in a table built from the
driver's SM split; SMs the driver leaves out get (smid >> 1) & 1, used only if
every reported SM follows that rule (a wrong guess costs locality, never
correctness). Cases, weights and per-request lengths are the repository decode
indexer cases (benchmarks/bench_q8kv4_decode_indexer.py). Every case rotates
through enough layers with disjoint pages that one sweep reads at least
--l2-multiple x L2. Times are the score kernel only (CUDA graph of one sweep);
scores are compared bitwise on every valid (request, query, page) and top-k on
every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "python" / "fmha_sm100" / "cute"), str(REPO / "python")]

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import q8_indexer_interface as qi  # noqa: E402

sys.path.insert(0, str(REPO / "benchmarks"))
from bench_q8kv4_decode_indexer import benchmark_cases, make_seq_lens  # noqa: E402

PAGE = 128
PAGE_BYTES = qi._NVFP4_PAGE_BYTES  # 9216
QLEN = 8
# SM107 one-head kernel traits (csrc/include/q8kv4_indexer_sm107/traits.hpp).
MAX_TILE_PAGES, BANDWIDTH_TILE_PAGES, MAX_PAGES_PER_CTA = 24, 16, 6


def load_locality_ext():
    from torch.utils.cpp_extension import load

    build = Path(os.environ.get("LOCALITY_EXT_BUILD", HERE / "build_locality_ext"))
    build.mkdir(parents=True, exist_ok=True)
    return load(name="indexer_locality_ext", sources=[str(HERE / "locality_ext.cu")],
                extra_ldflags=["-lcuda"], build_directory=str(build), verbose=False)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def random_pool(pages: int, device) -> torch.Tensor:
    pool = torch.randint(0, 256, (pages, PAGE, PAGE_BYTES // PAGE), dtype=torch.uint8, device=device)
    # Scales: finite positive E4M3 (exponent field 4..9), so no NaN scores.
    pool.view(pages, PAGE_BYTES)[:, PAGE * 64:] = torch.randint(
        0x20, 0x48, (pages, PAGE_BYTES - PAGE * 64), dtype=torch.uint8, device=device)
    return pool


def time_graph(fn, layers: int, warmup: int, iters: int) -> float:
    """Median microseconds per launch over ``iters`` replays of one all-layer sweep."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / layers)
    return statistics.median(samples)


class ParityCase:
    """Per-request lengths; logical page j of every request lives in domain j % 2."""

    def __init__(self, heads, seq_lens, layers, pool_pages, pps, device, gen):
        self.heads, self.layers = heads, layers
        self.batch = int(seq_lens.numel())
        self.seq_lens = seq_lens.to(device)
        self.max_pages = math.ceil(int(seq_lens.max()) / PAGE)
        self.hist = torch.div(self.seq_lens - 1, PAGE, rounding_mode="floor").clamp_(max=self.max_pages)
        pages = torch.arange(pool_pages)
        domain = (pages // pps) % 2
        even_cols, odd_cols = (self.max_pages + 1) // 2, self.max_pages // 2
        need = (layers * self.batch * even_cols, layers * self.batch * odd_cols)
        lists = []
        for d in range(2):
            mine = pages[domain == d]
            mine = mine[torch.randperm(mine.numel(), generator=gen)]
            assert mine.numel() >= need[d], (d, mine.numel(), need[d])
            lists.append(mine[: need[d]])
        bt = torch.empty((layers, self.batch, self.max_pages), dtype=torch.int64)
        bt[:, :, 0::2] = lists[0].view(layers, self.batch, even_cols)
        bt[:, :, 1::2] = lists[1].view(layers, self.batch, odd_cols)
        assert torch.equal((bt // pps) % 2, (torch.arange(self.max_pages) % 2).expand_as(bt))
        self.block_tables = bt.to(torch.int32).to(device).contiguous()
        self.qs = [
            torch.randn(self.batch * QLEN, heads, 128, device=device).mul_(0.5).to(torch.float8_e4m3fn)
            for _ in range(layers)
        ]
        # Domain d holds ceil((hist - d) / 2) of each request's historical pages.
        self.domain_pages = torch.stack([
            torch.div(self.hist - d + 1, 2, rounding_mode="floor").clamp_(min=0) for d in range(2)
        ])
        sched = torch.zeros((2, self.batch + 1), dtype=torch.int32, device=device)
        sched[:, 1:] = torch.cumsum(self.domain_pages, dim=1)
        self.dom_sched = sched

    def valid_mask(self) -> torch.Tensor:
        """[batch, 8*H, max_pages] bool: historical pages each query row scores."""
        t = torch.arange(QLEN, device=self.seq_lens.device)
        local = torch.div(self.seq_lens[:, None] - QLEN + t[None, :], PAGE, rounding_mode="floor")
        local = local[:, :, None].expand(-1, -1, self.heads).reshape(self.batch, QLEN * self.heads)
        return torch.arange(self.max_pages, device=local.device)[None, None, :] < local[:, :, None]


def choose_tile_pages(pages: torch.Tensor, sm_count: int, resident_ctas: int) -> int:
    """The SM107 one-head kernel's choose_tile_pages over both domains' page counts."""
    def tiles(t):
        return int(torch.div(pages + t - 1, t, rounding_mode="floor").sum())

    def bucket(t):
        return min(t + (1 if t > MAX_PAGES_PER_CTA and t & 1 else 0), MAX_TILE_PAGES)

    total = tiles(1)
    if total == 0:
        return 1
    if tiles(MAX_TILE_PAGES) > resident_ctas:
        return min(BANDWIDTH_TILE_PAGES, MAX_TILE_PAGES)
    t = bucket(max(math.ceil(math.sqrt(total / sm_count)), 1))
    while tiles(t) > resident_ctas:
        t = bucket(t + 1)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, nargs="+", default=[4, 2, 1], choices=(1, 2, 4))
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--l2-multiple", type=float, default=2.0)
    ap.add_argument("--max-layers", type=int, default=2048)
    ap.add_argument("--cases", nargs="*", default=None, help="subset of case names")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    sm_count, l2 = props.multi_processor_count, int(props.L2_cache_size)
    ext = load_locality_ext()
    if int(ext.domain_count(0)) != 2:
        raise RuntimeError(f"this experiment needs 2 locality domains, got {int(ext.domain_count(0))}")
    granule = int(ext.page_bytes(0))
    stripe_bytes = granule * PAGE_BYTES // math.gcd(granule, PAGE_BYTES)
    chunks_per_stripe, pps = stripe_bytes // granule, stripe_bytes // PAGE_BYTES

    reported = ext.sm_domains(0)[:sm_count]
    smids = torch.arange(sm_count, dtype=torch.int32)
    rule = (smids >> 1) & 1
    known = reported >= 0
    if int(known.sum()) == 0 or not torch.equal(reported[known], rule[known]):
        raise RuntimeError("driver SM domains do not follow (smid >> 1) & 1")
    sm_domains = torch.where(known, reported, rule)
    sm_table = [int(d) for d in sm_domains.tolist()]
    counts = [sm_table.count(d) for d in range(2)]

    from fmha_sm100 import jit
    from src.sm100.q8kv4_indexer_decode_local import Q8KV4DecodeIndexerLocalSm100

    env = {
        "gpu": props.name, "capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": sm_count, "l2_bytes": l2, "granule": granule, "pages_per_stripe": pps,
        "sms_per_domain": counts, "smids_filled_by_rule": smids[~known].tolist(),
        "harness_sha": sha(Path(__file__)), "torch": torch.__version__, "cutlass": cutlass.__version__,
    }
    print(json.dumps(env), flush=True)

    cases = benchmark_cases("full")
    results, compiled_c = [], {}
    for heads in args.heads:
        gen = torch.Generator().manual_seed(1234)
        mod_c = jit._load_module("q8kv4_indexer_decode_sm107_parity") if heads == 1 else None
        resident_ctas = sm_count * int(mod_c.q8kv4_indexer_parity_resident_ctas_per_sm()) if mod_c else 0
        bits = sum(d << sm for sm, d in enumerate(sm_table))
        masks = [(bits >> (64 * i)) & (2**64 - 1) for i in range(4)]
        masks = [m - 2**64 if m >= 2**63 else m for m in masks]
        for dc in cases:
            if args.cases and dc.name not in args.cases:
                continue
            seq_lens = make_seq_lens(dc)
            hist = torch.div(seq_lens - 1, PAGE, rounding_mode="floor")
            layer_bytes = int(hist.sum()) * PAGE_BYTES
            layers = max(2, min(args.max_layers, math.ceil(args.l2_multiple * l2 / layer_bytes)))
            max_pages = math.ceil(int(seq_lens.max()) / PAGE)
            per_domain = layers * dc.batch_size * ((max_pages + 1) // 2)
            pool_pages = math.ceil(per_domain / pps) * (2 * pps)
            pool_a = random_pool(pool_pages, device)
            raw_c = ext.alloc_striped(pool_pages * PAGE_BYTES, chunks_per_stripe, 0)
            pool_c = raw_c.view(pool_pages, PAGE, PAGE_BYTES // PAGE)
            pool_c.copy_(pool_a)
            case = ParityCase(heads, seq_lens, layers, pool_pages, pps, device, gen)

            wrappers = []
            for layer in range(layers):
                w = qi.BatchDecodeIndexerQ8KV4Wrapper(num_heads=heads)
                w.plan(case.block_tables[layer], case.seq_lens)
                wrappers.append(w)
            scores_c = [torch.empty((dc.batch_size, QLEN * heads, max_pages), dtype=torch.float32,
                                    device=device) for _ in range(layers)]
            # One counter pair per launch; launch l zeroes launch l-1's pair.
            counters = torch.zeros((layers, 2), dtype=torch.int32, device=device)
            extra = {}
            if heads == 1:
                torch.cuda.synchronize()
                tile_c = choose_tile_pages(case.domain_pages.cpu(), sm_count, resident_ctas)
                tiles = torch.div(case.domain_pages + tile_c - 1, tile_c, rounding_mode="floor")
                prefix = torch.zeros((2, dc.batch_size + 1), dtype=torch.int32, device=device)
                prefix[:, 1:] = torch.cumsum(tiles, dim=1)
                extra = {"tile_pages_c": tile_c, "domain_tiles": prefix[:, -1].tolist()}
                min_claims = resident_ctas

                def run_c():
                    s = torch.cuda.current_stream(device).cuda_stream
                    for layer in range(layers):
                        mod_c.q8kv4_indexer_parity_run(
                            case.qs[layer], pool_c, case.block_tables[layer], case.seq_lens, prefix,
                            counters[layer], counters[(layer - 1) % layers], tile_c, *masks, sm_count,
                            scores_c[layer], s)
            else:
                domain_ctas = (2 * counts[0], 2 * counts[1])
                min_claims = sum(domain_ctas)
                if heads not in compiled_c:
                    kernel = Q8KV4DecodeIndexerLocalSm100(
                        sm_count=sm_count, num_heads=heads, domain_ctas=domain_ctas, sm_domains=sm_table)
                    compiled_c[heads] = cute.compile(
                        kernel,
                        *qi._cute_arguments(
                            (case.qs[0], pool_c),
                            (case.seq_lens, scores_c[0], case.block_tables[0], case.dom_sched,
                             counters[0], counters[1])),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi --opt-level 2",
                    )
                run_c_kernel = compiled_c[heads]

                def run_c():
                    for layer in range(layers):
                        run_c_kernel(case.qs[layer], pool_c, case.seq_lens, scores_c[layer],
                                     case.block_tables[layer], case.dom_sched,
                                     counters[layer], counters[(layer - 1) % layers])

            def run_a():
                for layer in range(layers):
                    wrappers[layer]._launch_scores(case.qs[layer], pool_a)

            run_a()
            run_c()
            torch.cuda.synchronize()
            mask = case.valid_mask()
            mismatched = topk_mismatched = 0
            for layer in range(layers):
                sa = wrappers[layer]._scores.view(torch.int32)
                mismatched += int((sa != scores_c[layer].view(torch.int32))[mask].sum())
                rows = wrappers[layer]._num_valid_pages
                out_a = torch.empty((dc.batch_size * QLEN, heads, 16), dtype=torch.int32, device=device)
                out_c = torch.empty_like(out_a)
                qi._topk_select(wrappers[layer]._scores, rows, out_a, use_pdl=False)
                qi._topk_select(scores_c[layer], rows, out_c, use_pdl=False)
                topk_mismatched += int((out_a != out_c).sum())

            a_us, c_us = [], []
            for rnd in range(args.rounds):
                # Alternate which arm goes first.
                order = ((c_us, run_c), (a_us, run_a)) if rnd % 2 else ((a_us, run_a), (c_us, run_c))
                for acc, fn in order:
                    acc.append(time_graph(fn, layers, args.warmup, args.iters))
            torch.cuda.synchronize()
            held = counters.sum(dim=1).tolist()
            if any(held[:-1]) or held[-1] < min_claims:
                raise RuntimeError(f"unexpected counter state: {held[:4]}...{held[-2:]}")
            a_med, c_med = statistics.median(a_us), statistics.median(c_us)
            row = {
                "heads": heads, "case": dc.name, "batch": dc.batch_size, "seq": dc.nominal_seq_len,
                "weight": dc.weight, "layers": layers,
                "index_k_mb_per_launch": round(layer_bytes / 2**20, 1), **extra,
                "a_us": round(a_med, 3), "c_us": round(c_med, 3),
                "a_rounds": [round(x, 3) for x in a_us], "c_rounds": [round(x, 3) for x in c_us],
                "latency_reduction_pct": round(100 * (a_med - c_med) / a_med, 2),
                "score_mismatches": mismatched, "valid_scores_checked": int(mask.sum()) * layers,
                "topk_mismatches": topk_mismatched,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            del pool_a, pool_c, raw_c, wrappers, scores_c, case
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    total_weight = sum(dc.weight for dc in cases)
    summary = {}
    for heads in args.heads:
        rows = [r for r in results if r["heads"] == heads]
        if len(rows) != len(cases):
            summary[heads] = {"partial_cases": len(rows)}
            continue
        wa = sum(r["weight"] * r["a_us"] for r in rows) / total_weight
        wc = sum(r["weight"] * r["c_us"] for r in rows) / total_weight
        summary[heads] = {
            "weighted_mean_a_us": round(wa, 3), "weighted_mean_c_us": round(wc, 3),
            "weighted_reduction_pct": round(100 * (wa - wc) / wa, 2),
            "cases_faster": sum(r["c_us"] < r["a_us"] for r in rows),
            "all_scores_match": all(r["score_mismatches"] == 0 for r in rows),
            "all_topk_match": all(r["topk_mismatches"] == 0 for r in rows),
        }
    print("SUMMARY", json.dumps(summary), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"env": env, "args": vars(args) | {"out": str(args.out)},
                                    "results": results, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
