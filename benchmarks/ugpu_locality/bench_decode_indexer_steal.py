#!/usr/bin/env python3
"""Q8KV4 decode indexer, 4 index heads, mixed request lengths: plain vs localized.

  A: BatchDecodeIndexerQ8KV4Wrapper (Q8KV4DecodeIndexerSm100) on an ordinary pool.
  L: Q8KV4DecodeIndexerLocalSm100 (one claim per CTA) on the uGPU-striped pool.

Pool, block tables (logical page j in domain j % 2), layer rotation (one sweep reads at
least --l2-multiple x L2), CUDA-graph timing and the bitwise score / top-k checks are
those of bench_decode_indexer_locality.py. Cases model one DEP rank of MiniMax-M3
decode: batch 16 and 20, 8 query tokens per request (the repository decode cases),
lengths log-uniform in [4K, 200K] (two seeds), skewed (two 200K requests, the rest
8K) and equal 64K. Latency rounds run first (arm order alternates every round), then
the correctness checks on the outputs of the timed launches.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_decode_indexer_locality as loc  # noqa: E402  (sets up sys.path for the repo)

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import q8_indexer_interface as qi  # noqa: E402

HEADS, QLEN, PAGE = 4, loc.QLEN, loc.PAGE


def fan_cases():
    cases = []
    for batch in (16, 20):
        for seed in (0, 1):
            g = torch.Generator().manual_seed(1000 * batch + seed)
            u = torch.rand(batch, generator=g, dtype=torch.float64)
            lens = torch.exp(math.log(4096) + u * (math.log(200_000) - math.log(4096)))
            cases.append((f"b{batch}_loguni4k-200k_s{seed}", lens.round().to(torch.int32)))
        skew = torch.full((batch,), 8192, dtype=torch.int32)
        skew[:2] = 200_000
        g = torch.Generator().manual_seed(1000 * batch + 7)
        cases.append((f"b{batch}_skew2x200k", skew[torch.randperm(batch, generator=g)]))
        cases.append((f"b{batch}_eq64k", torch.full((batch,), 65536, dtype=torch.int32)))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--l2-multiple", type=float, default=2.0)
    ap.add_argument("--max-layers", type=int, default=2048)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    sm_count, l2 = props.multi_processor_count, int(props.L2_cache_size)
    ext = loc.load_locality_ext()
    if int(ext.domain_count(0)) != 2:
        raise RuntimeError(f"needs 2 locality domains, got {int(ext.domain_count(0))}")
    granule = int(ext.page_bytes(0))
    stripe_bytes = granule * loc.PAGE_BYTES // math.gcd(granule, loc.PAGE_BYTES)
    chunks_per_stripe, pps = stripe_bytes // granule, stripe_bytes // loc.PAGE_BYTES
    reported = ext.sm_domains(0)[:sm_count]
    smids = torch.arange(sm_count, dtype=torch.int32)
    rule = (smids >> 1) & 1
    known = reported >= 0
    if int(known.sum()) == 0 or not torch.equal(reported[known], rule[known]):
        raise RuntimeError("driver SM domains do not follow (smid >> 1) & 1")
    sm_table = [int(d) for d in torch.where(known, reported, rule).tolist()]
    counts = [sm_table.count(d) for d in range(2)]
    domain_ctas = (2 * counts[0], 2 * counts[1])

    from src.sm100.q8kv4_indexer_decode_local import Q8KV4DecodeIndexerLocalSm100

    kernel_path = HERE.parents[1] / "python/fmha_sm100/cute/src/sm100/q8kv4_indexer_decode_local.py"
    env = {
        "gpu": props.name, "capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": sm_count, "l2_bytes": l2, "granule": granule, "pages_per_stripe": pps,
        "sms_per_domain": counts, "domain_ctas": domain_ctas,
        "harness_sha": loc.sha(Path(__file__)), "local_kernel_sha": loc.sha(kernel_path),
        "torch": torch.__version__, "cutlass": cutlass.__version__,
    }
    print(json.dumps(env), flush=True)

    compiled = None
    results = []
    gen = torch.Generator().manual_seed(1234)
    for name, seq_lens in fan_cases():
        if args.cases and name not in args.cases:
            continue
        batch = int(seq_lens.numel())
        hist = torch.div(seq_lens - 1, PAGE, rounding_mode="floor")
        layer_bytes = int(hist.sum()) * loc.PAGE_BYTES
        layers = max(2, min(args.max_layers, math.ceil(args.l2_multiple * l2 / layer_bytes)))
        max_pages = math.ceil(int(seq_lens.max()) / PAGE)
        per_domain = layers * batch * ((max_pages + 1) // 2)
        pool_pages = math.ceil(per_domain / pps) * (2 * pps)
        pool_a = loc.random_pool(pool_pages, device)
        raw_c = ext.alloc_striped(pool_pages * loc.PAGE_BYTES, chunks_per_stripe, 0)
        pool_c = raw_c.view(pool_pages, PAGE, loc.PAGE_BYTES // PAGE)
        pool_c.copy_(pool_a)
        case = loc.ParityCase(HEADS, seq_lens, layers, pool_pages, pps, device, gen)

        wrappers = []
        for layer in range(layers):
            w = qi.BatchDecodeIndexerQ8KV4Wrapper(num_heads=HEADS)
            w.plan(case.block_tables[layer], case.seq_lens)
            wrappers.append(w)

        scores = [torch.empty((batch, QLEN * HEADS, max_pages), dtype=torch.float32, device=device)
                  for _ in range(layers)]
        counters = torch.zeros((layers, 2), dtype=torch.int32, device=device)
        if compiled is None:
            kernel = Q8KV4DecodeIndexerLocalSm100(
                sm_count=sm_count, num_heads=HEADS, domain_ctas=domain_ctas, sm_domains=sm_table)
            compiled = cute.compile(
                kernel,
                *qi._cute_arguments(
                    (case.qs[0], pool_c),
                    (case.seq_lens, scores[0], case.block_tables[0], case.dom_sched,
                     counters[0], counters[1])),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi --opt-level 2",
            )

        def run_a():
            for layer in range(layers):
                wrappers[layer]._launch_scores(case.qs[layer], pool_a)

        def run_l():
            for layer in range(layers):
                compiled(case.qs[layer], pool_c, case.seq_lens, scores[layer], case.block_tables[layer],
                         case.dom_sched, counters[layer], counters[(layer - 1) % layers])

        arms = {"A": run_a, "L": run_l}
        times = {k: [] for k in arms}
        keys = list(arms)
        for rnd in range(args.rounds):
            order = keys[rnd % 2:] + keys[:rnd % 2]
            for k in order:
                times[k].append(loc.time_graph(arms[k], layers, args.warmup, args.iters))
        torch.cuda.synchronize()

        # Correctness on the timed launches' outputs.
        held = counters.sum(dim=1).tolist()
        if any(held[:-1]) or held[-1] < sum(domain_ctas):
            raise RuntimeError(f"L: unexpected counter state {held[:4]}...{held[-2:]}")
        mask = case.valid_mask()
        mism = 0
        topk_mism = 0
        for layer in range(layers):
            sa = wrappers[layer]._scores
            rows = wrappers[layer]._num_valid_pages
            out_a = torch.empty((batch * QLEN, HEADS, 16), dtype=torch.int32, device=device)
            qi._topk_select(sa, rows, out_a, use_pdl=False)
            mism += int((sa.view(torch.int32) != scores[layer].view(torch.int32))[mask].sum())
            out_l = torch.empty_like(out_a)
            qi._topk_select(scores[layer], rows, out_l, use_pdl=False)
            topk_mism += int((out_a != out_l).sum())

        med = {k: statistics.median(v) for k, v in times.items()}
        row = {
            "case": name, "batch": batch, "seq_lens": seq_lens.tolist(),
            "total_hist_pages": int(hist.sum()), "layers": layers,
            "index_k_mb_per_launch": round(layer_bytes / 2**20, 1),
            "a_us": round(med["A"], 3), "l_us": round(med["L"], 3),
            "l_vs_a_pct": round(100 * (med["A"] - med["L"]) / med["A"], 2),
            "rounds": {k: [round(x, 3) for x in v] for k, v in times.items()},
            "score_mismatches": {"L": mism}, "topk_mismatches": {"L": topk_mism},
            "valid_scores_checked": int(mask.sum()) * layers,
            "claims_last_layer": {"L": counters[-1].tolist()},
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del pool_a, pool_c, raw_c, wrappers, scores, case
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if results:
        summ = {
            "a": round(statistics.mean(r["a_us"] for r in results), 3),
            "l": round(statistics.mean(r["l_us"] for r in results), 3),
            "mean_l_vs_a_pct": round(statistics.mean(r["l_vs_a_pct"] for r in results), 2),
            "all_scores_match": all(r["score_mismatches"]["L"] == 0 for r in results),
            "all_topk_match": all(r["topk_mismatches"]["L"] == 0 for r in results),
        }
        print("SUMMARY", json.dumps(summ), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"env": env, "args": vars(args) | {"out": str(args.out)},
                                        "results": results, "summary": summ}, indent=2))


if __name__ == "__main__":
    main()
