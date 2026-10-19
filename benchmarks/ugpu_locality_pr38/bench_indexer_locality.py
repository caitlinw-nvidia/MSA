#!/usr/bin/env python3
"""Q8KV4 decode indexer: ordinary pool + current kernel (A) vs localized pool + steered kernel (C).

The index-K pool is assumed to be already localized. Arm C allocates it as one
virtual range whose stripes alternate between the two locality domains; a stripe
holds a whole number of 9216-byte pages, so page p lives in domain
``(p // pages_per_stripe) % 2``. Block tables are random pages of the pool, so
every request's pages split about evenly. Several layers with disjoint pages
keep the working set larger than L2, as in a real decode step.

Both arms hold identical bytes. Scores are compared bitwise on every valid
(request, query, page) and the top-k on every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ["MSA_REPO"]).resolve()
sys.path[:0] = [str(REPO / "python" / "fmha_sm100" / "cute"), str(REPO / "python")]

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import q8_indexer_interface as qi  # noqa: E402

PAGE = 128
PAGE_BYTES = qi._NVFP4_PAGE_BYTES  # 9216
QLEN = 8


def load_locality_ext():
    from torch.utils.cpp_extension import load

    build = Path(os.environ.get("LOCALITY_EXT_BUILD", HERE / "build_locality_ext"))
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="indexer_locality_ext",
        sources=[str(HERE / "locality_ext.cu")],
        extra_ldflags=["-lcuda"],
        build_directory=str(build),
        verbose=False,
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def random_pool(pages: int, device) -> torch.Tensor:
    pool = torch.randint(0, 256, (pages, PAGE, PAGE_BYTES // PAGE), dtype=torch.uint8, device=device)
    flat = pool.view(pages, PAGE_BYTES)
    # Scales: finite positive E4M3 (exponent field 4..9), so no NaN scores.
    scales = torch.randint(0x20, 0x48, (pages, PAGE_BYTES - PAGE * 64), dtype=torch.uint8, device=device)
    flat[:, PAGE * 64 :] = scales
    return pool


class Case:
    def __init__(self, heads, batch, kv, layers, pool_pages, pages_per_stripe, device, gen):
        self.heads, self.batch, self.kv, self.layers = heads, batch, kv, layers
        self.max_pages = math.ceil(kv / PAGE)
        self.hist = (kv - 1) // PAGE
        self.seq_lens = torch.full((batch,), kv, dtype=torch.int32, device=device)
        need = layers * batch * self.max_pages
        perm = torch.randperm(pool_pages, generator=gen, device="cpu")[:need].to(device)
        self.block_tables = perm.view(layers, batch, self.max_pages).to(torch.int32).contiguous()
        self.qs = [
            torch.randn(batch * QLEN, heads, 128, device=device).mul_(0.5).to(torch.float8_e4m3fn)
            for _ in range(layers)
        ]
        self.pps = pages_per_stripe

    def domain_plan(self, layer: int):
        """Per-domain page lists sorted by (request, logical page) and their prefix sums."""

        bt = self.block_tables[layer]
        b, mp = bt.shape
        logical = torch.arange(mp, device=bt.device, dtype=torch.int32).expand(b, mp)
        valid = logical < self.hist
        domain = (bt // self.pps) % 2
        cap = b * mp
        sched = torch.zeros((2, b + 1), dtype=torch.int32, device=bt.device)
        lists_l = torch.zeros((2, cap), dtype=torch.int32, device=bt.device)
        lists_p = torch.zeros((2, cap), dtype=torch.int32, device=bt.device)
        for d in range(2):
            m = valid & (domain == d)
            sched[d, 1:] = torch.cumsum(m.sum(dim=1, dtype=torch.int32), 0)
            n = int(sched[d, -1])
            lists_l[d, :n] = logical[m]  # row-major: request then logical page
            lists_p[d, :n] = bt[m]
        return sched, lists_l, lists_p


def valid_mask(case: Case, device) -> torch.Tensor:
    """[batch, 8*H, max_pages] bool: historical pages each query row scores."""

    t = torch.arange(QLEN, device=device)
    pos = case.seq_lens[:, None] - QLEN + t[None, :]
    local = torch.div(pos, PAGE, rounding_mode="floor")  # [B, 8]
    local = local[:, :, None].expand(-1, -1, case.heads).reshape(case.batch, QLEN * case.heads)
    pages = torch.arange(case.max_pages, device=device)
    return pages[None, None, :] < local[:, :, None]


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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / layers)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--kv", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    ap.add_argument("--batch", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--l2-multiple", type=float, default=4.0)
    ap.add_argument("--max-layers", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    cap = torch.cuda.get_device_capability(device)
    sm_count = props.multi_processor_count
    l2 = int(getattr(props, "L2_cache_size", 0)) or 0

    ext = load_locality_ext()
    num_domains = int(ext.domain_count(0))
    if num_domains != 2:
        raise RuntimeError(f"expected 2 locality domains, got {num_domains}")
    granule = int(ext.page_bytes(0))
    # Smallest stripe holding a whole number of index pages.
    stripe_bytes = granule * PAGE_BYTES // math.gcd(granule, PAGE_BYTES)
    chunks_per_stripe = stripe_bytes // granule
    pps = stripe_bytes // PAGE_BYTES
    sm_domains = ext.sm_domains(0)
    # cuDevSmResourceSplit leaves some SMs in its remainder group, without a
    # domain. On Rubin the reported SMs follow domain = (smid >> 1) & 1 (TPCs
    # alternate); fill the remainder with that rule only if every reported SM
    # obeys it.
    smids = torch.arange(sm_domains.numel(), dtype=torch.int32)
    rule = (smids >> 1) & 1
    known = sm_domains >= 0
    unknown_smids = smids[~known].tolist()
    if not torch.equal(sm_domains[known], rule[known]):
        raise RuntimeError("reported SM domains do not follow (smid >> 1) & 1")
    sm_domains = torch.where(known, sm_domains, rule)
    counts = [int((sm_domains == d).sum()) for d in range(2)]
    if sum(counts) != sm_count:
        raise RuntimeError(f"sm_domains covers {counts} SMs, device has {sm_count}")
    domain_ctas = (2 * counts[0], 2 * counts[1])
    sm_domains_dev = sm_domains.to(device)
    counters = torch.zeros(4, dtype=torch.int32, device=device)

    env = {
        "gpu": props.name,
        "capability": cap,
        "sm_count": sm_count,
        "l2_bytes": l2,
        "uuid": str(getattr(props, "uuid", "")),
        "torch": torch.__version__,
        "cutlass": cutlass.__version__,
        "cutlass_path": cutlass.__file__,
        "cute_path": cute.__file__,
        "cuda_version": str(cutlass.CUDA_VERSION),
        "python": sys.executable,
        "host": platform.node(),
        "repo": str(REPO),
        "repo_head": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "steered_sha": sha(REPO / "python/fmha_sm100/cute/src/sm100/q8kv4_indexer_decode_steered.py"),
        "base_sha": sha(REPO / "python/fmha_sm100/cute/src/sm100/q8kv4_indexer_decode.py"),
        "harness_sha": sha(Path(__file__)),
        "ext_sha": sha(HERE / "locality_ext.cu"),
        "num_domains": num_domains,
        "granule_bytes": granule,
        "stripe_bytes": stripe_bytes,
        "pages_per_stripe": pps,
        "sm_per_domain": counts,
        "smids_filled_by_rule": unknown_smids,
        "domain_ctas": domain_ctas,
    }
    print(json.dumps(env, indent=2), flush=True)

    compiled_c = {}
    results = []
    gen = torch.Generator().manual_seed(1234)
    for heads in args.heads:
        for kv in args.kv:
            for batch in args.batch:
                max_pages = math.ceil(kv / PAGE)
                layer_bytes = batch * ((kv - 1) // PAGE) * PAGE_BYTES
                layers = max(2, min(args.max_layers, math.ceil(args.l2_multiple * l2 / layer_bytes)))
                pool_pages = layers * batch * max_pages
                pool_pages = math.ceil(pool_pages / (2 * pps)) * (2 * pps)

                pool_a = random_pool(pool_pages, device)
                raw_c = ext.alloc_striped(pool_pages * PAGE_BYTES, chunks_per_stripe, 0)
                pool_c = raw_c.view(pool_pages, PAGE, PAGE_BYTES // PAGE)
                pool_c.copy_(pool_a)
                case = Case(heads, batch, kv, layers, pool_pages, pps, device, gen)

                # Arm A: the dev wrapper's CuTe score kernel on the ordinary pool.
                wrappers = []
                for layer in range(layers):
                    w = qi.BatchDecodeIndexerQ8KV4Wrapper(num_heads=heads)
                    w.plan(case.block_tables[layer], case.seq_lens)
                    wrappers.append(w)

                def run_a():
                    for layer in range(layers):
                        wrappers[layer]._launch_scores(case.qs[layer], pool_a)

                # Arm C: steered kernel on the localized pool.
                t0 = time.perf_counter()
                plans = [case.domain_plan(layer) for layer in range(layers)]
                torch.cuda.synchronize()
                plan_ms = (time.perf_counter() - t0) * 1000 / layers
                scores_c = [
                    torch.empty((batch, QLEN * heads, max_pages), dtype=torch.float32, device=device)
                    for _ in range(layers)
                ]
                if heads not in compiled_c:
                    kernel = Q8KV4DecodeIndexerSteeredSm100(
                        sm_count=sm_count, num_heads=heads, domain_ctas=domain_ctas
                    )
                    sched, ll, lp = plans[0]
                    compiled_c[heads] = cute.compile(
                        kernel,
                        *qi._cute_arguments(
                            (case.qs[0], pool_c),
                            (case.seq_lens, scores_c[0], sched, ll, lp, sm_domains_dev, counters),
                        ),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi --opt-level 2",
                    )
                run_c_kernel = compiled_c[heads]

                def run_c():
                    for layer in range(layers):
                        sched, ll, lp = plans[layer]
                        run_c_kernel(
                            case.qs[layer], pool_c, case.seq_lens, scores_c[layer],
                            sched, ll, lp, sm_domains_dev, counters,
                        )

                # Correctness: bitwise scores on valid positions, top-k on every row.
                run_a()
                run_c()
                torch.cuda.synchronize()
                mask = valid_mask(case, device)
                mismatched = 0
                topk_mismatched = 0
                for layer in range(layers):
                    sa = wrappers[layer]._scores.view(torch.int32)
                    sc = scores_c[layer].view(torch.int32)
                    mismatched += int((sa != sc)[mask].sum())
                    rows = wrappers[layer]._num_valid_pages
                    out_a = torch.empty((batch * QLEN, heads, 16), dtype=torch.int32, device=device)
                    out_c = torch.empty_like(out_a)
                    qi._topk_select(wrappers[layer]._scores, rows, out_a, use_pdl=False)
                    qi._topk_select(scores_c[layer], rows, out_c, use_pdl=False)
                    topk_mismatched += int((out_a != out_c).sum())
                counters_after = counters.tolist()
                dom_pages = [int(plans[0][0][d, -1]) for d in range(2)]

                a_us, c_us = [], []
                for _ in range(args.rounds):
                    a_us.append(time_graph(run_a, layers, args.warmup, args.iters))
                    c_us.append(time_graph(run_c, layers, args.warmup, args.iters))
                if counters.tolist()[:3] != [0, 0, 0]:
                    raise RuntimeError(f"claim counters not reset: {counters.tolist()}")
                a_med, c_med = statistics.median(a_us), statistics.median(c_us)
                row = {
                    "heads": heads,
                    "kv": kv,
                    "batch": batch,
                    "layers": layers,
                    "working_set_mb": round(layers * layer_bytes / 2**20, 1),
                    "layer0_domain_pages": dom_pages,
                    "a_us": round(a_med, 3),
                    "c_us": round(c_med, 3),
                    "a_rounds": [round(x, 3) for x in a_us],
                    "c_rounds": [round(x, 3) for x in c_us],
                    "speedup_c_over_a": round(a_med / c_med, 4),
                    "latency_reduction_pct": round(100 * (a_med - c_med) / a_med, 2),
                    "score_mismatches": mismatched,
                    "valid_scores_checked": int(mask.sum()) * layers,
                    "topk_mismatches": topk_mismatched,
                    "counters_after_check": counters_after,
                    "plan_build_ms_per_layer_torch": round(plan_ms, 3),
                    "a_hbm_gbps": round(layer_bytes / (a_med * 1e-6) / 1e9, 1),
                    "c_hbm_gbps": round(layer_bytes / (c_med * 1e-6) / 1e9, 1),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
                del pool_a, pool_c, raw_c, wrappers, plans, scores_c, case
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"env": env, "args": vars(args) | {"out": str(args.out)}, "results": results}, indent=2))
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
