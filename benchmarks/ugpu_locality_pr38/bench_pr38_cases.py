#!/usr/bin/env python3
"""A vs C (K-localized kernel) on the repository's 28 decode indexer cases, PR #38 locality.

C's index-K pool comes from vllm-project/MSA PR #38's alloc_locality_striped (locality
pages alternate between the two domains); the steered kernel takes each CTA's domain
from PR #38's sm_locality_domains table. Pages that straddle two locality pages are
left unused, and every other physical page belongs to the domain of its locality page.

Adapted from bench_parity_cases.py (msa-indexer-ugpu-locality): A is the 736f507 wrapper
(FP16-K kernel) on an ordinary pool; C is q8kv4_indexer_decode_localized on the striped pool.


C-parity: block tables put logical page j of every request in locality domain
j % 2 (striped pool); the parity kernel scores each domain's parity of pages.

Cases, weights and per-request lengths are copied verbatim from
vllm-project/MSA dev benchmarks/bench_q8kv4_decode_indexer.py (itself nv_dev
benchmarks/inference/msa_v1/decode/cases.py). Every case rotates through enough
layers with disjoint pages that one replay reads at least ``--l2-multiple`` x L2,
as the repository benchmark requires (reuse distance >= 2 x L2). Times are the
score kernel only (A: dev Q8KV4 decode CuTe kernel on an ordinary pool; C:
steered kernel on the localized pool); top-k is identical for both and is only
used for the correctness check.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import bench_indexer_locality as b
from pr38_locality import load_pr38_locality

# ---- vllm-project/MSA dev benchmarks/bench_q8kv4_decode_indexer.py (nv_dev decode/cases.py) ----
DEFAULT_SEED = 1701
BATCH_SIZES = (8, 32, 64, 128)
SEQ_LENGTHS = (1_000, 4_000, 5_000, 10_000, 50_000, 100_000, 200_000)
Q_LEN_PER_REQ = 8
_WEIGHTS = {
    1_000: (30, 90, 180, 300),
    4_000: (40, 120, 240, 400),
    5_000: (50, 150, 300, 500),
    10_000: (140, 350, 490, 420),
    50_000: (500, 800, 500, 200),
    100_000: (1260, 980, 420, 140),
    200_000: (910, 350, 112, 28),
}


@dataclass(frozen=True)
class DecodeBenchmarkCase:
    batch_size: int
    nominal_seq_len: int
    weight: int
    q_len_per_req: int = Q_LEN_PER_REQ
    seed: int = DEFAULT_SEED

    @property
    def name(self) -> str:
        return f"b{self.batch_size}_s{self.nominal_seq_len}_q{self.q_len_per_req}"


FULL_CASES = tuple(
    DecodeBenchmarkCase(batch_size=bs, nominal_seq_len=s, weight=_WEIGHTS[s][i])
    for s in SEQ_LENGTHS
    for i, bs in enumerate(BATCH_SIZES)
)


def make_seq_lens(case: DecodeBenchmarkCase) -> torch.Tensor:
    generator = torch.Generator().manual_seed(case.seed + case.batch_size * 1009 + case.nominal_seq_len)
    pair_count = case.batch_size // 2
    minimum = max(1, case.nominal_seq_len // 20)
    maximum = max(minimum, case.nominal_seq_len // 4)
    deltas = torch.randint(minimum, maximum + 1, (pair_count,), generator=generator, dtype=torch.int64)
    parts = [case.nominal_seq_len - deltas, case.nominal_seq_len + deltas]
    if case.batch_size % 2:
        parts.append(torch.tensor([case.nominal_seq_len], dtype=torch.int64))
    values = torch.cat(parts)
    values = values[torch.randperm(case.batch_size, generator=generator)]
    assert int(values.sum()) == case.batch_size * case.nominal_seq_len
    assert int(values.min()) >= case.q_len_per_req
    return values.to(torch.int32)


assert len(FULL_CASES) == 28 and sum(c.weight for c in FULL_CASES) == 10_000
# ---- end of copied section ----


class ParityCase:
    """Per-request lengths; logical page j of every request lives in domain j % 2."""

    def __init__(self, heads, seq_lens, layers, page_domain, device, gen):
        self.heads, self.layers = heads, layers
        self.batch = int(seq_lens.numel())
        self.seq_lens = seq_lens.to(device)
        self.max_pages = math.ceil(int(seq_lens.max()) / b.PAGE)
        self.hist = torch.div(self.seq_lens - 1, b.PAGE, rounding_mode="floor").clamp_(max=self.max_pages)
        pages = torch.arange(page_domain.numel())
        domain = page_domain
        even_cols = (self.max_pages + 1) // 2
        odd_cols = self.max_pages // 2
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
        assert torch.equal(page_domain[bt], (torch.arange(self.max_pages) % 2).expand_as(bt))
        self.block_tables = bt.to(torch.int32).to(device).contiguous()
        self.qs = [
            torch.randn(self.batch * b.QLEN, heads, 128, device=device).mul_(0.5).to(torch.float8_e4m3fn)
            for _ in range(layers)
        ]
        # Domain d holds ceil((hist - d) / 2) of each request's historical pages.
        sched = torch.zeros((2, self.batch + 1), dtype=torch.int32, device=device)
        for d in range(2):
            count = torch.div(self.hist - d + 1, 2, rounding_mode="floor").clamp_(min=0)
            sched[d, 1:] = torch.cumsum(count, 0)
        self.dom_sched = sched


def pr38_page_domains(pool_pages: int, locality_page: int) -> torch.Tensor:
    """Domain of each 9216-byte K page in a PR #38 striped buffer; -1 if it straddles."""
    start = torch.arange(pool_pages, dtype=torch.int64) * b.PAGE_BYTES
    first = start // locality_page
    last = (start + b.PAGE_BYTES - 1) // locality_page
    domain = first % 2
    domain[first != last] = -1
    return domain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--l2-multiple", type=float, default=2.0)
    ap.add_argument("--max-layers", type=int, default=2048)
    ap.add_argument("--cases", nargs="*", default=None, help="subset of case names (smoke)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    l2 = int(props.L2_cache_size)
    ext = load_pr38_locality(b.REPO)
    num_domains = int(ext.locality_domain_count(0))
    if num_domains != 2:
        raise RuntimeError(f"expected 2 memory locality domains, the device reports {num_domains}")
    granule = int(ext.locality_page_bytes(0))
    sm_table = ext.sm_locality_domains(0)  # int32 per %smid, -1 where unknown
    # PR #38's driver split (cuDevSmResourceSplit, coscheduled pairs) can leave some SMs in
    # neither domain group (-1). Fill only those from the %smid rule, and only if the rule
    # agrees with the table on every SM the table does cover.
    smids = torch.arange(sm_table.numel(), dtype=torch.int32)
    known = sm_table >= 0
    rule = (smids >> 1) & 1
    rule_agrees = bool(torch.equal(sm_table[known], rule[known]))
    if not rule_agrees:
        raise RuntimeError("the %smid rule disagrees with PR #38's SM table")
    filled = torch.where(known, sm_table, rule)
    counts = [int((filled == d).sum()) for d in range(2)]
    if sum(counts) != sm_count:
        raise RuntimeError(f"SM table covers {sum(counts)} SMs, the device has {sm_count}")
    sm_dom_dev = filled.to(torch.int32).to(device)
    from src.sm100.q8kv4_indexer_decode import Q8KV4DecodeIndexerSm100
    ctas_per_sm = {h: Q8KV4DecodeIndexerSm100(sm_count=sm_count, num_heads=h).target_ctas_per_sm for h in args.heads}
    print("CTAS_PER_SM", ctas_per_sm, flush=True)
    from src.sm100.q8kv4_indexer_decode_localized import Q8KV4DecodeIndexerLocalizedSm100

    api = b.REPO / "python/fmha_sm100/prefill_q8kv4/csrc/api"
    env = {
        "gpu": props.name, "sm_count": sm_count, "l2_bytes": l2, "locality_page_bytes": granule,
        "num_domains": num_domains, "sm_per_domain": counts, "smid_table_size": int(sm_table.numel()),
        "smid_rule_agrees": rule_agrees, "sm_per_domain_pr38_table": [int((sm_table == d).sum()) for d in range(2)],
        "smids_filled_by_rule": smids[~known].tolist(), "ctas_per_sm": ctas_per_sm,
        "harness_sha": b.sha(Path(__file__)), "base_harness_sha": b.sha(Path(b.__file__)),
        "pr38_locality_domains_sha": b.sha(api / "locality_domains.cu"),
        "localized_sha": b.sha(b.REPO / "python/fmha_sm100/cute/src/sm100/q8kv4_indexer_decode_localized.py"),
        "base_kernel_sha": b.sha(b.REPO / "python/fmha_sm100/cute/src/sm100/q8kv4_indexer_decode.py"),
        "cutlass": b.cutlass.__version__, "torch": torch.__version__,
    }
    print(json.dumps(env), flush=True)

    compiled_c = {}
    results = []
    for heads in args.heads:
        gen = torch.Generator().manual_seed(1234)
        domain_ctas = (ctas_per_sm[heads] * counts[0], ctas_per_sm[heads] * counts[1])
        for dc in FULL_CASES:
            if args.cases and dc.name not in args.cases:
                continue
            seq_lens = make_seq_lens(dc)
            hist = torch.div(seq_lens - 1, b.PAGE, rounding_mode="floor")
            layer_bytes = int(hist.sum()) * b.PAGE_BYTES
            layers = max(2, min(args.max_layers, math.ceil(args.l2_multiple * l2 / layer_bytes)))
            max_pages = math.ceil(int(seq_lens.max()) / b.PAGE)
            per_domain = layers * dc.batch_size * ((max_pages + 1) // 2)
            # Each domain gets at least granule // PAGE_BYTES - 1 whole K pages per locality page.
            whole_per_locality_page = granule // b.PAGE_BYTES - 1
            stripes = math.ceil(per_domain / whole_per_locality_page) + 1
            nbytes = stripes * 2 * granule
            pool_pages = nbytes // b.PAGE_BYTES
            page_domain = pr38_page_domains(pool_pages, granule)

            pool_a = b.random_pool(pool_pages, device)
            raw_c = ext.alloc_locality_striped(nbytes, 0)
            pool_c = raw_c[: pool_pages * b.PAGE_BYTES].view(pool_pages, b.PAGE, b.PAGE_BYTES // b.PAGE)
            pool_c.copy_(pool_a)
            case = ParityCase(heads, seq_lens, layers, page_domain, device, gen)

            wrappers = []
            for layer in range(layers):
                w = b.qi.BatchDecodeIndexerQ8KV4Wrapper(num_heads=heads)
                w.plan(case.block_tables[layer], case.seq_lens)
                wrappers.append(w)
            scores_c = [
                torch.empty((dc.batch_size, b.QLEN * heads, max_pages), dtype=torch.float32, device=device)
                for _ in range(layers)
            ]
            # One counter pair per launch; launch l zeroes launch l-1's pair.
            counters = torch.zeros((layers, 2), dtype=torch.int32, device=device)
            if heads not in compiled_c:
                kernel = Q8KV4DecodeIndexerLocalizedSm100(
                    sm_count=sm_count, num_heads=heads, domain_ctas=domain_ctas
                )
                compiled_c[heads] = b.cute.compile(
                    kernel,
                    *b.qi._cute_arguments(
                        (case.qs[0], pool_c),
                        (case.seq_lens, scores_c[0], case.block_tables[0], case.dom_sched,
                         counters[0], counters[1], sm_dom_dev),
                    ),
                    b.cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                    options="--enable-tvm-ffi --opt-level 2",
                )
            run_c_kernel = compiled_c[heads]

            def run_a():
                for layer in range(layers):
                    wrappers[layer]._launch_scores(case.qs[layer], pool_a)

            def run_c():
                for layer in range(layers):
                    run_c_kernel(case.qs[layer], pool_c, case.seq_lens, scores_c[layer],
                                 case.block_tables[layer], case.dom_sched,
                                 counters[layer], counters[(layer - 1) % layers], sm_dom_dev)

            run_a()
            run_c()
            torch.cuda.synchronize()
            mask = b.valid_mask(case, device)
            mismatched = topk_mismatched = 0
            for layer in range(layers):
                sa = wrappers[layer]._scores.view(torch.int32)
                sc = scores_c[layer].view(torch.int32)
                mismatched += int((sa != sc)[mask].sum())
                rows = wrappers[layer]._num_valid_pages
                out_a = torch.empty((dc.batch_size * b.QLEN, heads, 16), dtype=torch.int32, device=device)
                out_c = torch.empty_like(out_a)
                b.qi._topk_select(wrappers[layer]._scores, rows, out_a, use_pdl=False)
                b.qi._topk_select(scores_c[layer], rows, out_c, use_pdl=False)
                topk_mismatched += int((out_a != out_c).sum())

            a_us, c_us = [], []
            for _ in range(args.rounds):
                a_us.append(b.time_graph(run_a, layers, args.warmup, args.iters))
                c_us.append(b.time_graph(run_c, layers, args.warmup, args.iters))
            torch.cuda.synchronize()
            # After full sweeps only the last launch's pair is still set, and it
            # must hold at least one claim per CTA (a steal adds one).
            held = counters.sum(dim=1).tolist()
            if any(held[:-1]) or held[-1] < sum(domain_ctas):
                raise RuntimeError(f"unexpected counter state: {held[:4]}...{held[-2:]}")
            a_med, c_med = statistics.median(a_us), statistics.median(c_us)
            row = {
                "heads": heads, "case": dc.name, "batch": dc.batch_size, "seq": dc.nominal_seq_len,
                "weight": dc.weight, "layers": layers,
                "index_k_mb_per_launch": round(layer_bytes / 2**20, 1),
                "reuse_over_l2": round(layers * layer_bytes / l2, 2),
                "domain_pages": [int(case.dom_sched[d, -1]) for d in range(2)],
                "a_us": round(a_med, 3), "c_us": round(c_med, 3),
                "a_rounds": [round(x, 3) for x in a_us], "c_rounds": [round(x, 3) for x in c_us],
                "speedup_c_over_a": round(a_med / c_med, 4),
                "latency_reduction_pct": round(100 * (a_med - c_med) / a_med, 2),
                "score_mismatches": mismatched, "valid_scores_checked": int(mask.sum()) * layers,
                "topk_mismatches": topk_mismatched,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            del pool_a, pool_c, raw_c, wrappers, scores_c, case
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    summary = {}
    for heads in args.heads:
        rows = [r for r in results if r["heads"] == heads]
        if len(rows) != len(FULL_CASES):
            summary[heads] = {"partial_cases": len(rows)}
            continue
        wa = sum(r["weight"] * r["a_us"] for r in rows) / 10_000
        wc = sum(r["weight"] * r["c_us"] for r in rows) / 10_000
        summary[heads] = {
            "weighted_mean_a_us": round(wa, 3), "weighted_mean_c_us": round(wc, 3),
            "weighted_speedup": round(wa / wc, 4),
            "weighted_reduction_pct": round(100 * (wa - wc) / wa, 2),
            "cases_faster": sum(r["c_us"] < r["a_us"] for r in rows),
            "cases_slower_gt_1pct": sum(r["latency_reduction_pct"] < -1 for r in rows),
            "all_scores_match": all(r["score_mismatches"] == 0 for r in rows),
            "all_topk_match": all(r["topk_mismatches"] == 0 for r in rows),
        }
    print("SUMMARY", json.dumps(summary), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"env": env, "args": vars(args) | {"out": str(args.out)},
                                    "results": results, "summary": summary}, indent=2))
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
