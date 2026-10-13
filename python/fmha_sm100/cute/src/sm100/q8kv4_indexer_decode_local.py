# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""uGPU-localized Q8KV4 decode indexer for two or four index heads (experiment).

On GPUs with two memory locality domains (Rubin uGPUs) the index-K pool is
assumed to be allocated so logical page ``j`` of every request lives in domain
``j % 2`` (page-interleaved, like DCP with ``cp_kv_cache_interleave_size`` equal
to the block size). The CTAs of domain ``d`` score logical pages ``d, d + 2,
d + 4, ...`` of every request: per-domain prefix sums of
``ceil((history_pages - d) / 2)`` split that work into contiguous slots, and a
CTA claims the next slot of its SM's domain from this launch's counter pair,
taking one of the other domain only once its own are gone. A CTA finds its
domain from ``%smid`` in an SM -> domain table compiled into the kernel (built
on the host from the driver's SM split, ``cuDevSmResourceSplit``). The loader
reads the ordinary page table at stride 2 and the score warp computes the
logical page. Each launch zeroes its predecessor's counters after the PDL wait.

Per-page math, pipelines and output layout are those of
``Q8KV4DecodeIndexerSm100``; the scores are bitwise identical.
"""

from __future__ import annotations

import enum

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

import cuda.bindings.driver as cuda

from src.sm100.q8kv4_indexer_decode import Q8KV4DecodeIndexerSm100

NUM_DOMAINS = 2
_TABLE_WORDS = 8  # SM ids 0..255


@dsl_user_op
def _smid(*, loc=None, ip=None) -> Int32:
    out = llvm.inline_asm(
        T.i32(), [], "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(out)


class _NamedBarrier(enum.IntEnum):
    TmemPtr = enum.auto()
    Score = enum.auto()
    Final = enum.auto()
    Claim = enum.auto()


class Q8KV4DecodeIndexerLocalSm100(Q8KV4DecodeIndexerSm100):
    """``Q8KV4DecodeIndexerSm100`` with each domain scoring its parity of logical pages."""

    def __init__(
        self,
        *,
        sm_count: int,
        num_heads: int,
        domain_ctas: tuple[int, int],
        sm_domains: list[int],
    ) -> None:
        """``sm_domains[s]``: locality domain (0 or 1) of SM id ``s``, for every SM."""
        super().__init__(sm_count=sm_count, num_heads=num_heads)
        if len(sm_domains) < sm_count or len(sm_domains) > 32 * _TABLE_WORDS:
            raise ValueError("sm_domains needs one entry per SM id (at most 256)")
        if any(d not in range(NUM_DOMAINS) for d in sm_domains):
            raise ValueError("sm_domains entries must be 0 or 1")
        words = [0] * _TABLE_WORDS
        for sm, domain in enumerate(sm_domains):
            words[sm >> 5] |= domain << (sm & 31)
        self.sm_domain_words = tuple(w - (1 << 32) if w >= (1 << 31) else w for w in words)
        if len(domain_ctas) != NUM_DOMAINS or min(domain_ctas) <= 0:
            raise ValueError("domain_ctas needs a positive CTA count per domain")
        if sum(domain_ctas) != self.grid_ctas:
            raise ValueError(f"domain_ctas must sum to the grid ({self.grid_ctas})")
        self.domain_ctas = tuple(int(c) for c in domain_ctas)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mSeqLens: cute.Tensor,
        mOut: cute.Tensor,
        mPageTable: cute.Tensor,
        mDomSched: cute.Tensor,
        mCounters: cute.Tensor,
        mClearCounters: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        if cutlass.const_expr(mQ.element_type is not cutlass.Float8E4M3FN):
            raise TypeError("q must be Float8E4M3FN")
        if cutlass.const_expr(mKCache.element_type is not cutlass.Uint8):
            raise TypeError("k_cache must be Uint8 NVFP4 pages")
        if cutlass.const_expr(mOut.element_type is not Float32):
            raise TypeError("out must be Float32")

        # mDomSched: [NUM_DOMAINS, batch + 1] prefix sums of each domain's pages.
        batch_size = mDomSched.shape[1] - 1
        query_rows = mQ.shape[0] // batch_size * self.num_heads
        physical_pages = mKCache.shape[0]
        page_stride = mKCache.stride[0]
        mKPacked = cute.make_tensor(
            mKCache.iterator,
            cute.make_layout(
                (self.packed_tma_row_bytes, self.packed_tma_rows, physical_pages),
                stride=(1, self.packed_tma_row_bytes, page_stride),
            ),
        )
        mKScale = cute.make_tensor(
            mKCache.iterator + self.packed_page_bytes,
            cute.make_layout(
                (self.scale_row_bytes, self.scale_rows, physical_pages),
                stride=(1, self.scale_row_bytes, page_stride),
            ),
        )
        mQ_krb = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (self.head_dim, query_rows * batch_size),
                stride=(1, self.head_dim),
            ),
        )
        mOut_pqb = cute.make_tensor(
            mOut.iterator,
            cute.select(mOut.layout, mode=[2, 1, 0]),
        )
        mPageTable_lb = cute.make_tensor(
            mPageTable.iterator,
            cute.select(mPageTable.layout, mode=[1, 0]),
        )

        qk_tiler = (self.m_tile, self.n_tile, self.k_tile)
        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            mQ.element_type,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            cta_group,
            qk_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
        tK_layout = sm100_utils.make_smem_layout_a(tiled_mma, qk_tiler, cutlass.Float8E4M3FN, 1)
        sQ_layout = sm100_utils.make_smem_layout_b(tiled_mma, qk_tiler, mQ.element_type, 1)
        sQ_tma_layout = cute.make_composed_layout(
            sQ_layout.inner,
            0,
            cute.make_layout((self.head_dim, self.num_queries), stride=(1, self.head_dim)),
        )
        sPacked_layout = cute.make_composed_layout(
            cute.make_swizzle(*self.packed_swizzle),
            0,
            cute.make_layout(
                (self.packed_tma_row_bytes, self.packed_tma_rows, self.raw_stages),
                stride=(1, self.packed_tma_row_bytes, self.packed_page_bytes),
            ),
        )
        sScale_layout = cute.make_layout(
            (self.scale_row_bytes, self.scale_rows, self.raw_stages),
            stride=(1, self.scale_row_bytes, self.scale_page_bytes),
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_packed, tma_tensor_packed = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mKPacked,
            cute.slice_(sPacked_layout, (None, None, 0)),
            (self.packed_tma_row_bytes, self.packed_tma_rows),
        )
        tma_atom_scale, tma_tensor_scale = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mKScale,
            cute.select(sScale_layout, mode=[0, 1]),
            (self.scale_row_bytes, self.scale_rows),
        )
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mQ_krb,
            sQ_tma_layout,
            (self.head_dim, self.num_queries),
        )

        sPartialMax_layout = cute.make_layout(
            (2, self.score_warps, self.num_queries),
            stride=(self.score_warps * self.num_queries, self.num_queries, 1),
        )

        @cute.struct
        class SharedStorage:
            raw_mbar_ptr: cute.struct.MemRange[Int64, self.raw_stages * 2]
            k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stages * 2]
            q_mbar_ptr: cute.struct.MemRange[Int64, 2]
            acc_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stages * 2]
            tmem_holding_buf: Int32
            claim: cute.struct.MemRange[Int32, 2]
            sPacked: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, cute.cosize(sPacked_layout)], 1024
            ]
            sScale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, cute.cosize(sScale_layout)], 128
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(sQ_layout)], 128
            ]
            sPartialMax: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(sPartialMax_layout)], 16
            ]

        self.shared_storage = SharedStorage
        smem_bytes = SharedStorage.size_in_bytes() if self.min_blocks_per_mp > 1 else None
        self.kernel(
            tiled_mma,
            tma_atom_packed,
            tma_tensor_packed,
            tma_atom_scale,
            tma_tensor_scale,
            tma_atom_Q,
            tma_tensor_Q,
            mSeqLens,
            mOut_pqb,
            mPageTable_lb,
            mDomSched,
            mCounters,
            mClearCounters,
            batch_size,
            query_rows,
            sPacked_layout,
            sScale_layout,
            tK_layout,
            sQ_layout,
            sQ_tma_layout,
            sPartialMax_layout,
        ).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=self.min_blocks_per_mp,
            use_pdl=True,
        )

    @cute.jit
    def _sm_domain(self) -> Int32:
        """0 or 1: bit ``%smid`` of the compile-time SM -> domain table (no memory load)."""
        sm = _smid()
        word = Int32(self.sm_domain_words[0])
        for i in cutlass.range_constexpr(1, _TABLE_WORDS):
            if (sm >> 5) == Int32(i):
                word = Int32(self.sm_domain_words[i])
        return (word >> (sm & Int32(31))) & Int32(1)

    @cute.jit
    def _claim(self, mCounters: cute.Tensor) -> tuple[Int32, Int32]:
        """Thread 0 only: the (domain, slot) this CTA scores, or slot -1.

        ``mCounters`` is this launch's own pair of slot counters; the next
        launch zeroes it after its PDL wait, so no CTA here has to detect the
        end of the claims or reset anything.
        """

        domain = self._sm_domain()
        slot = cute.arch.atomic_add(
            (mCounters.iterator + domain).llvm_ptr, Int32(1), sem="relaxed", scope="gpu"
        )
        domain_slots = Int32(self.domain_ctas[0])
        if domain == Int32(1):
            domain_slots = Int32(self.domain_ctas[1])
        if slot >= domain_slots:
            # Own domain exhausted (more of this domain's SMs ran CTAs than it
            # has slots): take a slot of the other domain.
            domain = domain ^ Int32(1)
            slot = cute.arch.atomic_add(
                (mCounters.iterator + domain).llvm_ptr, Int32(1), sem="relaxed", scope="gpu"
            )
            domain_slots = Int32(self.domain_ctas[0])
            if domain == Int32(1):
                domain_slots = Int32(self.domain_ctas[1])
            if slot >= domain_slots:
                slot = Int32(-1)
        return domain, slot

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_packed: cute.CopyAtom,
        mKPacked: cute.Tensor,
        tma_atom_scale: cute.CopyAtom,
        mKScale: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mQ_krb: cute.Tensor,
        mSeqLens: cute.Tensor,
        mOut_pqb: cute.Tensor,
        mPageTable_lb: cute.Tensor,
        mDomSched: cute.Tensor,
        mCounters: cute.Tensor,
        mClearCounters: cute.Tensor,
        batch_size: Int32,
        query_rows: Int32,
        sPacked_layout: cute.ComposedLayout,
        sScale_layout: cute.Layout,
        tK_layout: cute.ComposedLayout,
        sQ_layout: cute.ComposedLayout,
        sQ_tma_layout: cute.ComposedLayout,
        sPartialMax_layout: cute.Layout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        if warp_idx == Int32(1):
            cpasync.prefetch_descriptor(tma_atom_packed)
            cpasync.prefetch_descriptor(tma_atom_scale)
        elif warp_idx == Int32(2):
            cpasync.prefetch_descriptor(tma_atom_Q)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sPacked = storage.sPacked.get_tensor(sPacked_layout.outer, swizzle=sPacked_layout.inner)
        sScale = storage.sScale.get_tensor(sScale_layout)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQ_tma = cute.make_tensor(sQ.iterator, sQ_tma_layout.outer)
        sPartialMax = storage.sPartialMax.get_tensor(sPartialMax_layout)
        sClaim = storage.claim.get_tensor(cute.make_layout(2))
        # Claim first: the counters do not depend on the predecessor, so the
        # atomic overlaps its tail and the setup below instead of following
        # the PDL wait.
        if tidx == Int32(0):
            claimed_domain, claimed_slot = self._claim(mCounters)
            sClaim[0] = claimed_domain
            sClaim[1] = claimed_slot

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(_NamedBarrier.TmemPtr),
            num_threads=self.threads_per_cta,
        )
        score_barrier = pipeline.NamedBarrier(
            barrier_id=int(_NamedBarrier.Score),
            num_threads=self.score_threads,
        )
        claim_barrier = pipeline.NamedBarrier(
            barrier_id=int(_NamedBarrier.Claim),
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        tmem.allocate(self.tmem_cols)
        tmem.relinquish_alloc_permit()

        raw_producer, raw_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.raw_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.dequant_warps),
            tx_count=self.raw_page_bytes,
            barrier_storage=storage.raw_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        ).make_participants()
        k_producer, k_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.k_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.dequant_warps),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_queries * self.head_dim,
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.tmem_copy_threads,
            ),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        pipeline.pipeline_init_arrive(is_relaxed=True)

        tPsP, tPgP = cpasync.tma_partition(
            tma_atom_packed,
            0,
            cute.make_layout(1),
            cute.group_modes(sPacked, 0, 2),
            cute.group_modes(
                cute.flat_divide(mKPacked, (self.packed_tma_row_bytes, self.packed_tma_rows)), 0, 2
            ),
        )
        tSsS, tSgS = cpasync.tma_partition(
            tma_atom_scale,
            0,
            cute.make_layout(1),
            cute.group_modes(sScale, 0, 2),
            cute.group_modes(
                cute.flat_divide(mKScale, (self.scale_row_bytes, self.scale_rows)), 0, 2
            ),
        )

        tCrQ = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C((self.m_tile, self.n_tile))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))

        pipeline.pipeline_init_wait()
        cute.arch.griddepcontrol_wait()
        if tidx == Int32(0) and cute.arch.block_idx()[0] == Int32(0):
            # The previous launch has completed: zero its counters for its
            # next use.
            for counter in cutlass.range_constexpr(NUM_DOMAINS):
                mClearCounters[counter] = Int32(0)
        claim_barrier.arrive_and_wait()
        domain = sClaim[0]
        slot = sClaim[1]
        domain_ctas = Int32(self.domain_ctas[0])
        if domain == Int32(1):
            domain_ctas = Int32(self.domain_ctas[1])

        mScheduler = mDomSched[domain, None]

        total_pages = Int32(0)
        if lane_idx == Int32(0):
            total_pages = mScheduler[batch_size]
        total_pages = cute.arch.shuffle_sync(total_pages, 0)
        pages_per_worker = (total_pages + domain_ctas - Int32(1)) // domain_ctas
        global_page_begin = Int32(0)
        num_pages = Int32(0)
        if slot >= Int32(0):
            global_page_begin = slot * pages_per_worker
            if global_page_begin < total_pages:
                global_page_end = global_page_begin + pages_per_worker
                if global_page_end > total_pages:  # noqa: PLR1730
                    global_page_end = total_pages
                num_pages = global_page_end - global_page_begin

        batch_idx = Int32(0)
        if lane_idx == Int32(0):
            batch_hi = batch_size
            while batch_idx < batch_hi:
                batch_mid = (batch_idx + batch_hi) // Int32(2)
                if mScheduler[batch_mid + 1] <= global_page_begin:
                    batch_idx = batch_mid + Int32(1)
                else:
                    batch_hi = batch_mid
        batch_idx = cute.arch.shuffle_sync(batch_idx, 0)
        # The k-th page of a request in this domain is logical page 2k + domain.
        logical_page_begin = (global_page_begin - mScheduler[batch_idx]) * Int32(2) + domain
        thr_mma = tiled_mma.get_slice(0)
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tCtAcc_staged = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tCrK_base = tiled_mma.make_fragment_A(cute.make_tensor(tmem_ptr, tK_layout.outer))[
            None, None, None, 0
        ]
        k_tmem_col = tmem_ptr.toint() + Int32(self.acc_tmem_cols)
        tCrK = cute.make_tensor(
            tCrK_base.iterator + k_tmem_col * Int32(self.e4m3_per_tmem_col),
            cute.append(
                tCrK_base.layout,
                cute.make_layout(
                    (self.k_stages,),
                    stride=(self.k_stage_tmem_cols * self.e4m3_per_tmem_col,),
                ),
            ),
        )
        k_view_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (self.m_tile, self.k_stage_tmem_cols),
            tcgen05.OperandSource.TMEM,
        )
        k_view_layout = k_view_mma.make_fragment_C(
            k_view_mma.partition_shape_C((self.m_tile, self.k_stage_tmem_cols))
        ).layout
        tKtK = cute.make_tensor(
            tmem_ptr + self.acc_tmem_cols,
            cute.append(
                cute.composition(
                    k_view_layout,
                    cute.make_layout((self.page_size, self.k_stage_tmem_cols)),
                ),
                cute.make_layout((self.k_stages,), stride=(self.k_stage_tmem_cols,)),
            ),
        )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(getattr(tcgen05.Repetition, f"x{self.num_queries}")),
            Float32,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(
            tmem_load_atom,
            tCtAcc_staged[(None, None, None, 0)],
        )
        copy_tidx = tidx % Int32(self.tmem_copy_threads)
        thr_tmem_copy = tmem_tiled_copy.get_slice(copy_tidx)
        tTR_tAcc_staged = thr_tmem_copy.partition_S(tCtAcc_staged)
        cScores = cute.make_identity_tensor((self.m_tile, self.n_tile))
        tCcScores = thr_mma.partition_C(cScores)
        tTR_cScores = thr_tmem_copy.partition_D(tCcScores)

        if warp_idx == Int32(1):
            current_batch = batch_idx
            logical_page = logical_page_begin
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                # Each lane fetches one of this domain's next 32 page-table
                # entries (every other logical page).
                for chunk_begin in cutlass.range(0, segment_pages, self.threads_per_warp, unroll=1):
                    chunk_pages = segment_pages - chunk_begin
                    if chunk_pages > Int32(self.threads_per_warp):  # noqa: PLR1730
                        chunk_pages = Int32(self.threads_per_warp)
                    lane_page = Int32(0)
                    if lane_idx < chunk_pages:
                        lane_page = mPageTable_lb[logical_page + Int32(2) * lane_idx, current_batch]
                    for chunk_page in cutlass.range(chunk_pages, unroll=1):
                        physical_page = cute.arch.shuffle_sync(lane_page, chunk_page)
                        raw_empty = raw_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_packed,
                            tPgP[(None, 0, 0, physical_page)],
                            tPsP[(None, raw_empty.index)],
                            tma_bar_ptr=raw_empty.barrier,
                        )
                        cute.copy(
                            tma_atom_scale,
                            tSgS[(None, 0, 0, physical_page)],
                            tSsS[(None, raw_empty.index)],
                            tma_bar_ptr=raw_empty.barrier,
                        )
                    global_page += chunk_pages
                    logical_page += Int32(2) * chunk_pages
                pages_remaining -= segment_pages
                while (
                    current_batch < batch_size - Int32(1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
                    logical_page = (global_page - mScheduler[current_batch]) * Int32(2) + domain
            raw_producer.tail()
        elif warp_idx == Int32(2):
            current_batch = batch_idx
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                q_empty = q_producer.acquire_and_advance()
                q_row = (current_batch + Int32(1)) * query_rows - Int32(self.num_queries)
                tQsQ, tQgQ = cpasync.tma_partition(
                    tma_atom_Q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ_tma, 0, 2),
                    cute.group_modes(
                        cute.flat_divide(
                            cute.domain_offset((0, q_row), mQ_krb),
                            (self.head_dim, self.num_queries),
                        ),
                        0,
                        2,
                    ),
                )
                cute.copy(
                    tma_atom_Q,
                    tQgQ[(None, 0, 0)],
                    tQsQ,
                    tma_bar_ptr=q_empty.barrier,
                )
                global_page += segment_pages
                pages_remaining -= segment_pages
                while (
                    current_batch < batch_size - Int32(1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
            q_producer.tail()
        elif warp_idx == Int32(0):
            num_k_blocks = cute.size(tCrK, mode=[2])
            current_batch = batch_idx
            global_page = global_page_begin
            pages_remaining = num_pages
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                q_full = q_consumer.wait_and_advance()
                for _ in cutlass.range(segment_pages, unroll=1):
                    acc_empty = acc_producer.acquire_and_advance()
                    tCtAcc = tCtAcc_staged[(None, None, None, acc_empty.index)]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    k_full = k_consumer.wait_and_advance()
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrK[(None, None, k_block_idx, k_full.index)],
                            tCrQ[(None, None, k_block_idx, 0)],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    k_full.release()
                    acc_empty.commit()
                q_full.release()
                global_page += segment_pages
                pages_remaining -= segment_pages
                while (
                    current_batch < batch_size - Int32(1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
            acc_producer.tail()

        rScores = cute.make_rmem_tensor(tTR_cScores.shape, Float32)
        if warp_idx >= Int32(self.dequant_warp_begin):
            self._dequant_pages(
                num_pages,
                raw_consumer,
                k_producer,
                sPacked,
                sScale,
                tKtK,
                copy_tidx,
            )
        elif warp_idx >= Int32(self.score_warp_begin):
            score_warp_idx = warp_idx - Int32(self.score_warp_begin)
            rScoresFlat = cute.make_tensor(
                rScores.iterator,
                cute.make_layout(self.num_queries),
            )
            current_batch = batch_idx
            logical_page = logical_page_begin
            global_page = global_page_begin
            last_global_page = global_page_begin + num_pages - Int32(1)
            pages_remaining = num_pages
            local_block = Int32(0)
            if (
                num_pages > Int32(0)
                and score_warp_idx == Int32(0)
                and lane_idx < Int32(self.num_queries)
            ):
                seq_len = mSeqLens[current_batch]
                query_position = (
                    seq_len - Int32(self.query_length) + lane_idx // Int32(self.num_heads)
                )
                local_block = query_position // Int32(self.page_size)
            if num_pages == Int32(0):
                # No scores to write: release the dependents right away.
                cute.arch.griddepcontrol_launch_dependents()
            while pages_remaining > Int32(0):
                segment_pages = mScheduler[current_batch + 1] - global_page
                if segment_pages > pages_remaining:  # noqa: PLR1730
                    segment_pages = pages_remaining
                for _ in cutlass.range(segment_pages, unroll=1):
                    partial_stage = global_page % Int32(2)
                    acc_full = acc_consumer.wait_and_advance()
                    cute.copy(
                        tmem_tiled_copy,
                        tTR_tAcc_staged[(None, None, None, None, acc_full.index)],
                        rScores,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    for query_idx in cutlass.range_constexpr(self.num_queries):
                        partial_max = cute.arch.warp_redux_sync(rScoresFlat[query_idx], "fmax")
                        if lane_idx == Int32(0):
                            sPartialMax[partial_stage, score_warp_idx, query_idx] = partial_max
                    acc_full.release()
                    score_barrier.arrive_and_wait()

                    if score_warp_idx == Int32(0) and lane_idx < Int32(self.num_queries):
                        query_idx = lane_idx
                        row_max = -Float32.inf
                        for partial_idx in cutlass.range_constexpr(self.score_warps):
                            row_max = cute.arch.fmax(
                                row_max, sPartialMax[partial_stage, partial_idx, query_idx]
                            )
                        if global_page == last_global_page:
                            cute.arch.griddepcontrol_launch_dependents()
                        if logical_page < local_block:
                            mOut_pqb[logical_page, query_idx, current_batch] = row_max
                    global_page += Int32(1)
                    logical_page += Int32(2)
                pages_remaining -= segment_pages
                while (
                    current_batch < batch_size - Int32(1)
                    and mScheduler[current_batch + 1] <= global_page
                ):
                    current_batch += Int32(1)
                    logical_page = (global_page - mScheduler[current_batch]) * Int32(2) + domain
                if score_warp_idx == Int32(0) and lane_idx < Int32(self.num_queries):
                    seq_len = mSeqLens[current_batch]
                    query_position = (
                        seq_len - Int32(self.query_length) + lane_idx // Int32(self.num_heads)
                    )
                    local_block = query_position // Int32(self.page_size)

        pipeline.sync(barrier_id=int(_NamedBarrier.Final))
        tmem.free(tmem_ptr)


__all__ = ["Q8KV4DecodeIndexerLocalSm100"]
