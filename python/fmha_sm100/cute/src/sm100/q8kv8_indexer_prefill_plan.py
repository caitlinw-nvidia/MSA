# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Device-side varlen task planning for the Q8KV8 prefill indexer."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from src.sm100.q8kv8_indexer_prefill import Q8KV8PrefillIndexerSm100


class Q8KV8PrefillIndexerPlanReset:
    """Reset device-resident queue counters and error state."""

    num_buckets = Q8KV8PrefillIndexerSm100.num_task_buckets
    threads_per_cta = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(mTaskCounts, mPlanError).launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
    ) -> None:
        lane_idx = cute.arch.lane_idx()
        if lane_idx < cutlass.Int32(self.num_buckets):
            mTaskCounts[lane_idx] = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0):
            mPlanError[0] = cutlass.Int32(0)


class Q8KV8PrefillIndexerPlanBuild:
    """Expand device varlen metadata into page-range task buckets.

    Tasks tile Q rows ``token * num_heads + head``; a Q tile of 256 rows covers
    ``256 / num_heads`` tokens of one request. ``num_heads`` divides the tile,
    so every tile starts at a token boundary and the kernel can locate row
    ``r`` of a tile at token offset ``r // num_heads``.
    """

    q_tile = Q8KV8PrefillIndexerSm100.q_tile
    page_size = Q8KV8PrefillIndexerSm100.k_tile
    num_buckets = Q8KV8PrefillIndexerSm100.num_task_buckets
    split_page_chunk = 64
    large_page_chunk = 128
    min_page_chunk = 8
    split_q_tile_threshold = 60
    descriptor_words = Q8KV8PrefillIndexerSm100.task_descriptor_words
    page_count_shift = Q8KV8PrefillIndexerSm100.task_page_count_shift
    threads_per_cta = cute.arch.WARP_SIZE

    def __init__(self, pair_tiles: bool = False) -> None:
        # SM107 paired Q tiles: even tiles emit the tasks of (tile, tile + 1).
        self.pair_tiles = pair_tiles
        # SM107 stream-K: write one unit record per candidate (bucket 2, row =
        # candidate) and leave the cuts to Q8KV8PrefillIndexerPlanBalance.
        self.stream_k = pair_tiles

    @cute.jit
    def __call__(
        self,
        mCuSeqlensQ: cute.Tensor,
        mSeqLens: cute.Tensor,
        mLengths: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
        num_candidate_q_tiles: cutlass.Int32,
        task_capacity: cutlass.Int32,
        num_heads: cutlass.Int32,
        max_page_chunk: cutlass.Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mCuSeqlensQ,
            mSeqLens,
            mLengths,
            mTaskDescriptors,
            mTaskCounts,
            mPlanError,
            task_capacity,
            num_heads,
            max_page_chunk,
        ).launch(
            grid=(num_candidate_q_tiles, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _cost_bucket(
        self,
        q_rows: cutlass.Int32,
        page_count: cutlass.Int32,
    ) -> cutlass.Int32:
        work = q_rows * page_count
        bucket = cutlass.Int32(0)
        if work > cutlass.Int32(1 << 10):
            bucket = cutlass.Int32(1)
        if work > cutlass.Int32(1 << 11):
            bucket = cutlass.Int32(2)
        if work > cutlass.Int32(1 << 12):
            bucket = cutlass.Int32(3)
        if work > cutlass.Int32(1 << 13):
            bucket = cutlass.Int32(4)
        if work > cutlass.Int32(1 << 14):
            bucket = cutlass.Int32(5)
        if work > cutlass.Int32(1 << 15):
            bucket = cutlass.Int32(6)
        if work > cutlass.Int32(1 << 16):
            bucket = cutlass.Int32(7)
        return bucket

    @cute.jit
    def _pack_extent(
        self,
        q_rows: cutlass.Int32,
        page_count: cutlass.Int32,
    ) -> cutlass.Int32:
        return (q_rows - cutlass.Int32(1)) | (
            (page_count - cutlass.Int32(1)) << cutlass.Int32(self.page_count_shift)
        )

    @cute.kernel
    def kernel(
        self,
        mCuSeqlensQ: cute.Tensor,
        mSeqLens: cute.Tensor,
        mLengths: cute.Tensor,
        mTaskDescriptors: cute.Tensor,
        mTaskCounts: cute.Tensor,
        mPlanError: cute.Tensor,
        task_capacity: cutlass.Int32,
        num_heads: cutlass.Int32,
        max_page_chunk: cutlass.Int32,
    ) -> None:
        lane_idx = cute.arch.lane_idx()
        candidate_idx, _, _ = cute.arch.block_idx()
        batch = cute.size(mCuSeqlensQ) - 1
        batch_idx = candidate_idx % batch
        q_tile_idx = candidate_idx // batch
        q_local_begin = q_tile_idx * cutlass.Int32(self.q_tile)

        # ``q_*`` values below count Q rows (token * num_heads + head).
        q_start = cutlass.Int32(0)
        seq_q = cutlass.Int32(0)
        seq_rows = cutlass.Int32(0)
        seq_k = cutlass.Int32(0)
        q_rows = cutlass.Int32(0)
        num_pages = cutlass.Int32(0)
        num_q_tiles = cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0):
            token_start = mCuSeqlensQ[batch_idx]
            seq_q = mCuSeqlensQ[batch_idx + cutlass.Int32(1)] - token_start
            q_start = token_start * num_heads
            seq_rows = seq_q * num_heads
            seq_k = mSeqLens[batch_idx]
            if q_local_begin < seq_rows:
                q_rows = seq_rows - q_local_begin
                if q_rows > cutlass.Int32(self.q_tile):
                    q_rows = cutlass.Int32(self.q_tile)
                last_position = (
                    seq_k
                    - seq_q
                    + (q_local_begin + q_rows - cutlass.Int32(1)) // num_heads
                )
                num_pages = last_position // cutlass.Int32(self.page_size)
                if num_pages < cutlass.Int32(0):
                    num_pages = cutlass.Int32(0)
                num_q_tiles = cute.ceil_div(seq_rows, self.q_tile)

        q_start = cute.arch.shuffle_sync(q_start, 0)
        seq_q = cute.arch.shuffle_sync(seq_q, 0)
        seq_rows = cute.arch.shuffle_sync(seq_rows, 0)
        seq_k = cute.arch.shuffle_sync(seq_k, 0)
        q_rows = cute.arch.shuffle_sync(q_rows, 0)
        num_pages = cute.arch.shuffle_sync(num_pages, 0)
        num_q_tiles = cute.arch.shuffle_sync(num_q_tiles, 0)

        for row_iter in cutlass.range_constexpr(self.q_tile // cute.arch.WARP_SIZE):
            q_local = (
                q_local_begin
                + lane_idx
                + cutlass.Int32(row_iter * cute.arch.WARP_SIZE)
            )
            if q_local < seq_rows:
                query_position = seq_k - seq_q + q_local // num_heads
                local_block = query_position // cutlass.Int32(self.page_size)
                mLengths[q_start + q_local] = local_block + cutlass.Int32(1)

        q_rows_b = cutlass.Int32(0)
        emit = q_rows > cutlass.Int32(0)
        if cutlass.const_expr(self.pair_tiles):
            if (q_tile_idx & cutlass.Int32(1)) != cutlass.Int32(0):
                emit = q_rows < cutlass.Int32(0)
            q_rows_b = seq_rows - q_local_begin - cutlass.Int32(self.q_tile)
            if q_rows_b < cutlass.Int32(0):
                q_rows_b = cutlass.Int32(0)
            if q_rows_b > cutlass.Int32(self.q_tile):
                q_rows_b = cutlass.Int32(self.q_tile)
            if q_rows_b > cutlass.Int32(0):
                # The pair's page range is tile B's (later rows see more pages).
                last_position_b = (
                    seq_k - seq_q
                    + (q_local_begin + cutlass.Int32(self.q_tile) + q_rows_b - cutlass.Int32(1))
                    // num_heads
                )
                num_pages = last_position_b // cutlass.Int32(self.page_size)
        if cutlass.const_expr(self.stream_k):
            unit_pages = cutlass.Int32(0)
            if emit:
                unit_pages = num_pages
            if lane_idx == cutlass.Int32(0):
                mTaskDescriptors[2, candidate_idx, 0] = q_start + q_local_begin
                mTaskDescriptors[2, candidate_idx, 1] = seq_k - seq_q + q_local_begin // num_heads
                mTaskDescriptors[2, candidate_idx, 2] = unit_pages
                mTaskDescriptors[2, candidate_idx, 3] = batch_idx
                mTaskDescriptors[2, candidate_idx, 4] = q_rows
                mTaskDescriptors[2, candidate_idx, 5] = q_rows_b
            emit = q_rows < cutlass.Int32(0)
        if lane_idx == cutlass.Int32(0) and emit:
            page_chunk = cutlass.Int32(self.split_page_chunk)
            if num_q_tiles >= cutlass.Int32(self.split_q_tile_threshold):
                page_chunk = cutlass.Int32(self.large_page_chunk)
            # Host-computed cap keeps the task count near the resident cluster
            # count when few Q tiles would otherwise leave most clusters idle.
            if page_chunk > max_page_chunk:
                page_chunk = max_page_chunk
            if page_chunk > num_pages:
                page_chunk = num_pages
            page_begin = cutlass.Int32(0)
            while page_begin < num_pages:
                page_count = num_pages - page_begin
                if page_count > page_chunk:
                    page_count = page_chunk
                bucket = self._cost_bucket(q_rows + q_rows_b, page_count)
                slot = cute.arch.atomic_add(
                    (mTaskCounts.iterator + bucket).llvm_ptr,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
                if slot < task_capacity:
                    mTaskDescriptors[bucket, slot, 0] = q_start + q_local_begin
                    mTaskDescriptors[bucket, slot, 1] = (
                        seq_k - seq_q + q_local_begin // num_heads
                    )
                    mTaskDescriptors[bucket, slot, 2] = page_begin
                    mTaskDescriptors[bucket, slot, 3] = batch_idx
                    mTaskDescriptors[bucket, slot, 4] = self._pack_extent(
                        q_rows,
                        page_count,
                    )
                    mTaskDescriptors[bucket, slot, 5] = q_rows_b
                else:
                    cute.arch.atomic_exch(
                        mPlanError.iterator.llvm_ptr,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                page_begin += page_count


class Q8KV8PrefillIndexerPlanBalance:
    """Stream-K cut of the SM107 unit records into one contiguous run per cluster.

    Units (bucket 2, one per candidate Q tile; pages == 0 for none) are visited
    request by request, tile by tile. A unit with pages costs pages +
    ``unit_overhead`` (Q load and pipeline ramp); the cost line is cut at
    multiples of ``target = max(ceil(total / clusters), min_slice)`` and every
    non-empty piece becomes one bucket-0 task, in run order. Bucket 1 word 0 of
    row c (c = 0..clusters) is the first task of cluster c.
    """

    unit_overhead = 4
    min_slice = 4
    page_count_shift = Q8KV8PrefillIndexerSm100.task_page_count_shift
    threads_per_cta = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mTaskDescriptors: cute.Tensor,
        mPlanError: cute.Tensor,
        num_units: cutlass.Int32,
        batch: cutlass.Int32,
        num_clusters: cutlass.Int32,
        task_capacity: cutlass.Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mTaskDescriptors,
            mPlanError,
            num_units,
            batch,
            num_clusters,
            task_capacity,
        ).launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _unit_pages(
        self,
        mTaskDescriptors: cute.Tensor,
        order: cutlass.Int32,
        num_units: cutlass.Int32,
        batch: cutlass.Int32,
        tiles: cutlass.Int32,
        num_orders: cutlass.Int32,
    ):
        """Unit at run position ``order`` (request-major) and its page count."""
        req = order // tiles
        unit = (order - req * tiles) * batch + req
        pages = cutlass.Int32(0)
        if order >= num_orders:
            unit = num_units
        if unit < num_units:
            pages = mTaskDescriptors[2, unit, 2]
        return unit, pages

    @cute.jit
    def _inclusive_scan(self, value: cutlass.Int32) -> cutlass.Int32:
        lane = cute.arch.lane_idx()
        for i in cutlass.range_constexpr(5):
            src = lane - cutlass.Int32(1 << i)
            if src < cutlass.Int32(0):
                src = cutlass.Int32(0)
            other = cute.arch.shuffle_sync(value, src)
            if lane >= cutlass.Int32(1 << i):
                value = value + other
        return value

    @cute.kernel
    def kernel(
        self,
        mTaskDescriptors: cute.Tensor,
        mPlanError: cute.Tensor,
        num_units: cutlass.Int32,
        batch: cutlass.Int32,
        num_clusters: cutlass.Int32,
        task_capacity: cutlass.Int32,
    ) -> None:
        lane = cute.arch.lane_idx()
        warp = cutlass.Int32(cute.arch.WARP_SIZE)
        overhead = cutlass.Int32(self.unit_overhead)
        tiles = (num_units + batch - cutlass.Int32(1)) // batch
        num_orders = tiles * batch

        # Pass 1: total cost.
        total = cutlass.Int32(0)
        base = cutlass.Int32(0)
        while base < num_orders:
            _, pages = self._unit_pages(mTaskDescriptors, base + lane, num_units, batch, tiles, num_orders)
            if pages > cutlass.Int32(0):
                total = total + pages + overhead
            base += warp
        for i in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, 1 << i)
        target = (total + num_clusters - cutlass.Int32(1)) // num_clusters
        if target < cutlass.Int32(self.min_slice):
            target = cutlass.Int32(self.min_slice)

        # Pass 2: cut the cost line; piece f of the run order is bucket-0 task f.
        carry_cost = cutlass.Int32(0)
        carry_tasks = cutlass.Int32(0)
        base = cutlass.Int32(0)
        while base < num_orders:
            unit, pages = self._unit_pages(mTaskDescriptors, base + lane, num_units, batch, tiles, num_orders)
            cost = cutlass.Int32(0)
            if pages > cutlass.Int32(0):
                cost = pages + overhead
            cost_end = carry_cost + self._inclusive_scan(cost)
            cost_begin = cost_end - cost
            num_pieces = cutlass.Int32(0)
            if cost > cutlass.Int32(0):
                c = cost_begin // target
                while c * target < cost_end:
                    lo = cost_begin
                    if lo < c * target:
                        lo = c * target
                    hi = cost_end
                    if hi > (c + cutlass.Int32(1)) * target:
                        hi = (c + cutlass.Int32(1)) * target
                    lo = lo - cost_begin - overhead
                    hi = hi - cost_begin - overhead
                    if lo < cutlass.Int32(0):
                        lo = cutlass.Int32(0)
                    if hi > lo:
                        num_pieces += cutlass.Int32(1)
                    c += cutlass.Int32(1)
            tasks_end = carry_tasks + self._inclusive_scan(num_pieces)
            f = tasks_end - num_pieces
            if cost > cutlass.Int32(0):
                q_global_begin = mTaskDescriptors[2, unit, 0]
                q_position_begin = mTaskDescriptors[2, unit, 1]
                batch_idx = mTaskDescriptors[2, unit, 3]
                q_rows = mTaskDescriptors[2, unit, 4]
                q_rows_b = mTaskDescriptors[2, unit, 5]
                c = cost_begin // target
                while c * target < cost_end:
                    lo = cost_begin
                    if lo < c * target:
                        lo = c * target
                    hi = cost_end
                    if hi > (c + cutlass.Int32(1)) * target:
                        hi = (c + cutlass.Int32(1)) * target
                    lo = lo - cost_begin - overhead
                    hi = hi - cost_begin - overhead
                    if lo < cutlass.Int32(0):
                        lo = cutlass.Int32(0)
                    if hi > lo:
                        if f < task_capacity:
                            mTaskDescriptors[0, f, 0] = q_global_begin
                            mTaskDescriptors[0, f, 1] = q_position_begin
                            mTaskDescriptors[0, f, 2] = lo
                            mTaskDescriptors[0, f, 3] = batch_idx
                            mTaskDescriptors[0, f, 4] = (q_rows - cutlass.Int32(1)) | (
                                (hi - lo - cutlass.Int32(1)) << cutlass.Int32(self.page_count_shift)
                            )
                            mTaskDescriptors[0, f, 5] = q_rows_b
                            # Cluster of the task, read back by pass 3.
                            mTaskDescriptors[3, f, 0] = c
                        else:
                            cute.arch.atomic_exch(
                                mPlanError.iterator.llvm_ptr,
                                cutlass.Int32(1),
                                sem="relaxed",
                                scope="gpu",
                            )
                        f += cutlass.Int32(1)
                    c += cutlass.Int32(1)
            carry_cost = cute.arch.shuffle_sync(cost_end, 31)
            carry_tasks = cute.arch.shuffle_sync(tasks_end, 31)
            base += warp
        num_tasks = carry_tasks
        if num_tasks > task_capacity:
            num_tasks = task_capacity

        # Pass 3: run starts. Task f opens clusters (cluster[f - 1], cluster[f]];
        # clusters after the last task's start at num_tasks.
        cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_warp()
        f = lane
        while f < num_tasks:
            cur = mTaskDescriptors[3, f, 0]
            prev = cutlass.Int32(-1)
            if f > cutlass.Int32(0):
                prev = mTaskDescriptors[3, f - cutlass.Int32(1), 0]
            c = prev + cutlass.Int32(1)
            while c <= cur:
                mTaskDescriptors[1, c, 0] = f
                c += cutlass.Int32(1)
            f += warp
        last = cutlass.Int32(-1)
        if num_tasks > cutlass.Int32(0):
            last = mTaskDescriptors[3, num_tasks - cutlass.Int32(1), 0]
        c = last + cutlass.Int32(1) + lane
        while c <= num_clusters:
            mTaskDescriptors[1, c, 0] = num_tasks
            c += warp


__all__ = [
    "Q8KV8PrefillIndexerPlanBalance",
    "Q8KV8PrefillIndexerPlanBuild",
    "Q8KV8PrefillIndexerPlanReset",
]
