# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Shared decode-indexer plan: one CTA builds a decode step's whole page schedule.

Ported from MiniMax-AI/MSA nv_dev ``inference/msa_v1/indexer/decode/plan_kernel.py``
(fused logical-page scheduling following DeepGEMM's paged metadata design). The
``mScheduler`` int32 words are, in order:

* ``[0, batch]``: prefix sum of each request's historical pages,
  ``clamp((seq_len - 1) // 128, 0, max_pages)``. The CuTe DSL decode indexers read
  only this prefix and partition it themselves.
* ``[batch + 1, batch + workers + 1]``: first global page of each worker, then the total.
* ``[batch + workers + 2, ...)``: per-worker request index, first logical page and first
  segment count, read by the single-head Q8KV4 CUTLASS kernel (one CTA per worker).

``mValidPages`` holds each (token, head) row's candidate page count including its local
page, token-major (``row * heads + head``) to match the TopK rows; nv_dev stores it
head-major. ``mSnapshot`` copies ``mLengths`` so every layer reads the planned lengths.
"""

import cutlass
import cutlass.cute as cute  # noqa: PLR0402

# isort: split
import cuda.bindings.driver as cuda


class DecodeIndexerPlanSm100:
    """Generate a complete immutable-for-consumers plan in one CTA."""

    @cute.jit
    def __call__(
        self,
        mLengths: cute.Tensor,
        mSnapshot: cute.Tensor,
        mScheduler: cute.Tensor,
        mValidPages: cute.Tensor,
        query_length: cutlass.Int32,
        heads: cutlass.Int32,
        max_pages: cutlass.Int32,
        workers: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mLengths,
            mSnapshot,
            mScheduler,
            mValidPages,
            query_length,
            heads,
            max_pages,
            workers,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        mLengths: cute.Tensor,
        mSnapshot: cute.Tensor,
        mScheduler: cute.Tensor,
        mValidPages: cute.Tensor,
        query_length: cutlass.Int32,
        heads: cutlass.Int32,
        max_pages: cutlass.Int32,
        workers: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        batch = cutlass.Int32(mLengths.shape[0])
        if tid == 0:
            mScheduler[0] = 0
        for request in cutlass.range(tid, batch, 256):
            length = mLengths[request]
            mSnapshot[request] = length
            count = (length - 1) // 128
            if count < 0:
                count = cutlass.Int32(0)
            if count > max_pages:  # noqa: PLR1730
                count = max_pages
            mScheduler[request + 1] = count
        cute.arch.sync_threads()

        # One warp scans successive chunks; no batch-sized shared allocation.
        if tid < 32:
            carry = cutlass.Int32(0)
            for base in cutlass.range(0, batch, 32):
                request = base + lane
                value = cutlass.Int32(0)
                if request < batch:
                    value = mScheduler[request + 1]
                for shift in cutlass.range_constexpr(5):
                    offset = 1 << shift
                    previous = cute.arch.shuffle_sync_up(
                        value, offset=offset, mask_and_clamp=0
                    )
                    if lane >= offset:
                        value += previous
                value += carry
                if request < batch:
                    mScheduler[request + 1] = value
                carry = cute.arch.shuffle_sync(value, 31)
        cute.arch.sync_threads()

        total = mScheduler[batch]
        quotient = total // workers
        remainder = total % workers
        start_offset = batch + workers + 2
        for worker in cutlass.range(tid, workers + 1, 256):
            extra = remainder
            if worker < remainder:
                extra = worker
            begin = worker * quotient + extra
            mScheduler[batch + 1 + worker] = begin
            if worker < workers:
                lower = cutlass.Int32(0)
                upper = batch
                while lower < upper:
                    middle = (lower + upper) // 2
                    if mScheduler[middle + 1] <= begin:
                        lower = middle + 1
                    else:
                        upper = middle
                logical_page = begin - mScheduler[lower]
                end_extra = remainder
                if worker + 1 < remainder:
                    end_extra = worker + 1
                end = (worker + 1) * quotient + end_extra
                request_end = begin
                if lower < batch:
                    request_end = mScheduler[lower + 1]
                mScheduler[start_offset + worker] = lower
                mScheduler[start_offset + workers + worker] = logical_page
                segment = end - begin
                if request_end < end:
                    segment = request_end - begin
                if segment > 16:
                    segment = cutlass.Int32(16)
                mScheduler[start_offset + 2 * workers + worker] = segment

        rows = batch * query_length
        for row in cutlass.range(tid, rows, 256):
            request = row // query_length
            token = row % query_length
            valid = (mSnapshot[request] - query_length + token) // 128 + 1
            if valid < 1:
                valid = cutlass.Int32(1)
            if valid > max_pages:  # noqa: PLR1730
                valid = max_pages
            for head in cutlass.range(heads):
                mValidPages[row * heads + head] = valid
