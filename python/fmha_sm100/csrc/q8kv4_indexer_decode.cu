// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// Q8KV4 paged decode indexer proxy scores over the vLLM packed NVFP4 page for one, two or
// four index heads, on the SM100 TMEM-source FP8 MMA. The work schedule comes from the shared
// decode plan (cute/src/sm100/decode_indexer_plan.py).

#include <cstddef>
#include <cstdint>
#include <limits>

#include "q8kv4_indexer/runner.hpp"
#include "tvm_ffi_utils.h"

namespace q8kv4_indexer {

template <typename Traits>
cudaError_t launch_indexer_gemm(IndexerGemmArguments const &arguments, cudaStream_t stream) {
  using Runner = IndexerGemmRunner<Traits>;
  if (!Runner::can_run(arguments)) {
    return cudaErrorNotSupported;
  }
  IndexerGemmParams params{};
  cudaError_t const status = Runner::to_underlying_arguments(arguments, params);
  if (status != cudaSuccess) {
    return status;
  }
  return Runner::run(params, stream);
}

} // namespace q8kv4_indexer

namespace {

using q8kv4_indexer::IndexerGemmArguments;
using q8kv4_indexer::IndexerGemmTraitsForHeads;
// Head-independent limits shared by every specialization.
using IndexerGemmTraits = IndexerGemmTraitsForHeads<1, 16>;

template <typename T> T *tensor_data(TensorView tensor) {
  return reinterpret_cast<T *>(static_cast<char *>(tensor.data_ptr()) + tensor.byte_offset());
}

int checked_batch(int64_t batch_size) {
  TVM_FFI_ICHECK(batch_size > 0 && batch_size <= std::numeric_limits<int>::max())
      << "batch_size must be positive and fit in int32";
  return static_cast<int>(batch_size);
}

int64_t scheduler_words(int batch, int64_t workers) {
  return static_cast<int64_t>(batch) + (1 + q8kv4_indexer::kWorkerStartFields) * workers + 2;
}

void check_metadata(TensorView page_table, TensorView seq_lens) {
  CHECK_INPUT_AND_TYPE(page_table, dl_int32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_DEVICE(seq_lens, page_table);
  CHECK_DIM(2, page_table);
  CHECK_DIM(1, seq_lens);
  TVM_FFI_ICHECK(page_table.size(0) > 0 && page_table.size(0) <= std::numeric_limits<int>::max())
      << "block_table batch must be positive and fit in int32";
  TVM_FFI_ICHECK(page_table.size(1) > 0 && page_table.size(1) <= IndexerGemmTraits::kMaximumPages)
      << "max_pages must be in [1, " << IndexerGemmTraits::kMaximumPages << "]";
  TVM_FFI_ICHECK(seq_lens.size(0) == page_table.size(0)) << "seq_lens must have shape [batch]";
}

template <typename Traits>
void run_indexer(TensorView q, TensorView k_cache, TensorView page_table, TensorView seq_lens,
                 TensorView scheduler, int64_t sm_count, TensorView output, int64_t stream_ptr) {
  check_metadata(page_table, seq_lens);
  CHECK_INPUT_AND_TYPE(q, dl_float8_e4m3fn);
  CHECK_CUDA(k_cache);
  CHECK_INPUT_TYPE(k_cache, dl_uint8);
  CHECK_INPUT_AND_TYPE(scheduler, dl_int32);
  CHECK_INPUT_AND_TYPE(output, dl_float32);
  CHECK_DEVICE(q, page_table);
  CHECK_DEVICE(k_cache, page_table);
  CHECK_DEVICE(scheduler, page_table);
  CHECK_DEVICE(output, page_table);
  CHECK_DIM(1, scheduler);

  int const batch = checked_batch(page_table.size(0));
  int const max_pages = static_cast<int>(page_table.size(1));
  TVM_FFI_ICHECK(sm_count > 0 && sm_count <= std::numeric_limits<int>::max())
      << "sm_count must be positive and fit in int32";
  TVM_FFI_ICHECK(q.ndim() == 3 && q.size(0) > 0 && q.size(0) % batch == 0 &&
                 q.size(1) == Traits::kNumIndexHeads && q.size(2) == Traits::kHeadDim)
      << "q must have shape [batch * Q, heads, 128]";
  int64_t const query_length = q.size(0) / batch;
  TVM_FFI_ICHECK(query_length >= 1 && query_length <= Traits::kMaxQueryLength &&
                 query_length * Traits::kNumIndexHeads <= Traits::kQueryColumns)
      << "query tokens per request must be in [1, "
      << Traits::kQueryColumns / Traits::kNumIndexHeads << "]";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor_data<uint8_t>(q)) % 16 == 0)
      << "q data pointer must be 16-byte aligned";
  TVM_FFI_ICHECK(k_cache.ndim() == 3 && k_cache.size(0) > 0 &&
                 k_cache.size(0) <= std::numeric_limits<int>::max() &&
                 k_cache.size(1) == Traits::kPageTokens &&
                 k_cache.size(2) * Traits::kPageTokens == Traits::kPageBytes)
      << "k_cache must have shape [num_blocks, 128, 72]";
  TVM_FFI_ICHECK(k_cache.stride(2) == 1 && k_cache.stride(1) == k_cache.size(2))
      << "k_cache pages must be contiguous";
  TVM_FFI_ICHECK(k_cache.stride(0) >= Traits::kPageBytes && k_cache.stride(0) % 16 == 0)
      << "k_cache page stride must be at least 9216 bytes and 16-byte aligned";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor_data<uint8_t>(k_cache)) % 16 == 0)
      << "k_cache data pointer must be 16-byte aligned";
  TVM_FFI_ICHECK(output.ndim() == 3 && output.size(0) == batch &&
                 output.size(1) == query_length * Traits::kNumIndexHeads &&
                 output.size(2) == max_pages)
      << "scores must have shape [batch, Q * heads, max_pages]";
  TVM_FFI_ICHECK(scheduler.size(0) >= scheduler_words(batch, sm_count))
      << "scheduler is too small: need " << scheduler_words(batch, sm_count)
      << " int32 words for this batch and worker count";

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  IndexerGemmArguments arguments{};
  arguments.q_ptr = tensor_data<void const>(q);
  arguments.k_cache_ptr = tensor_data<void const>(k_cache);
  arguments.page_table_ptr = tensor_data<int32_t const>(page_table);
  arguments.kv_lengths_ptr = tensor_data<int32_t const>(seq_lens);
  arguments.scheduler_workspace_ptr = tensor_data<int32_t const>(scheduler);
  arguments.output_ptr = tensor_data<float>(output);
  arguments.batch = batch;
  arguments.query_length = static_cast<int>(query_length);
  arguments.max_pages = max_pages;
  arguments.physical_pages = static_cast<int>(k_cache.size(0));
  arguments.page_stride_bytes = k_cache.stride(0);
  arguments.sm_count = static_cast<int>(sm_count);

  cudaError_t const status =
      q8kv4_indexer::launch_indexer_gemm<Traits>(arguments,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Q8KV4 indexer run failed: " << cudaGetErrorString(status);
}

template <int NumIndexHeads>
void run_for_heads(int64_t query_columns, TensorView q, TensorView k_cache, TensorView page_table,
                   TensorView seq_lens, TensorView scheduler, int64_t sm_count, TensorView output,
                   int64_t stream_ptr) {
#define Q8KV4_RUN(COLUMNS)                                                                         \
  if (query_columns == COLUMNS) {                                                                  \
    if constexpr (COLUMNS <= 16 * NumIndexHeads) {                                                 \
      return run_indexer<IndexerGemmTraitsForHeads<NumIndexHeads, COLUMNS>>(                       \
          q, k_cache, page_table, seq_lens, scheduler, sm_count, output, stream_ptr);              \
    }                                                                                              \
  }
  Q8KV4_RUN(16)
  Q8KV4_RUN(32)
  Q8KV4_RUN(48)
  Q8KV4_RUN(64)
#undef Q8KV4_RUN
  TVM_FFI_THROW(ValueError) << "query tokens per request must be in [1, "
                            << IndexerGemmTraits::kMaxQueryLength << "]";
}

} // namespace

// ``q`` is [batch * Q, heads, 128] with heads 1, 2 or 4. ``seq_lens`` and ``scheduler`` are the
// shared plan's length snapshot and int32 schedule; ``sm_count`` must be the plan's worker count.
// Scores are written for the historical pages before each query's local page; the local page and
// later columns stay untouched.
void q8kv4_indexer_run(TensorView q, TensorView k_cache, TensorView page_table, TensorView seq_lens,
                       TensorView scheduler, int64_t sm_count, TensorView output,
                       int64_t stream_ptr) {
  CHECK_DIM(2, page_table);
  TVM_FFI_ICHECK(q.ndim() == 3 && page_table.size(0) > 0 && q.size(0) % page_table.size(0) == 0)
      << "q must have shape [batch * Q, heads, 128]";
  int64_t const heads = q.size(1);
  int64_t const query_length = q.size(0) / page_table.size(0);
  int64_t const query_columns = (query_length * heads + 15) / 16 * 16;
  switch (heads) {
  case 1:
    return run_for_heads<1>(query_columns, q, k_cache, page_table, seq_lens, scheduler, sm_count,
                            output, stream_ptr);
  case 2:
    return run_for_heads<2>(query_columns, q, k_cache, page_table, seq_lens, scheduler, sm_count,
                            output, stream_ptr);
  case 4:
    return run_for_heads<4>(query_columns, q, k_cache, page_table, seq_lens, scheduler, sm_count,
                            output, stream_ptr);
  default:
    TVM_FFI_THROW(ValueError) << "q8kv4_indexer_run supports 1, 2 or 4 index heads, got " << heads;
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(q8kv4_indexer_run, q8kv4_indexer_run);
