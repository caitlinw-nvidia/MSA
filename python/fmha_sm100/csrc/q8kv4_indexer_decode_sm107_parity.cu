// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

// Locality-domain parity split of the SM107 one-head Q8KV4 decode indexer:
// logical page j of every request lives in locality domain j % 2, and each
// CTA scores its own domain's pages first. Scores are bitwise those of
// q8kv4_indexer_decode_sm107 (same per-page math, same output layout).

#include <cstddef>
#include <cstdint>
#include <limits>

#include "q8kv4_indexer_sm107/runner.hpp"
#include "q8kv4_indexer_sm107_parity/kernel.hpp"
#include "tvm_ffi_utils.h"

namespace {

using q8kv4_indexer_sm107::IndexerGemmArguments;
using q8kv4_indexer_sm107::IndexerGemmTraits;
using q8kv4_indexer_sm107_parity::ParityIndexerGemmParams;
using Runner = q8kv4_indexer_sm107::IndexerGemmRunner<IndexerGemmTraits>;
using ParityRunner = q8kv4_indexer_sm107_parity::ParityIndexerGemmRunner<IndexerGemmTraits>;

template <typename T> T *tensor_data(TensorView tensor) {
  return reinterpret_cast<T *>(static_cast<char *>(tensor.data_ptr()) + tensor.byte_offset());
}

} // namespace

int64_t q8kv4_indexer_parity_resident_ctas_per_sm() {
  return ParityRunner::resident_ctas_per_sm();
}

int64_t q8kv4_indexer_parity_max_tile_pages() { return IndexerGemmTraits::kPagesPerWorkTile; }

void q8kv4_indexer_parity_run(TensorView q, TensorView k_cache, TensorView page_table,
                              TensorView seq_lens, TensorView domain_prefix, TensorView counters,
                              TensorView clear_counters, int64_t tile_pages, int64_t sm_mask0,
                              int64_t sm_mask1, int64_t sm_mask2, int64_t sm_mask3,
                              int64_t sm_count, TensorView output, int64_t stream_ptr) {
  using Traits = IndexerGemmTraits;
  CHECK_INPUT_AND_TYPE(page_table, dl_int32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(domain_prefix, dl_int32);
  CHECK_INPUT_AND_TYPE(counters, dl_int32);
  CHECK_INPUT_AND_TYPE(clear_counters, dl_int32);
  CHECK_INPUT_AND_TYPE(q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(output, dl_float32);
  CHECK_CUDA(k_cache);
  CHECK_INPUT_TYPE(k_cache, dl_uint8);
  int const batch = static_cast<int>(page_table.size(0));
  int const max_pages = static_cast<int>(page_table.size(1));
  TVM_FFI_ICHECK(page_table.ndim() == 2 && batch > 0 && max_pages > 0 &&
                 max_pages <= Traits::kMaximumPages)
      << "page_table must be [batch, max_pages <= " << Traits::kMaximumPages << "]";
  TVM_FFI_ICHECK(seq_lens.ndim() == 1 && seq_lens.size(0) == batch) << "seq_lens must be [batch]";
  TVM_FFI_ICHECK(domain_prefix.ndim() == 2 && domain_prefix.size(0) == 2 &&
                 domain_prefix.size(1) == batch + 1)
      << "domain_prefix must be [2, batch + 1]";
  TVM_FFI_ICHECK(counters.numel() == 2 && clear_counters.numel() == 2)
      << "counters must hold one int32 per domain";
  TVM_FFI_ICHECK(tile_pages >= 1 && tile_pages <= Traits::kPagesPerWorkTile)
      << "tile_pages must be in [1, " << Traits::kPagesPerWorkTile << "]";
  int64_t const query_length = q.ndim() == 3 ? q.size(0) / batch : 0;
  TVM_FFI_ICHECK(q.ndim() == 3 && query_length >= 1 && query_length <= Traits::kQueryLength &&
                 q.size(0) == static_cast<int64_t>(batch) * query_length && q.size(1) == 1 &&
                 q.size(2) == Traits::kHeadDim)
      << "q must have shape [batch * query_length, 1, 128]";
  TVM_FFI_ICHECK(k_cache.ndim() == 3 && k_cache.size(1) == Traits::kPageTokens &&
                 k_cache.size(2) * Traits::kPageTokens == Traits::kPageBytes &&
                 k_cache.stride(2) == 1 && k_cache.stride(1) == k_cache.size(2))
      << "k_cache must have shape [num_blocks, 128, 72] with contiguous pages";
  TVM_FFI_ICHECK(output.ndim() == 3 && output.size(0) == batch &&
                 output.size(1) == Traits::kQueryLength && output.size(2) == max_pages)
      << "scores must have shape [batch, 8, max_pages]";
  TVM_FFI_ICHECK(sm_count > 0 && sm_count <= 256) << "sm_count must be in [1, 256]";

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  IndexerGemmArguments arguments{};
  arguments.q_ptr = tensor_data<void const>(q);
  arguments.k_cache_ptr = tensor_data<void const>(k_cache);
  arguments.page_table_ptr = tensor_data<int32_t const>(page_table);
  arguments.kv_lengths_ptr = tensor_data<int32_t const>(seq_lens);
  arguments.output_ptr = tensor_data<float>(output);
  arguments.batch = batch;
  arguments.query_length = static_cast<int>(query_length);
  arguments.max_pages = max_pages;
  arguments.physical_pages = static_cast<int>(k_cache.size(0));
  arguments.page_stride_bytes = k_cache.stride(0);
  arguments.sm_count = static_cast<int>(sm_count);

  ParityIndexerGemmParams params{};
  cudaError_t status = Runner::to_underlying_arguments(arguments, params);
  TVM_FFI_ICHECK(status == cudaSuccess) << "TMA descriptor setup failed";
  params.domain_prefix_ptr = tensor_data<int32_t const>(domain_prefix);
  params.counters_ptr = tensor_data<int32_t>(counters);
  params.clear_counters_ptr = tensor_data<int32_t>(clear_counters);
  params.tile_pages = static_cast<int>(tile_pages);
  int64_t const masks[4] = {sm_mask0, sm_mask1, sm_mask2, sm_mask3};
  for (int i = 0; i < 4; ++i) {
    params.sm_domain_bits[2 * i] = static_cast<uint32_t>(static_cast<uint64_t>(masks[i]));
    params.sm_domain_bits[2 * i + 1] = static_cast<uint32_t>(static_cast<uint64_t>(masks[i]) >> 32);
  }
  status = ParityRunner::run(params, reinterpret_cast<cudaStream_t>(stream_ptr));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Q8KV4 parity indexer run failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(q8kv4_indexer_parity_resident_ctas_per_sm,
                              q8kv4_indexer_parity_resident_ctas_per_sm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(q8kv4_indexer_parity_max_tile_pages,
                              q8kv4_indexer_parity_max_tile_pages);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(q8kv4_indexer_parity_run, q8kv4_indexer_parity_run);
