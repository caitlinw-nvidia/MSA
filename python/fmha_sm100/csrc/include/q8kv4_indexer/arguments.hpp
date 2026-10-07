// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q8kv4_indexer {

// The shared decode plan (cute/src/sm100/decode_indexer_plan.py) writes, in int32 words:
//   [0, batch]                         prefix sum of each request's historical pages
//   [batch + 1, batch + workers + 1]   first global page of each worker, then the total
//   [batch + workers + 2, ...)         structure-of-arrays worker start fields: request
//                                      index, first logical page, first segment count
inline constexpr int kWorkerStartFields = 3;

struct IndexerGemmArguments {
  void const *q_ptr = nullptr;
  // vLLM packed NVFP4 page: packed E2M1 tokens followed by E4M3 group scales.
  void const *k_cache_ptr = nullptr;
  int32_t const *page_table_ptr = nullptr;
  int32_t const *kv_lengths_ptr = nullptr;
  int32_t const *scheduler_workspace_ptr = nullptr;
  float *output_ptr = nullptr;
  int batch = 0;
  int query_length = 0;
  int max_pages = 0;
  int physical_pages = 0;
  int64_t page_stride_bytes = 0;
  int sm_count = 0;
};

cudaError_t launch_indexer_gemm(IndexerGemmArguments const &arguments, cudaStream_t stream);

} // namespace q8kv4_indexer
