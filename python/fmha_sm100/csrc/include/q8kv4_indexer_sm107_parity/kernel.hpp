// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>

#include "q8kv4_indexer_sm107_parity/collective.hpp"

namespace q8kv4_indexer_sm107_parity {

template <class Traits, bool Postscale>
__global__ void __launch_bounds__(Traits::kThreads, 3)
    parity_indexer_gemm_kernel(const __grid_constant__ ParityIndexerGemmParams params) {
  using Collective = ParityIndexerGemmCollective<Traits, 0, Postscale>;
  using SharedStorage = typename Collective::SharedStorage;
  extern __shared__ __align__(1024) uint8_t shared_memory[];
  SharedStorage &storage = *reinterpret_cast<SharedStorage *>(shared_memory);
  Collective::run(params, storage);
}

template <class Traits> struct ParityIndexerGemmRunner {
  using Collective = ParityIndexerGemmCollective<Traits, 0, false>;
  static constexpr int kSharedStorageBytes = sizeof(typename Collective::SharedStorage);

  static cudaError_t initialize() {
    static std::once_flag once;
    static cudaError_t status = cudaSuccess;
    std::call_once(once, [] {
      for (auto kernel : {parity_indexer_gemm_kernel<Traits, false>,
                          parity_indexer_gemm_kernel<Traits, true>}) {
        if (status == cudaSuccess) {
          status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kSharedStorageBytes);
        }
      }
      int device = 0;
      int oversized_bytes = 0;
      if (status == cudaSuccess && cudaGetDevice(&device) == cudaSuccess &&
          cudaDeviceGetAttribute(&oversized_bytes, cudaDevAttrOversizedSharedMemoryPerBlock,
                                 device) == cudaSuccess &&
          oversized_bytes > 0) {
        cudaFuncSetAttribute(parity_indexer_gemm_kernel<Traits, false>,
                             cudaFuncAttributeSharedMemoryMode,
                             cudaSharedMemoryModeAllowOversizedSharedMemory);
        cudaFuncSetAttribute(parity_indexer_gemm_kernel<Traits, true>,
                             cudaFuncAttributeSharedMemoryMode,
                             cudaSharedMemoryModeAllowOversizedSharedMemory);
        cudaGetLastError();
      }
    });
    return status;
  }

  // Same grid as the one-head kernel: one wave of resident CTAs.
  static int resident_ctas_per_sm() {
    static std::once_flag once;
    static int ctas = 0;
    std::call_once(once, [] {
      if (initialize() != cudaSuccess ||
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &ctas, parity_indexer_gemm_kernel<Traits, true>, Traits::kThreads,
              kSharedStorageBytes) != cudaSuccess ||
          ctas < 1) {
        ctas = 1;
      }
    });
    return ctas;
  }

  static cudaError_t run(ParityIndexerGemmParams const &params, cudaStream_t stream) {
    cudaError_t const status = initialize();
    if (status != cudaSuccess) {
      return status;
    }
    dim3 const grid(params.sm_count * resident_ctas_per_sm(), 1, 1);
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = 1;
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[1].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = grid;
    launch_config.blockDim = dim3(Traits::kThreads, 1, 1);
    launch_config.dynamicSmemBytes = kSharedStorageBytes;
    launch_config.stream = stream;
    launch_config.attrs = attributes;
    launch_config.numAttrs = 2;
    cudaError_t const launch_status =
        params.query_length <= 4
            ? cudaLaunchKernelEx(&launch_config, parity_indexer_gemm_kernel<Traits, true>, params)
            : cudaLaunchKernelEx(&launch_config, parity_indexer_gemm_kernel<Traits, false>, params);
    return launch_status == cudaSuccess ? cudaGetLastError() : launch_status;
  }
};

} // namespace q8kv4_indexer_sm107_parity
