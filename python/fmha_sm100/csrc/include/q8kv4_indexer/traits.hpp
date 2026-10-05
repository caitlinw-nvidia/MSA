// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include "cutlass/cutlass.h"

namespace q8kv4_indexer {

// Ported from MiniMax-AI/MSA nv_dev inference/msa_v1/indexer/decode/q8kv4 (see
// vendor_manifest.json). Including the tile in the type keeps modules from sharing
// launch-initialization state.
template <int NumIndexHeads, int QueryColumns> struct IndexerGemmTraitsForHeads {
  static constexpr int kNumIndexHeads = NumIndexHeads;
  static constexpr int kPageWorkerGroups = 4;
  static constexpr int kPageWorkerWarps = 4;
  static constexpr int kAccumulatorStages = kPageWorkerGroups;
  static constexpr int kDequantStages = 4;
  static constexpr int kDequantGroups = kPageWorkerGroups;
  static constexpr int kDequantWarpsPerGroup = kPageWorkerWarps;
  static constexpr int kConsumerGroups = kPageWorkerGroups;
  static constexpr int kConsumerWarpsPerGroup = kPageWorkerWarps;
  static constexpr int kQueryColumns = QueryColumns;
  static_assert(kQueryColumns >= 16 && kQueryColumns <= 64 && kQueryColumns % 16 == 0);
  static constexpr int kMaxQueryLength = 16;
  static constexpr int kHeadDim = 128;
  static constexpr int kPageTokens = 128;
  static constexpr int kScaleGroupSize = 16;
  static constexpr int kScaleGroups = kHeadDim / kScaleGroupSize;
  static constexpr int kPackedKBytes = kPageTokens * kHeadDim / 2;
  static constexpr int kScaleBytes = kPageTokens * kScaleGroups;
  static constexpr int kPageBytes = kPackedKBytes + kScaleBytes;
  static constexpr int kThreads =
      (kPageWorkerGroups * kPageWorkerWarps + 2) * cutlass::NumThreadsPerWarp;
  static constexpr int kMaxPagesPerCta = 8;
  static constexpr int kMaximumPages = 8192;
};

// The single-index-head decode path: eight MTP tokens fill half of the smallest legal
// TMEM-source MMA width. Two and four heads run the CuTe DSL kernel
// (cute/src/sm100/q8kv4_indexer_decode.py).
using IndexerGemmTraits = IndexerGemmTraitsForHeads<1, 16>;

} // namespace q8kv4_indexer
