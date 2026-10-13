// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "q8kv4_indexer_sm107/params.hpp"

namespace q8kv4_indexer_sm107_parity {

// The one-head kernel's parameters plus the per-locality-domain schedule.
// Logical page j of every request lives in domain j % 2.
struct alignas(128) ParityIndexerGemmParams : q8kv4_indexer_sm107::IndexerGemmParams {
  // [2][batch + 1]: per domain, the exclusive prefix of the requests' tile
  // counts, then the domain's total.
  int32_t const *domain_prefix_ptr = nullptr;
  // This launch's claim counter per domain, and the previous launch's (zeroed
  // after the PDL wait).
  int32_t *counters_ptr = nullptr;
  int32_t *clear_counters_ptr = nullptr;
  int tile_pages = 0;
  // Bit s: domain of SM id s (%smid).
  uint32_t sm_domain_bits[8] = {};
};

} // namespace q8kv4_indexer_sm107_parity
