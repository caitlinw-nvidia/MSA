// Standalone binding for vllm-project/MSA PR #38's locality_domains.cu (compiled unchanged from
// python/fmha_sm100/prefill_q8kv4/csrc/api/): only the four locality functions, without the
// prefill attention kernel.

#include <torch/extension.h>

#include "prefill_attention_api.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("locality_domain_count", &fmha_sm100::prefill_q8kv4::locality_domain_count,
             py::arg("device"));
  module.def("locality_page_bytes", &fmha_sm100::prefill_q8kv4::locality_page_bytes,
             py::arg("device"));
  module.def("sm_locality_domains", &fmha_sm100::prefill_q8kv4::sm_locality_domains,
             py::arg("device"));
  module.def("alloc_locality_striped", &fmha_sm100::prefill_q8kv4::alloc_locality_striped,
             py::arg("bytes"), py::arg("device"));
}
