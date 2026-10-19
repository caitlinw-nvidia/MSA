"""Build PR #38's locality_domains.cu (unchanged, from the MSA tree) as a small extension.

Exposes locality_domain_count, locality_page_bytes, sm_locality_domains and
alloc_locality_striped exactly as fmha_sm100.prefill_q8kv4's extension does.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_pr38_locality(repo: Path):
    from torch.utils.cpp_extension import load

    api = Path(repo) / "python" / "fmha_sm100" / "prefill_q8kv4" / "csrc" / "api"
    build = Path(os.environ.get("PR38_LOCALITY_BUILD", HERE / "build_pr38_locality"))
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="pr38_locality",
        sources=[str(api / "locality_domains.cu"), str(HERE / "pr38_locality_binding.cpp")],
        extra_include_paths=[str(api)],
        extra_ldflags=["-lcuda"],
        build_directory=str(build),
        verbose=False,
    )
