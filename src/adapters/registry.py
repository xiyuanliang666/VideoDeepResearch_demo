"""Benchmark adapter registry."""

from __future__ import annotations

from . import browsecomp_vl, mmdeepresearch_bench, mmsearch_plus, vdr_bench, videodr

_ADAPTERS = {
    "mmsearch_plus": mmsearch_plus.adapt_sample,
    "vdr_bench": vdr_bench.adapt_sample,
    "videodr": videodr.adapt_sample,
    "browsecomp_vl": browsecomp_vl.adapt_sample,
    "mmdeepresearch_bench": mmdeepresearch_bench.adapt_sample,
}


def get_adapter(name: str):
    key = name.lower().replace("-", "_")
    if key not in _ADAPTERS:
        raise KeyError(f"Unknown adapter: {name}")
    return _ADAPTERS[key]


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)
