"""Dataset backends for temporal experiments, imported lazily for CLI safety."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "FORMULA_FAMILIES",
    "TemporalLogicDataset",
    "TemporalQuery",
    "build_temporal_logic_manifest",
    "encode_query",
    "evaluate_query",
    "UCIHARDataset",
    "download_uci_har",
    "prepare_uci_har",
]

_MODULE_BY_NAME = {
    "FORMULA_FAMILIES": ".temporal_logic",
    "TemporalLogicDataset": ".temporal_logic",
    "TemporalQuery": ".temporal_logic",
    "build_temporal_logic_manifest": ".temporal_logic",
    "encode_query": ".temporal_logic",
    "evaluate_query": ".temporal_logic",
    "UCIHARDataset": ".uci_har",
    "download_uci_har": ".uci_har",
    "prepare_uci_har": ".uci_har",
}


def __getattr__(name: str):
    if name not in _MODULE_BY_NAME:
        raise AttributeError(name)
    module = import_module(_MODULE_BY_NAME[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
