"""Data node types for Renormalizer objects."""

__all__ = [
    "BasisSetData",
    "BasisTreeData",
    "ConfigData",
    "ModelData",
    "MPOData",
    "MPSData",
    "OpData",
    "TTNOData",
    "TTNSData",
]


def __getattr__(name: str):
    """Lazily import data classes to keep submodule imports lightweight."""
    if name == "BasisSetData":
        from aiida_renormalizer.data.basis import BasisSetData

        return BasisSetData
    if name == "BasisTreeData":
        from aiida_renormalizer.data.basis_tree import BasisTreeData

        return BasisTreeData
    if name == "ConfigData":
        from aiida_renormalizer.data.config import ConfigData

        return ConfigData
    if name == "ModelData":
        from aiida_renormalizer.data.model import ModelData

        return ModelData
    if name == "MPOData":
        from aiida_renormalizer.data.mpo import MPOData

        return MPOData
    if name == "MPSData":
        from aiida_renormalizer.data.mps import MPSData

        return MPSData
    if name == "OpData":
        from aiida_renormalizer.data.op import OpData

        return OpData
    if name == "TTNOData":
        from aiida_renormalizer.data.ttno import TTNOData

        return TTNOData
    if name == "TTNSData":
        from aiida_renormalizer.data.ttns import TTNSData

        return TTNSData
    raise AttributeError(name)
