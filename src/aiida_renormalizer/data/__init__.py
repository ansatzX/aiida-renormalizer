"""Data node types for Renormalizer objects."""

__all__ = [
    "BasisSpecData",
    "BasisSetData",
    "BasisTreeData",
    "ConfigData",
    "ModelData",
    "MPOData",
    "MPSData",
    "OpData",
    "OpSpecData",
    "TensorNetworkLayoutData",
    "TopologyData",
    "TTNOData",
    "TTNSData",
]


def __getattr__(name: str):
    """Lazily import data classes to keep submodule imports lightweight."""
    if name == "BasisSetData":
        from aiida_renormalizer.data.basis import BasisSetData

        return BasisSetData
    if name == "BasisSpecData":
        from aiida_renormalizer.data.basis_spec import BasisSpecData

        return BasisSpecData
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
    if name == "OpSpecData":
        from aiida_renormalizer.data.op_spec import OpSpecData

        return OpSpecData
    if name == "TensorNetworkLayoutData":
        from aiida_renormalizer.data.tensor_network_layout import TensorNetworkLayoutData

        return TensorNetworkLayoutData
    if name == "TopologyData":
        from aiida_renormalizer.data.topology import TopologyData

        return TopologyData
    if name == "TTNOData":
        from aiida_renormalizer.data.ttno import TTNOData

        return TTNOData
    if name == "TTNSData":
        from aiida_renormalizer.data.ttns import TTNSData

        return TTNSData
    raise AttributeError(name)
