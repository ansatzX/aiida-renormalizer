"""WorkChains for aiida-renormalizer (code-generation only process layer)."""

from .bundle_runner import BundleRunnerWorkChain

ACTIVE_WORKCHAINS = [
    "BundleRunnerWorkChain",
]

OLD_WORKCHAINS = [
    "RenoRestartWorkChain",
    "TimeEvolutionWorkChain",
    "GroundStateWorkChain",
    "AbsorptionWorkChain",
    "ConvergenceWorkChain",
    "ThermalStateWorkChain",
    "KuboTransportWorkChain",
    "CustomPipelineWorkChain",
    "ParameterSweepWorkChain",
    "TemperatureSweepWorkChain",
    "BondDimensionSweepWorkChain",
    "FrequencySweepWorkChain",
    "CorrectionVectorWorkChain",
    "ChargeDiffusionWorkChain",
    "BathMPOPipelineWorkChain",
    "BathSpinBosonModelWorkChain",
    "OhmicRenormModesWorkChain",
    "ModelToMPOWorkChain",
    "MPOToInitialMPSWorkChain",
    "MPSDynamicsWorkChain",
    "SpinBosonWorkChain",
    "VibronicWorkChain",
    "TTNGroundStateWorkChain",
    "TTNTimeEvolutionWorkChain",
    "TTNMPSComparisonWorkChain",
    "TTNSymbolicDynamicsWorkChain",
    "SbmModelFromModesWorkChain",
    "TTNSymbolicModelWorkChain",
]

WORKCHAIN_PROCESS_STATUS = {
    **{name: "active" for name in ACTIVE_WORKCHAINS},
    **{name: "old" for name in OLD_WORKCHAINS},
}

__all__ = [
    "BundleRunnerWorkChain",
    "ACTIVE_WORKCHAINS",
    "OLD_WORKCHAINS",
    "WORKCHAIN_PROCESS_STATUS",
]
