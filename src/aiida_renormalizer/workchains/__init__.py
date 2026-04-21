"""WorkChains for aiida-renormalizer."""
from .absorption import AbsorptionWorkChain
from .bath_mpo_pipeline import BathMPOPipelineWorkChain
from .ohmic_renorm_modes import OhmicRenormModesWorkChain
from .bath_spin_boson_model import BathSpinBosonModelWorkChain
from .charge_diffusion import ChargeDiffusionWorkChain
from .convergence import ConvergenceWorkChain
from .correction_vector import CorrectionVectorWorkChain
from .custom import CustomPipelineWorkChain
from .ground_state import GroundStateWorkChain
from .model_to_mpo import ModelToMPOWorkChain
from .mpo_to_initial_mps import MPOToInitialMPSWorkChain
from .mps_dynamics import MPSDynamicsWorkChain
from .restart import RenoRestartWorkChain
from .sbm_model_from_modes import SbmModelFromModesWorkChain
from .spin_boson import SpinBosonWorkChain
from .sweep import (
    BondDimensionSweepWorkChain,
    FrequencySweepWorkChain,
    ParameterSweepWorkChain,
    TemperatureSweepWorkChain,
)
from .thermal import ThermalStateWorkChain
from .time_evolution import TimeEvolutionWorkChain
from .transport import KuboTransportWorkChain
from .ttn_ground_state import TTNGroundStateWorkChain
from .ttn_mps_comparison import TTNMPSComparisonWorkChain
from .ttn_symbolic_dynamics import TTNSymbolicDynamicsWorkChain
from .ttn_symbolic_model import TTNSymbolicModelWorkChain
from .ttn_time_evolution import TTNTimeEvolutionWorkChain
from .vibronic import VibronicWorkChain

ACTIVE_WORKCHAINS = [
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
    "SbmModelFromModesWorkChain",
    "ModelToMPOWorkChain",
    "MPOToInitialMPSWorkChain",
    "MPSDynamicsWorkChain",
    "SpinBosonWorkChain",
    "VibronicWorkChain",
    "TTNGroundStateWorkChain",
    "TTNTimeEvolutionWorkChain",
    "TTNMPSComparisonWorkChain",
    "TTNSymbolicModelWorkChain",
    "TTNSymbolicDynamicsWorkChain",
]

WORKCHAIN_PROCESS_STATUS = {
    **{name: "active" for name in ACTIVE_WORKCHAINS},
}

__all__ = [
    "RenoRestartWorkChain",
    "TimeEvolutionWorkChain",
    "GroundStateWorkChain",
    "OhmicRenormModesWorkChain",
    "SbmModelFromModesWorkChain",
    "ModelToMPOWorkChain",
    "MPOToInitialMPSWorkChain",
    "MPSDynamicsWorkChain",
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
    "SpinBosonWorkChain",
    "VibronicWorkChain",
    "TTNGroundStateWorkChain",
    "TTNTimeEvolutionWorkChain",
    "TTNMPSComparisonWorkChain",
    "TTNSymbolicModelWorkChain",
    "TTNSymbolicDynamicsWorkChain",
    "ACTIVE_WORKCHAINS",
    "WORKCHAIN_PROCESS_STATUS",
]
