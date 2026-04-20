#!/usr/bin/env python
"""Example usage of RenoScriptCalcJob.

This script demonstrates how to use the L3 ScriptedCalcJob for custom
multi-step workflows with AiiDA data provenance.
"""
from aiida import orm, load_profile
from aiida.engine import run_get_node

from aiida_renormalizer.data import ModelData, MpsData, MpoData
from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob


_PROFILE_LOADED = False


def _ensure_profile():
    global _PROFILE_LOADED
    if not _PROFILE_LOADED:
        load_profile()
        _PROFILE_LOADED = True


def example_1_simple_expectation():
    """Example 1: Calculate expectation value with custom script."""
    _ensure_profile()
    from renormalizer.model.basis import BasisSHO
    from renormalizer.model import Model
    from renormalizer.model.op import Op

    # 1. Create model
    basis = [BasisSHO(f"v{i}", omega=1.0, nbas=4) for i in range(4)]
    ham_terms = [Op(r"b^\dagger b", f"v{i}", 1.0) for i in range(4)]
    model = Model(basis, ham_terms)
    model_data = ModelData.from_model(model)

    # 2. Create MPS
    from renormalizer.mps import Mps
    mps = Mps.random(model, qntot=0, m_max=20)
    mps_data = MpsData.from_mps(mps, model_data)

    # 3. Create MPO
    from renormalizer.mps import Mpo
    mpo = Mpo(model)
    mpo_data = MpoData.from_mpo(mpo, model_data)

    # 4. Define custom script
    script = """
# Calculate expectation value
energy = mps.expectation(mpo)

# Save results
save_output_parameters({
    'energy': energy,
    'bond_dims': [int(d) for d in mps.bond_dims],
    'n_sites': len(mps),
})
"""

    # 5. Create code
    from aiida.orm import Code
    code = Code.get(label='renormalizer@localhost')

    # 6. Build and submit
    builder = RenoScriptCalcJob.get_builder()
    builder.code = code
    builder.script = orm.Str(script)
    builder.model = model_data
    builder.mps = mps_data
    builder.mpo = mpo_data

    # Run
    result, node = run_get_node(builder)
    print(f"Energy: {result['output_parameters']['energy']}")


def example_2_time_evolution():
    """Example 2: Custom time evolution workflow."""
    _ensure_profile()
    from renormalizer.model.basis import BasisSHO
    from renormalizer.model import Model
    from renormalizer.model.op import Op

    # 1. Create model
    basis = [BasisSHO(f"v{i}", omega=1.0, nbas=4) for i in range(6)]
    ham_terms = [Op(r"b^\dagger b", f"v{i}", 1.0) for i in range(6)]
    model = Model(basis, ham_terms)
    model_data = ModelData.from_model(model)

    # 2. Create initial MPS
    from renormalizer.mps import Mps
    mps = Mps.random(model, qntot=0, m_max=30)
    mps_data = MpsData.from_mps(mps, model_data)

    # 3. Create MPO
    from renormalizer.mps import Mpo
    mpo = Mpo(model)
    mpo_data = MpoData.from_mpo(mpo, model_data)

    # 4. Define custom evolution script
    script = """
# Time evolution with custom observables
import numpy as np

# Evolution parameters
n_steps = inputs['n_steps']
dt = inputs['dt']

# Track observables
energies = []
for i in range(n_steps):
    # Evolve
    mps.evolve(mpo, dt)

    # Measure
    e = mps.expectation(mpo)
    energies.append(e)

# Calculate additional observables
final_energy = energies[-1]
energy_drift = abs(energies[-1] - energies[0])

# Save outputs
save_output_parameters({
    'energies': energies,
    'final_energy': final_energy,
    'energy_drift': energy_drift,
    'n_steps': n_steps,
    'dt': dt,
})

# Save final MPS
save_mps(mps, 'output_mps.npz', f'Evolved state after {n_steps} steps')
"""

    # 5. Build and submit
    from aiida.orm import Code
    code = Code.get(label='renormalizer@localhost')

    builder = RenoScriptCalcJob.get_builder()
    builder.code = code
    builder.script = orm.Str(script)
    builder.model = model_data
    builder.mps = mps_data
    builder.mpo = mpo_data
    builder.inputs = orm.Dict({
        'n_steps': 100,
        'dt': 0.01,
    })

    # Run
    result, node = run_get_node(builder)
    print(f"Final energy: {result['output_parameters']['final_energy']}")
    print(f"Energy drift: {result['output_parameters']['energy_drift']}")


def example_3_multi_observable_scan():
    """Example 3: Scan multiple observables."""
    _ensure_profile()
    from renormalizer.model.basis import BasisSHO
    from renormalizer.model import Model
    from renormalizer.model.op import Op

    # 1. Create model
    basis = [BasisSHO(f"v{i}", omega=1.0, nbas=4) for i in range(4)]
    ham_terms = [Op(r"b^\dagger b", f"v{i}", 1.0) for i in range(4)]
    model = Model(basis, ham_terms)
    model_data = ModelData.from_model(model)

    # 2. Create MPS
    from renormalizer.mps import Mps
    mps = Mps.random(model, qntot=0, m_max=20)
    mps_data = MpsData.from_mps(mps, model_data)

    # 3. Define custom observable calculation script
    script = """
# Multi-observable calculation
import numpy as np
from renormalizer.mps import Mpo
from renormalizer.model.op import Op

observables = {}

# 1. Energy (from existing MPO)
observables['energy'] = mps.expectation(mpo)

# 2. Build and measure additional observables
# Example: spin correlation
for i in range(len(model)):
    for j in range(i+1, len(model)):
        # Build correlation operator
        op_terms = [Op(r"a^\\dagger a", (i, j), 1.0)]
        correlation_mpo = Mpo(model, op_terms)
        value = mps.expectation(correlation_mpo)
        observables[f'correlation_{i}_{j}'] = value

# 3. Entanglement entropy
entropies = [float(s) for s in mps.calc_entropy("bond")]
observables['entanglement_entropy'] = entropies
observables['mean_entropy'] = float(np.mean(entropies))

# Save all observables
save_output_parameters(observables)
"""

    # 4. Build and submit
    from aiida.orm import Code
    code = Code.get(label='renormalizer@localhost')

    builder = RenoScriptCalcJob.get_builder()
    builder.code = code
    builder.script = orm.Str(script)
    builder.model = model_data
    builder.mps = mps_data

    # Note: MPO is optional for this script
    from renormalizer.mps import Mpo
    mpo = Mpo(model)
    mpo_data = MpoData.from_mpo(mpo, model_data)
    builder.mpo = mpo_data

    # Run
    result, node = run_get_node(builder)
    print(f"Energy: {result['output_parameters']['energy']}")
    print(f"Mean entropy: {result['output_parameters']['mean_entropy']}")


def example_4_custom_analysis():
    """Example 4: Custom analysis without Reno data."""
    _ensure_profile()
    from aiida.orm import Code

    # Define a script that doesn't need Reno objects
    script = """
import numpy as np

# Generate spectral data (could come from other calculations)
frequencies = np.linspace(0, 10, 1000)
intensities = np.exp(-(frequencies - 5)**2 / 0.5)

# Perform analysis
peak_freq = frequencies[np.argmax(intensities)]
integrated_intensity = np.trapz(intensities, frequencies)
fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(0.5)

# Save results
save_output_parameters({
    'peak_frequency': peak_freq,
    'integrated_intensity': integrated_intensity,
    'fwhm': fwhm,
})

# Save full spectrum for plotting
save_data({
    'frequencies': frequencies.tolist(),
    'intensities': intensities.tolist(),
}, 'output_data.json')
"""

    # Build and submit (no model/mps/mpo needed)
    code = Code.get(label='python@localhost')

    builder = RenoScriptCalcJob.get_builder()
    builder.code = code
    builder.script = orm.Str(script)

    # Run
    result, node = run_get_node(builder)
    print(f"Peak frequency: {result['output_parameters']['peak_frequency']}")
    print(f"FWHM: {result['output_parameters']['fwhm']}")


if __name__ == '__main__':
    print("Example 1: Simple expectation calculation")
    print("=" * 50)
    # example_1_simple_expectation()

    print("\nExample 2: Time evolution workflow")
    print("=" * 50)
    # example_2_time_evolution()

    print("\nExample 3: Multi-observable scan")
    print("=" * 50)
    # example_3_multi_observable_scan()

    print("\nExample 4: Custom analysis")
    print("=" * 50)
    # example_4_custom_analysis()

    print("\nNote: Uncomment examples to run (requires AiiDA profile setup)")
