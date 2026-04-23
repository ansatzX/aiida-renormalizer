from __future__ import annotations

from renormalizer.model import Op

from aiida_renormalizer.data import BasisSpecData, OpData, OpSpecData


def test_op_spec_data_roundtrip(aiida_profile):
    op_specs = [
        {"symbol": "sigma_z", "dofs": "spin", "factor": "epsilon", "qn": 0},
        {"symbol": "sigma_z x", "dofs": ["spin", "v_0"], "factor": 0.125, "qn": [0, 0]},
    ]

    node = OpSpecData.from_list(op_specs)

    assert node.base.attributes.get("schema") == "op_spec_v1"
    assert node.base.attributes.get("n_terms") == 2
    assert node.base.attributes.get("dof_list") == ["spin", "v_0"]
    assert node.as_list() == op_specs


def test_basis_spec_data_roundtrip(aiida_profile):
    basis_specs = [
        {"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]},
        {"kind": "sho", "dof": "v_0", "omega": 0.5, "nbas": 8},
    ]

    node = BasisSpecData.from_list(basis_specs)

    assert node.base.attributes.get("schema") == "basis_spec_v1"
    assert node.base.attributes.get("n_items") == 2
    assert node.base.attributes.get("basis_kinds") == ["half_spin", "sho"]
    assert node.base.attributes.get("dof_list") == ["spin", "v_0"]
    assert node.as_list() == basis_specs


def test_basis_spec_data_accepts_list_forms(aiida_profile):
    basis_specs = [
        ["half_spin", "spin", [0, 0]],
        ["sho", "v_0", 0.5, 8],
    ]

    node = BasisSpecData.from_list(basis_specs)

    assert node.as_list() == [
        {"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]},
        {"kind": "sho", "dof": "v_0", "omega": 0.5, "nbas": 8},
    ]


def test_spec_data_supports_integer_and_tuple_dofs(aiida_profile):
    op_specs = [
        {"symbol": "Z + Z -", "dofs": [0, 0, 1, 2], "factor": -1.0, "qn": [[0, 0], [-1, 0], [0, 0], [1, 0]]},
        {"symbol": "a^dagger a", "dofs": [(0, 0), (1, 0)], "factor": 0.5, "qn": 0},
        {"symbol": "+ -", "dofs": ("L0", "p"), "factor": 1.25, "qn": 0},
    ]
    basis_specs = [
        {"kind": "half_spin", "dof": 0, "sigmaqn": [[0, 0], [1, 0]]},
        {"kind": "sho", "dof": (0, 0), "omega": 0.5, "nbas": 8},
        {"kind": "half_spin", "dof": ("L0", "p"), "sigmaqn": [0, 0]},
    ]

    op_node = OpSpecData.from_list(op_specs)
    basis_node = BasisSpecData.from_list(basis_specs)

    assert op_node.as_list() == op_specs
    assert basis_node.as_list() == basis_specs


def test_op_data_roundtrip_from_serialized_opsum(aiida_profile):
    term = Op(r"a^\dagger a", [0, 1], 0.5) * Op(r"b^\dagger+b", (1, 0), -0.7)
    node = OpData.from_op(term)

    serialized = node.as_serialized_opsum()

    assert node.base.attributes.get("op_type") == "Op"
    assert node.base.attributes.get("n_terms") == 1
    assert serialized
    assert isinstance(serialized, list)
