from __future__ import annotations

from aiida_renormalizer.data.topology import TopologyData


def test_topology_data_roundtrip(aiida_profile):
    topology = {
        "schema": "topology_v1",
        "nodes": [
            {"node_id": 0, "children": [1, 2], "dofs": ["spin"]},
            {"node_id": 1, "children": [], "dofs": ["v_0"]},
            {"node_id": 2, "children": [], "dofs": ["v_1"]},
        ],
    }

    node = TopologyData.from_dict(topology)
    assert node.base.attributes.get("schema") == "topology_v1"
    assert node.base.attributes.get("n_nodes") == 3
    assert node.as_dict() == topology


def test_topology_data_accepts_subtree_assembly_schema(aiida_profile):
    topology = {
        "schema": "topology_v1",
        "subtrees": [
            {
                "subtree_id": "phonon",
                "builder": "binary_mctdh",
                "basis_dofs": ["v_0", "v_1"],
                "dummy_label": "phonon-dummy",
            }
        ],
        "assembly": [
            {
                "node_id": "root",
                "basis_items": [{"kind": "dof", "value": "spin"}, {"kind": "dummy", "label": "dummy"}],
                "children": ["phonon"],
            }
        ],
        "root": "root",
    }

    node = TopologyData.from_dict(topology)

    assert node.base.attributes.get("schema") == "topology_v1"
    assert node.base.attributes.get("n_nodes") == 2
    assert node.as_dict() == topology
