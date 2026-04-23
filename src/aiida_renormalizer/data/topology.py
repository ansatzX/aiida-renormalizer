"""TopologyData stores reusable tensor-network topology skeletons."""
from __future__ import annotations

from aiida.orm import Data

from aiida_renormalizer.data.utils import read_json_from_repository, write_json_to_repository


class TopologyData(Data):
    """AiiDA Data node for reusable topology metadata.

    This node stores only topology-level information (node relations and dof order),
    so multiple code-generation workfunctions can reuse it without rebuilding a new
    tree description from scratch.
    """

    @classmethod
    def from_dict(cls, topology: dict) -> "TopologyData":
        if not isinstance(topology, dict):
            raise TypeError("topology must be a dict")

        schema = str(topology.get("schema", "topology_v1"))
        if "nodes" in topology:
            if not isinstance(topology["nodes"], list):
                raise ValueError("topology['nodes'] must be a list")
            n_nodes = len(topology["nodes"])
        else:
            if "subtrees" not in topology or not isinstance(topology["subtrees"], list):
                raise ValueError("topology['subtrees'] must be a list")
            if "assembly" not in topology or not isinstance(topology["assembly"], list):
                raise ValueError("topology['assembly'] must be a list")
            if "root" not in topology:
                raise ValueError("topology['root'] is required")
            n_nodes = len(topology["subtrees"]) + len(topology["assembly"])

        node = cls()
        write_json_to_repository(node, topology, "topology.json")
        node.base.attributes.set("n_nodes", n_nodes)
        node.base.attributes.set("schema", schema)
        return node

    def as_dict(self) -> dict:
        return read_json_from_repository(self, "topology.json")
