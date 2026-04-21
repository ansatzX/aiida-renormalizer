"""Tests for Op/OpSum serialization and OpData."""
from __future__ import annotations

import json

import numpy as np
import pytest


class TestOpSerialization:
    def test_single_site_op_roundtrip(self):
        from renormalizer.model import Op

        from aiida_renormalizer.data.op import deserialize_op, serialize_op

        # Use a single-site operator symbol "b" instead of "b^\\dagger b"
        op = Op("b", "v0", 1.0)
        data = serialize_op(op)

        assert data["symbol"] == "b"
        assert data["dofs"] == ["v0"]
        assert data["factor"]["real"] == 1.0
        assert data["factor"]["imag"] == 0.0

        restored = deserialize_op(data)
        assert restored.symbol == op.symbol
        assert list(restored.dofs) == list(op.dofs)
        assert restored.factor == pytest.approx(op.factor)

    def test_multi_site_op_roundtrip(self):
        from renormalizer.model import Op

        from aiida_renormalizer.data.op import deserialize_op, serialize_op

        op = Op("b^\\dagger b", ["v0", "v1"], -0.5)
        data = serialize_op(op)

        assert data["dofs"] == ["v0", "v1"]

        restored = deserialize_op(data)
        assert list(restored.dofs) == ["v0", "v1"]
        assert restored.factor == pytest.approx(-0.5)

    def test_complex_factor(self):
        from renormalizer.model import Op

        from aiida_renormalizer.data.op import deserialize_op, serialize_op

        op = Op("b^\\dagger", "v0", 1.0 + 0.5j)
        data = serialize_op(op)

        assert data["factor"]["real"] == pytest.approx(1.0)
        assert data["factor"]["imag"] == pytest.approx(0.5)

        restored = deserialize_op(data)
        assert restored.factor == pytest.approx(1.0 + 0.5j)

    def test_json_safe(self):
        from renormalizer.model import Op

        from aiida_renormalizer.data.op import serialize_op

        op = Op("b^\\dagger b", "v0", np.float64(1.0))
        data = serialize_op(op)
        # Must not raise
        json.dumps(data)

    def test_explicit_quantum_numbers(self):
        """Op with explicit qn must roundtrip via to_tuple comparison."""
        from renormalizer.model import Op

        from aiida_renormalizer.data.op import deserialize_op, serialize_op

        op = Op("b^\\dagger", "v0", 1.0, qn=1)
        data = serialize_op(op)
        restored = deserialize_op(data)

        assert restored.to_tuple() == op.to_tuple()

    def test_integer_dof(self):
        """Reno allows arbitrary hashable dofs — integers must roundtrip."""
        from renormalizer.model import Op

        from aiida_renormalizer.data.op import deserialize_op, serialize_op

        op = Op("b^\\dagger b", 0, 1.0)
        data = serialize_op(op)
        restored = deserialize_op(data)

        assert list(restored.dofs) == list(op.dofs)
        assert restored.factor == pytest.approx(op.factor)


class TestOpSumSerialization:
    def test_roundtrip(self):
        from renormalizer.model import Op
        from renormalizer.model.op import OpSum

        from aiida_renormalizer.data.op import deserialize_opsum, serialize_opsum

        # Use single-site operators
        opsum = OpSum([Op("b", "v0", 1.0), Op("b", "v1", 1.5)])
        data = serialize_opsum(opsum)

        assert len(data) == 2

        restored = deserialize_opsum(data)
        assert len(restored) == 2
        assert restored[0].symbol == "b"
        assert list(restored[0].dofs) == ["v0"]


class TestOpData:
    def test_from_op_roundtrip(self, aiida_profile):
        from renormalizer.model import Op

        from aiida_renormalizer.data.op import OpData

        op = Op("b^\\dagger b", "v0", 1.0)
        node = OpData.from_op(op)
        node.store()

        assert node.base.attributes.get("op_type") == "Op"

        restored = node.load_op()
        assert restored.symbol == op.symbol
        assert restored.factor == pytest.approx(op.factor)

    def test_from_opsum_roundtrip(self, aiida_profile):
        from renormalizer.model import Op
        from renormalizer.model.op import OpSum

        from aiida_renormalizer.data.op import OpData

        opsum = OpSum([Op("b^\\dagger b", "v0", 1.0), Op("b^\\dagger b", "v1", 1.5)])
        node = OpData.from_opsum(opsum)
        node.store()

        assert node.base.attributes.get("op_type") == "OpSum"
        assert node.base.attributes.get("n_terms") == 2

        restored = node.load_opsum()
        assert len(restored) == 2

    def test_from_serialized_opsum_roundtrip(self, aiida_profile):
        from aiida_renormalizer.data.op import OpData

        serialized = [
            {
                "symbol": "b^\\dagger b",
                "dofs": ["v0"],
                "factor": {"real": 1.0, "imag": 0.0},
                "qn": None,
            },
            {
                "symbol": "b^\\dagger b",
                "dofs": ["v1"],
                "factor": {"real": 1.5, "imag": 0.0},
                "qn": None,
            },
        ]
        node = OpData.from_serialized_opsum(serialized)
        node.store()

        assert node.base.attributes.get("op_type") == "OpSum"
        assert node.base.attributes.get("n_terms") == 2
        assert node.base.attributes.get("dofs") == ["v0", "v1"]

        restored = node.load_opsum()
        assert len(restored) == 2
