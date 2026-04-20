"""Tests for ConfigData serialization."""
from __future__ import annotations

import numpy as np
import pytest


class TestEvolveConfig:
    def test_roundtrip_defaults(self, aiida_profile):
        from renormalizer.utils.configs import EvolveConfig

        from aiida_renormalizer.data.config import ConfigData

        config = EvolveConfig()
        node = ConfigData.from_config(config)
        node.store()

        assert node.base.attributes.get("config_class") == "EvolveConfig"

        restored = node.load_config()
        assert type(restored).__name__ == "EvolveConfig"
        assert restored.method.name == config.method.name
        assert restored.adaptive == config.adaptive
        assert restored.guess_dt == pytest.approx(config.guess_dt)

    def test_roundtrip_custom(self, aiida_profile):
        from renormalizer.utils.configs import EvolveConfig, EvolveMethod

        from aiida_renormalizer.data.config import ConfigData

        config = EvolveConfig(method=EvolveMethod.tdvp_ps, adaptive=True, guess_dt=0.05)
        node = ConfigData.from_config(config)
        node.store()

        restored = node.load_config()
        assert restored.method == EvolveMethod.tdvp_ps
        assert restored.adaptive is True
        assert restored.guess_dt == pytest.approx(0.05)


class TestOptimizeConfig:
    def test_roundtrip_defaults(self, aiida_profile):
        from renormalizer.utils.configs import OptimizeConfig

        from aiida_renormalizer.data.config import ConfigData

        config = OptimizeConfig()
        node = ConfigData.from_config(config)
        node.store()

        restored = node.load_config()
        assert type(restored).__name__ == "OptimizeConfig"
        assert restored.method == config.method
        assert restored.algo == config.algo
        assert restored.nroots == config.nroots
        assert restored.e_rtol == pytest.approx(config.e_rtol)

    def test_roundtrip_custom_procedure(self, aiida_profile):
        from renormalizer.utils.configs import OptimizeConfig

        from aiida_renormalizer.data.config import ConfigData

        proc = [[10, 0.4], [20, 0.2], [30, 0]]
        config = OptimizeConfig(procedure=proc)
        node = ConfigData.from_config(config)
        node.store()

        restored = node.load_config()
        assert restored.procedure == proc


class TestCompressConfig:
    def test_roundtrip_defaults(self, aiida_profile):
        from renormalizer.utils.configs import CompressConfig, CompressCriteria

        from aiida_renormalizer.data.config import ConfigData

        config = CompressConfig()
        node = ConfigData.from_config(config)
        node.store()

        restored = node.load_config()
        assert type(restored).__name__ == "CompressConfig"
        assert restored.criteria == CompressCriteria.threshold
        assert restored.threshold == pytest.approx(1e-3)

    def test_roundtrip_vprocedure(self, aiida_profile):
        from renormalizer.utils.configs import CompressConfig

        from aiida_renormalizer.data.config import ConfigData

        config = CompressConfig(vprocedure=[[20, 0.4], [40, 0.2], [40, 0]])
        node = ConfigData.from_config(config)
        node.store()

        restored = node.load_config()
        assert restored.vprocedure == [[20, 0.4], [40, 0.2], [40, 0]]

    def test_roundtrip_dump_matrix_size_inf(self, aiida_profile):
        """np.inf must survive JSON roundtrip via None sentinel."""
        import numpy as np

        from renormalizer.utils.configs import CompressConfig

        from aiida_renormalizer.data.config import ConfigData

        config = CompressConfig(dump_matrix_size=np.inf)
        node = ConfigData.from_config(config)
        node.store()

        restored = node.load_config()
        assert np.isinf(restored.dump_matrix_size)

    def test_roundtrip_vguess_m_tuple(self, aiida_profile):
        """vguess_m is a tuple; AiiDA attributes store it as list — must convert back."""
        from renormalizer.utils.configs import CompressConfig

        from aiida_renormalizer.data.config import ConfigData

        config = CompressConfig(vguess_m=(10, 10))
        node = ConfigData.from_config(config)
        node.store()

        restored = node.load_config()
        assert restored.vguess_m == (10, 10)
        assert isinstance(restored.vguess_m, tuple)
