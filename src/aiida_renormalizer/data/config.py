"""ConfigData node for Renormalizer configuration objects."""
from __future__ import annotations

import enum
import typing as t

import numpy as np
from aiida.orm import Data

from aiida_renormalizer.data.utils import to_native


# Field extraction specs: (attribute_name, constructor_kwarg_name) pairs.
# When they differ (e.g. bond_dim_max_value vs max_bonddim), both are listed.

_EVOLVE_FIELDS = [
    "method", "adaptive", "guess_dt", "adaptive_rtol",
    "reg_epsilon", "ivp_rtol", "ivp_atol", "ivp_solver", "force_ovlp",
]

_OPTIMIZE_FIELDS = [
    "procedure", "method", "algo", "nroots",
    "e_rtol", "e_atol", "inverse",
]

_COMPRESS_FIELDS = [
    # (attr_name, constructor_kwarg)
    ("criteria", "criteria"),
    ("threshold", "threshold"),
    ("bond_dim_max_value", "max_bonddim"),  # attr != kwarg
    ("vmethod", "vmethod"),
    ("vprocedure", "vprocedure"),
    ("vrtol", "vrtol"),
    ("vguess_m", "vguess_m"),
    ("dump_matrix_size", "dump_matrix_size"),
    ("dump_matrix_dir", "dump_matrix_dir"),
]


class ConfigData(Data):
    """AiiDA Data node wrapping EvolveConfig, OptimizeConfig, or CompressConfig."""

    @classmethod
    def from_config(cls, config: t.Any) -> ConfigData:
        """Serialize a Reno config object into a ConfigData node.

        Serializes only plugin-supported compress parameters.
        """
        from renormalizer.utils.configs import CompressConfig, EvolveConfig, OptimizeConfig

        node = cls()
        config_class = type(config).__name__
        node.base.attributes.set("config_class", config_class)

        if isinstance(config, EvolveConfig):
            fields = {f: to_native(getattr(config, f)) for f in _EVOLVE_FIELDS}
        elif isinstance(config, OptimizeConfig):
            fields = {f: to_native(getattr(config, f)) for f in _OPTIMIZE_FIELDS}
        elif isinstance(config, CompressConfig):
            fields = {}
            for attr_name, _ in _COMPRESS_FIELDS:
                fields[attr_name] = to_native(getattr(config, attr_name))
        else:
            raise TypeError(f"Unsupported config type: {config_class}")

        node.base.attributes.set("fields", fields)
        return node

    def load_config(self) -> t.Any:
        """Reconstruct the original Reno config object."""
        from renormalizer.utils.configs import (
            CompressConfig,
            CompressCriteria,
            EvolveConfig,
            EvolveMethod,
            OptimizeConfig,
        )

        config_class = self.base.attributes.get("config_class")
        fields = dict(self.base.attributes.get("fields"))

        if config_class == "EvolveConfig":
            if fields.get("method") is not None:
                fields["method"] = EvolveMethod[fields["method"]]
            return EvolveConfig(**fields)

        elif config_class == "OptimizeConfig":
            # OptimizeConfig only accepts 'procedure' in __init__
            procedure = fields.pop("procedure", None)
            config = OptimizeConfig(procedure=procedure)
            for attr, val in fields.items():
                setattr(config, attr, val)
            return config

        elif config_class == "CompressConfig":
            # Remap attr names to constructor kwargs
            kwargs: dict[str, t.Any] = {}
            for attr_name, kwarg_name in _COMPRESS_FIELDS:
                val = fields.get(attr_name)
                if attr_name == "criteria" and val is not None:
                    val = CompressCriteria[val]
                elif attr_name == "dump_matrix_size" and val is None:
                    val = np.inf  # Restore sentinel
                elif attr_name == "vguess_m" and isinstance(val, list):
                    val = tuple(val)
                kwargs[kwarg_name] = val
            return CompressConfig(**kwargs)

        raise TypeError(f"Unknown config class: {config_class}")
