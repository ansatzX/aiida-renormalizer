"""Snapshot native Reno evolution settings without losing user-supplied values."""

import math
from numbers import Real

from renormalizer.utils import EvolveConfig, EvolveMethod

TTN_METHODS = {"tdvp_ps", "tdvp_ps2", "tdvp_vmf", "prop_and_compress_tdrk4"}
CONSTRUCTOR_FIELDS = (
    "adaptive",
    "guess_dt",
    "adaptive_rtol",
    "reg_epsilon",
    "ivp_rtol",
    "ivp_atol",
    "ivp_solver",
    "force_ovlp",
)
EXTRA_FIELDS = ("tdvp_cmf_midpoint", "tdvp_cmf_c_trapz", "vmf_auto_switch")


def _plain_values(value):
    """Compare native RK/Taylor arrays without an extra runtime NumPy import."""
    return value.tolist() if hasattr(value, "tolist") else value


def validate_evolve_settings(settings):
    """Check the recorded snapshot and reconstruct its native constructor inputs."""
    expected = {"method", "rk_solver", "taylor_order", *CONSTRUCTOR_FIELDS, *EXTRA_FIELDS}
    if set(settings) != expected:
        raise ValueError("evolution settings must contain the complete supported EvolveConfig")
    if settings["method"] not in TTN_METHODS:
        raise ValueError(f"unsupported TTN evolution method: {settings['method']}")
    for name in ("adaptive", "force_ovlp", *EXTRA_FIELDS):
        if type(settings[name]) is not bool:
            raise ValueError(f"EvolveConfig.{name} must be a bool")
    for name in ("guess_dt", "adaptive_rtol", "reg_epsilon", "ivp_rtol", "ivp_atol"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"EvolveConfig.{name} must be a positive finite real number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"EvolveConfig.{name} must be a positive finite real number")
    if type(settings["taylor_order"]) is not int or settings["taylor_order"] < 1:
        raise ValueError("EvolveConfig Taylor order must be a positive integer")
    if not isinstance(settings["ivp_solver"], str) or not settings["ivp_solver"]:
        raise ValueError("EvolveConfig.ivp_solver must be a non-empty string")
    kwargs = {name: settings[name] for name in CONSTRUCTOR_FIELDS}
    kwargs.update(rk_solver=settings["rk_solver"], taylor_order=settings["taylor_order"])
    method = EvolveMethod[settings["method"]]
    # These TTN routes do not consume the full MPS-oriented EvolveConfig surface.
    # Fail on requested changes that native TTN would silently ignore.
    defaults = EvolveConfig(method=method)
    effective = (
        {"ivp_rtol", "ivp_atol", "reg_epsilon"} if method == EvolveMethod.tdvp_vmf else set()
    )
    default_settings = {name: getattr(defaults, name) for name in CONSTRUCTOR_FIELDS + EXTRA_FIELDS}
    default_settings.update(
        rk_solver=defaults.rk_config.method, taylor_order=defaults.taylor_config.order
    )
    for name, default in default_settings.items():
        if name not in effective and settings[name] != default:
            raise ValueError(
                f"EvolveConfig.{name} is not used by Reno TTN {method.name}; "
                "leave it at its default instead of requesting an ineffective change"
            )
    return kwargs


def snapshot_evolve_config(config: EvolveConfig) -> dict:
    """Record constructor values and mutable flags, including resolved defaults.

    Custom RK tableaus/Taylor coefficients and unknown attributes are rejected:
    reconstructing them as an ordinary named Reno configuration would lose data.
    """
    if type(config) is not EvolveConfig:
        raise TypeError("evolve_config must be a native Renormalizer EvolveConfig")
    expected = {"method", "rk_config", "taylor_config", *CONSTRUCTOR_FIELDS, *EXTRA_FIELDS}
    if set(vars(config)) != expected:
        raise ValueError("unsupported EvolveConfig attributes; refusing to discard settings")
    if not isinstance(config.method, EvolveMethod):
        raise ValueError("EvolveConfig.method must be a native EvolveMethod")
    settings = {"method": config.method.name}
    settings.update({name: getattr(config, name) for name in CONSTRUCTOR_FIELDS + EXTRA_FIELDS})
    settings.update(rk_solver=config.rk_config.method, taylor_order=config.taylor_config.order)
    kwargs = validate_evolve_settings(settings)
    reference = EvolveConfig(method=config.method, **kwargs)
    for actual, canonical in (
        (config.rk_config, reference.rk_config),
        (config.taylor_config, reference.taylor_config),
    ):
        if type(actual) is not type(canonical) or set(vars(actual)) != set(vars(canonical)):
            raise ValueError("custom RK/Taylor configuration cannot be reconstructed losslessly")
        for name, value in vars(canonical).items():
            current = getattr(actual, name)
            if name == "tableau":
                equal = len(current) == len(value) and all(
                    _plain_values(a) == _plain_values(b) for a, b in zip(current, value)
                )
            else:
                equal = _plain_values(current) == _plain_values(value)
            if not equal:
                raise ValueError("custom RK/Taylor coefficients cannot be reconstructed losslessly")
    return settings
