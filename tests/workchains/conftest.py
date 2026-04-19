"""Shared fixtures for workchain unit tests."""
from __future__ import annotations

from unittest.mock import Mock


class Namespace:
    """Lightweight attribute namespace that supports ``'key' in ns`` containment.

    AiiDA code uses ``'key' in self.inputs`` and ``'key' in calc.outputs`` to
    test whether optional ports were populated.  A plain ``Mock`` does not support
    ``__contains__``, so this class fills that role while still allowing free
    attribute access.

    Pass keyword arguments to pre-populate::

        ns = Namespace(model=mock_model, code=mock_code)
        assert 'model' in ns
        assert 'mpo' not in ns
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        keys = ", ".join(self.__dict__)
        return f"Namespace({keys})"


def make_workchain(workchain_cls):
    """Create a bare WorkChain instance for unit-testing step methods.

    This bypasses the full AiiDA Process ``__init__`` (which requires a runner,
    profile storage, and validated inputs) by:

    1. Creating a thin subclass that shadows the read-only ``inputs`` and ``ctx``
       properties inherited from ``Process`` / ``WorkChain`` with plain class
       attributes, so they can be freely assigned on the instance.
    2. Using ``object.__new__`` to allocate the instance without triggering the
       plumpy ``StateMachine`` metaclass ``__call__``.
    3. Pre-wiring ``report``, ``out``, and ``submit`` as ``Mock`` objects so
       callers don't have to set up every piece of infrastructure.

    Returns an instance whose methods (``setup``, ``run_dmrg``, ``finalize``,
    etc.) can be invoked directly in tests after setting ``wc.inputs``,
    ``wc.ctx``, etc.
    """
    # Shadow read-only descriptors inherited from the AiiDA process hierarchy
    testable_cls = type(
        f"_Testable{workchain_cls.__name__}",
        (workchain_cls,),
        {"inputs": None, "ctx": None},
    )
    wc = object.__new__(testable_cls)
    wc.inputs = Namespace()
    wc.ctx = Namespace()
    wc.report = Mock()
    wc.out = Mock()
    wc.submit = Mock(return_value=Mock())
    return wc
