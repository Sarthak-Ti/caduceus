"""Registry of optimization variants applied to a built CaduceusMixerModel.

Each entry is a function `model -> model` performing in-place surgery so the same
benchmark driver can time any variant. Variants that change numerics (rather than
just the schedule of the same math) are marked NON-EQUIVALENT and are validated
separately in tests/test_equivalence.py.
"""

from typing import Callable, Dict, Optional

_REGISTRY: Dict[str, Optional[Callable]] = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_variant(name: str):
    if name in ("baseline", "none", None):
        return None
    if name not in _REGISTRY:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(_REGISTRY) + ['baseline']}")
    return _REGISTRY[name]


def list_variants():
    return ["baseline"] + sorted(_REGISTRY)


# --- variants are registered here as they are implemented -------------------
# (populated by caduceus.bimamba_opt once the profiling round identifies targets)
try:
    from caduceus import bimamba_opt  # noqa: F401  registers its variants on import
except ImportError:
    pass
