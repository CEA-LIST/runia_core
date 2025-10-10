""" Import all module's functions"""
from . import version
from .version import __version__

from . import feature_extraction
from . import baselines
from . import evaluation
from . import inference
from . import dimensionality_reduction
from . import rcnn
from . import yolov8
from . import llm_uncertainty

from .dimensionality_reduction import *

__all__ = [
    "baselines",
    "evaluation",
    "feature_extraction",
    "inference",
    "yolov8",
    "rcnn",
    "llm_uncertainty",
]
__all__ += dimensionality_reduction.__all__

# Try to catch deprecated or non-existing attributes
__deprecated_attrs__ = {}
__expired_functions__ = {}

def __getattr__(attr):
    # Warn for expired attributes, and return a dummy function
    # that always raises an exception.
    import warnings
    try:
        msg = __expired_functions__[attr]
    except KeyError:
        pass
    else:
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        def _expired(*args, **kwds):
            raise RuntimeError(msg)

        return _expired

    # Emit warnings for deprecated attributes
    try:
        val, msg = __deprecated_attrs__[attr]
    except KeyError:
        pass
    else:
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return val

    raise AttributeError("module {!r} has no attribute "
                         "{!r}".format(__name__, attr))
