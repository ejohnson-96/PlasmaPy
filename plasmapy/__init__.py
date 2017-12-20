import sys
import warnings

__minimum_python_version__ = '3.6'
__minimum_numpy_version__ = '1.13.0'
__minimum_astropy_version__ = '2.0.0'


def _split_version(version):
    return tuple(int(ver) for ver in version.split('.'))


def _min_required_version(required, current):  # coveralls: ignore
    r""" Return `True` if the current version meets the required
    minimum version and `False` if not or if not installed.

    Right now `required` and `current` are just '.' separated strings
    but it would be good to make this more general and accept modules.
    """
    return _split_version(current) >= _split_version(required)


def _check_numpy_version():  # coveralls: ignore
    r""" Make sure numpy in installed and meets the minimum version
    requirements."""
    required_version = False
    np_ver = None

    try:
        from numpy import __version__ as np_ver
        required_version = _min_required_version(__minimum_numpy_version__,
                                                 np_ver)
    except ImportError:
        pass

    if not required_version:
        raise ImportError(
            (f"NumPy {__minimum_numpy_version__} is required for "
             f"PlasmaPy. The currently installed version is {np_ver}"))

def _check_astropy_version():  # coveralls: ignore
    r""" Make sure astropy in installed and meets the minimum version
    requirements."""
    required_version = False
    ap_ver = None

    try:
        from astropy import __version__ as ap_ver
        required_version = _min_required_version(__minimum_astropy_version__,
                                                 ap_ver)
    except ImportError:
        pass

    if not required_version:
        raise ImportError(
            (f"Astropy {__minimum_astropy_version__} or above is required for "
             f"PlasmaPy.  The currently installed version is {ap_ver}"))


is_old_python = sys.version_info < _split_version(__minimum_python_version__)

if (is_old_python):  # coveralls: ignore
    warnings.warn("PlasmaPy does not support Python 3.5 and below")

_check_numpy_version()
_check_astropy_version()

try:
    from .classes import Plasma
    from . import classes
    from . import constants
    from . import atomic
    from . import mathematics
    from . import physics
    from . import utils
except ImportError:  # coveralls: ignore
    raise ImportError("Unable to load PlasmaPy subpackages.")
