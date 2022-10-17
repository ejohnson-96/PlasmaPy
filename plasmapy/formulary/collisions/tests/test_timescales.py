"""Test functionality of timescales in `plasmapy.formulary.analytical.collisions`."""
import numpy as np
import pytest

from astropy import units as u
from astropy.constants.si import c

from plasmapy.formulary.collisions import timescales as ts
from plasmapy.particles import Particle
from plasmapy.particles.exceptions import InvalidParticleError


class TestTimescales:
    _kwargs_single_valued = {
        "T": 8.3e-9 * u.T,
        "n_i": 4.0e5 * u.m ** -3,
        "ions": [Particle("H+"), Particle("He++")],
        "par_speeds": [500, 750] * u.m / u.s,
    }

    @pytest.mark.parametrize(
        "kwargs, _error",
        [
            ({**_kwargs_single_valued, "T": "wrong type"}, TypeError),
            ({**_kwargs_single_valued, "T": [8e-9, 8.5e-9] * u.K}, ValueError),
            ({**_kwargs_single_valued, "T": -1 * u.K}, ValueError),
            ({**_kwargs_single_valued, "T": 5 * u.m}, u.UnitTypeError),
            ({**_kwargs_single_valued, "n_i": "wrong type"}, TypeError),
            ({**_kwargs_single_valued, "n_i": -1 * u.m ** -3}, ValueError),
            ({**_kwargs_single_valued, "n_i": 6 * u.m}, u.UnitTypeError),
            ({**_kwargs_single_valued, "n_i": [4, 2, 3] * u.m ** -3}, ValueError),
            (
                    {**_kwargs_single_valued, "n_i": np.ones((2, 2)) * u.m ** -3},
                    ValueError,
            ),
            (
                    {**_kwargs_single_valued, "ions": {"not": "a particle"}},
                    InvalidParticleError,
            ),
            ({**_kwargs_single_valued, "par_speeds": 1.0 * u.m / u.s}, ValueError),
            ({**_kwargs_single_valued, "par_speeds": [1, 2, 3] * u.m / u.s}, ValueError),
            ({**_kwargs_single_valued, "par_speeds": [5, 6] * u.m / u.s}, u.UnitTypeError),
        ],
    )
    def test_raises(self, kwargs, _error):
        with pytest.raises(_error):
            ts.hellinger_2009(**kwargs)