"""
This module contains functionality for calculating the timescales
for a range of configurations.
"""

__all__ = []

import numpy as np
from math import pi as pi
import astropy.units as u

from astropy.constants.si import eps0

from plasmapy.formulary.collisions import coulomb
from plasmapy.particles import Particle, ParticleList


def hellinger_2009(
        T: u.K,
        n_i: u.m ** -3,
        ions: (Particle, Particle),
        speeds: (u.m / u.s, u.m / u.s)
):
    # Validate ions argument
    if not isinstance(ions, (list, tuple, ParticleList)):
        ions = [ions]
    ions = ParticleList(ions)

    if not all(failed := [ion.is_ion and abs(ion.charge_number) > 0 for ion in ions]):
        raise ValueError(
            "Particle(s) passed to 'ions' must be a charged"
            " ion. The following particle(s) is(are) not allowed "
            f"{[ion for ion, fail in zip(ions, failed) if not fail]}"
        )

    # Validate ions dimension
    if len(ions) != 2:
        raise ValueError(
            f"Argument 'ions' can only take 2 inputs, received {ions}"
            f"with {len(ions)} inputs. Please try again."
        )

    # Validate speeds argument
    if not isinstance(speeds, (list, tuple)):
        speeds = [speeds]
    elif len(speeds) != 2:
        raise ValueError(
            "Argument 'speeds' can only take 2 inputs, received "
            f"{speeds} with {len(speeds)} inputs. Please try again."
        )

    if not all(isinstance(speed.value, (int, float)) for speed in speeds):
        raise TypeError(
            "Speed(s) passed to 'speeds' must be float or integer."
            "The following speed(s) is(are) not allowed "
            f"{[speed for speed, fail in zip(speeds, failed) if not fail]}"
        )

    # Validate temperature argument
    T = T.squeeze()
    if T.ndim != 0:
        raise ValueError(
            "Argument 'T' must be single values and not an array of"
            f" shape {T.shape}."
        )

    # Validate n_i argument
    n_i = n_i.squeeze()
    if n_i.ndim != 0:
        raise ValueError(
            "Argument 'n_i' must be single valued and not an array of"
            f" shape  {n_i.shape}."
        )

    v_par = np.sqrt((speeds[0].value**2 + speeds[1].value**2) / 2)

    return (((ions[0].charge.value**2)*(ions[1].charge.value**2)*n_i)/(12 * pi) * ((ions[0].mass.value) * (ions[1].mass.value)) * (eps0 ** 2) * (v_par ** 3))*coulomb.Coulomb_logarithm(T, n_i, ions)



inpts = {
    "T": 1000 * u.K,
    "n_i": 100 * u.m ** -3,
    "ions": (Particle('p'), Particle('He++')),
    "speeds": (3 * u.m/u.s, 6 * u.m/u.rad)
}

x = hellinger_2009(**inpts)

print(x)
