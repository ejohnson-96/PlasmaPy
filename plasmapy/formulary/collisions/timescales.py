"""
Module of collision timescales related to collisions.
"""
__all__ = []

import astropy.units as u
from math import pi

from plasmapy.formulary.collisions import coulomb
from plasmapy.particles import Particle, ParticleList


class validate:

    def __init__(self):


    def ions(
        self,
        ions: (Particle, Particle),
    ):
        if not isinstance(ions, (list, tuple, ParticleList)):
            ions = [ions]
        ions = ParticleList(ions)

        if not all(failed := [ion.is_ion for ion in ions]):
            raise ValueError(
                "Particle(s) passed to 'ions' must be a charged ion. "
                "The following particle(s) is(are) not allowed "
                f"{[ion for ion, fail in zip(ions, failed) if not fail]}"
            )
        return ions


# Neutral close collisions
def function_name(
    ions: (Particle, Particle),
    speed: u.m/u.s,
    n_i: u.m ** -3,
):
    # Validate ions argument
    ions = validate.ions(ions)

    # Validate n_i argument
    if n_i.ndim != 0:
        raise ValueError(
            "Argument 'n_i' must be a single value, "
            f"instead got shape of {n_i.shape}"
        )

    # Validate speed argument
    if not isinstance(speed.value, (int, float)):
        raise TypeError(
            "Argument 'speed' is of incorrect type, float or integer "
            f"is required. Instead got type of {type(speed)}."
        )

    d0 = (ions[0].charge*ions[1].charge)/(ions[0].mass*speed.value**2)

    return 1 / (pi*n_i.value*speed.value*d0**2)

# Relaxation time



# Slowing down time

# Equipartion time