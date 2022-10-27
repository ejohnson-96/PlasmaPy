"""
This module contains functionality for calculating the timescales
for a range of configurations.
"""

__all__ = ["Hellinger", "USSR"]

import numpy as np
from math import pi as pi, gamma as gamma, factorial as fact
import astropy.units as u

from astropy.constants.si import eps0
from scipy.stats import gausshyper as gh

from plasmapy.formulary.collisions import coulomb
from plasmapy.particles import Particle, ParticleList
from plasmapy.utils.decorators import validate_quantities


class Hellinger:

    def __init__(self):
        self.CoulombCollisionsBiMaxwellian
        self.LangevinCoulombCollisionsBiMaxwellian
        self.IonCollisionalTransportCoefficents
        return

    # @validate_quantities(
    #    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    #    n_i={"can_be_negative": False},
    # )

    # Hellinger and Trávnícek 2009
    def CoulombCollisionsBiMaxwellian(
            self,
            T: u.K,
            n_i: u.m ** -3,
            ions: (Particle, Particle),
            par_speeds: (u.m / u.s, u.m / u.s)
    ):
        r"""
        Compute the collisional timescale as presented by :cite:t:`hellinger:2009`.

        Parameters
        ----------
        T : `~astropy.units.Quantity`
            The scalar temperature magnitude in units convertible to K.

        n_i : `~astropy.units.Quantity`
            Ion number density in units convertible to m\ :sup:`-3`.  Must
            be single value and should be the ion of prime interest.

        ions :  a `list` of length 2 containing :term:`particle-like` objects
            A list of length 2 with an instance of the
            :term:`particle-like` object representing the ion species in
            each entry. (e.g., ``"p"`` for protons, ``"D+"`` for deuterium,
             `["p", ``D+``]).

        par_speeds : a `list` of length 2 containing :term:`particle-like` objects
            A list of length 2 with an `~astropy.units.Quantity` representing
            the PARALLEL velocity with units of  in each entry. (e.g [
            500 * u.m / u.s, 745 * u.m / u.s]).

        Returns
        -------
        :math:`\tau` : `~astropy.units.Quantity`
            The collisional timescale in units of seconds.

        Raises
        ------
        `TypeError`
            If applicable arguments are not instances of
            `~astropy.units.Quantity` or cannot be converted into one.

        `ValueError`
            Number of particles in ``ions`` is not 2 or the input values
            are not valid particles

        `ValueError`
            If ``n_i`` or ``T`` is negative or not a single value.

        `TypeError`
            If ``n_i`` or ``T`` is not of type integer or float.

        `ValueError`
            Number of parallel speeds in``par_speeds`` is not 2.

        `TypeError`
            If the parallel speeds in ``par_speeds`` is not of type
            integer or float

        ~astropy.units.UnitTypeError
            If applicable arguments do not have units convertible to the
            expected units.

        Notes
        -----

        species s on species t.



        Example
        -------
        >>> from astropy import units as u
        >>> from plasmapy.particles import Particle
        >>> from plasmapy.formulary.collisions.timescales import Hellinger
        >>> inputs = {
        ...     "T": 8.3e-9 * u.T,
        ...     "n_i": 4.0e5 * u.m**-3,
        ...     "ions": [Particle("H+"), Particle("He+")],
        ...     "par_speeds":  [500, 750] * u.m /u.s,
        ... }
        >>> Hellinger.CoulombCollisionsBiMaxwellian(**inputs)
        <Quantity 1 / s>
        """

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

        # Validate par_speeds argument
        if len(par_speeds) != 2:
            raise ValueError(
                "Argument 'par_speeds' can only take 2 inputs, received "
                f"{par_speeds} with {len(par_speeds)} inputs."
            )
        else:
            for j in range(2):
                if not isinstance(par_speeds[j].value, (float, int)):
                    raise TypeError(
                        f"Argument {par_speeds[j].value} is of incorrect type, "
                        f"type int or float require and got {type(par_speeds[j].value)}"
                    )

        # Validate temperature argument
        if T.shape != ():
            raise ValueError(
                "Argument 'T' must be single value and not an array of"
                f" shape {T.shape}."
            )
        elif not isinstance(T.value, (int, float)):
            raise TypeError(
                f"Argument 'T' must be an integer or float, received {T} "
                f"with type of {type(T)}."
            )
        elif not T.value > 0:
            raise ValueError(
                f"Argument 'T' must be a positive argument, received "
                f"{T} of type {type(T)}."
            )

        # Validate n_i argument
        n_i = n_i.squeeze()
        if n_i.ndim != 0:
            raise ValueError(
                "Argument 'n_i' must be single value and not an array of"
                f" shape {n_i.shape}."
            )
        elif not isinstance(n_i.value, (int, float)):
            raise TypeError(
                "Argument 'n_i' must be an integer or float, received "
                f"{n_i} of type {type(n_i)}."
            )
        elif not n_i.value > 0:
            raise ValueError(
                f"Argument 'n_i' must be an positive argument, received "
                f"{n_i} of type {type(n_i)}."
            )
        v_par = np.sqrt((par_speeds[0].value ** 2 + par_speeds[1].value ** 2) / 2)

        a = ((ions[0].charge.value ** 2) * (ions[1].charge.value ** 2) * n_i.value)

        b = (12 * (pi ** 1.5)) * (ions[0].mass.value * ions[1].mass.value) * (eps0 ** 2) * (v_par ** 3)

        c = coulomb.Coulomb_logarithm(T, n_i, ions)

        return ((a / b.value) * c) / u.s

    # Hellinger and Trávnícek 2010
    def LangevinCoulombCollisionsBiMaxwellian(
            self,
            T_par: u.K,
            T_perp: u.K,
            n_i: u.m ** -3,
            ions: (Particle, Particle),
            par_speeds: (u.m / u.s, u.m / u.s)
    ):

        # Validate t_par and t_perp


        if T_par == 0:
            raise ValueError(
                f""
            )
        else:
            return Hellinger.CoulombCollisionsBiMaxwellian(T, n_i, ions, par_speeds) * 2 / 5 * gh(a=2, b=1.5, c=7 / 2, x=(1 - (T_perp / T_par)))

    # Hellinger 2016
    def IonCollisionalTransportCoefficents(
            self,
    ):

        return


class USSR:
    def ts_ii(self, T_i: u.K, n_i: u.m ** -3, particle: Particle):
        c = coulomb.Coulomb_logarithm(T_i, n_i, (particle, particle))

        if c == 0:
            raise ValueError(
                f"Coulomb logarithm returned {c} and will cause a "
                "divide by zero error. Please try again"
            )
        else:
            return (5 / 8) * np.sqrt((particle.mass * (T_i ** 3)) / pi) * (1 / ((particle.charge ** 4) * n_i * c))

    def ts_ee(self, T_e: u.K, n_e: u.m ** -3):
        return USSR.ts_ii(T_e, n_e, Particle("e-"))

    def ts_ei(self, T_i: u.K, T_e: u.K, n_i: u.m ** -3, n_e: u.m ** -3, particle: Particle):
        e = Particle("e-")
        c = coulomb.Coulomb_logarithm(T_i, n_i, (particle, e))

        if c == 0:
            raise ValueError(
                f"Coulomb logarithm returned {c} and will cause a "
                "divide by zero error. Please try again"
            )
        else:
            return (3 / 8) * ((particle.mass * T_i + e.mass * T_e) ** (1.5)) / (
                np.sqrt(2 * pi * e.mass * particle.mass)) / ((e.charge ** 2) * (particle.charge ** 2) * c * (n_i + n_e))


inpts = {
    "T": 1000 * u.K,
    "n_i": 100 * u.m ** -3,
    "ions": (Particle('p'), Particle('He++')),
    "par_speeds": [3 * u.m / u.s, 6 * u.m / u.s]
}

arg = inpts['par_speeds']

for i in arg:
    print(f"Instance of {i} of type {type(i)}.")
    print(f"Instance has unit of {i.unit}.")

H = Hellinger()
x = H.CoulombCollisionsBiMaxwellian(**inpts)
print(x)
