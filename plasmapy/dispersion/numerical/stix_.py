"""
This module contains functionality for calculating the numerical
solutions to the Stix cold plasma function.
"""

__all__ = ["stix"]

from typing import List, Union, Any

import astropy.units as u
import numpy as np

from astropy.constants.si import c
from sympy import Symbol
from sympy.solvers import solve

from plasmapy.formulary.frequencies import gyrofrequency, plasma_frequency
from plasmapy.particles import Particle, ParticleList
from plasmapy.utils.decorators import validate_quantities

c_si_unitless = c.value


@validate_quantities(
    B={"can_be_negative": False},
    n_i={"can_be_negative": False},
    k={"can_be_negative": False, "equivalencies": u.spectral()},
)
def stix(
    B: u.T,
    k: u.rad / u.m,
    ions: Particle,
    n_i: u.m ** -3,
    theta: u.rad,
):
    r"""
    Calculate the cold plasma function solution by using
    :cite:t:`bellan:2012`, this uses the numerical method to find
    (:math:`\omega`) dispersion relation provided by
    :cite:t:`stringer:1963`. This dispersion relation also assumes
    uniform magnetic field :math:`\mathbf{B_0}`, theta is the angle
    between the magnetic and the normal surface of the wave vector.
    For more information see the **Notes** section below.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to T.

    k : `~astropy.units.Quantity`, single valued or 1-D array
        Wavenumber in units convertible to rad/m.  Either single
        valued or 1-D array of length :math:`N`.

    ions: a single or `list` of :term:`particle-like` object(s)
        epresentation of the ion species (e.g., ``"p"`` for protons,
        ``"D+"`` for deuterium, ``["H+", "He+"]`` for hydrogen and
        helium, etc.).  The charge state for each species must be
        specified.

    n_i: `~astropy.units.Quantity`, single valud or 1-D array
        Ion number density in units convertible to m\ :sup:`-3`.  Must
        be single valued or equal length to ``ions``.

    theta: `~astropy.units.Quantity`, single valued or 1-D array
        The angle of propagation of the wave with respect to the
        magnetic field, :math:`\cos^{-1}(k_z / k)`, in units convertible
        to radians.  Either single valued or 1-D array of size
        :math:`M`.

    Returns
    -------
    omegas : Dict[`str`, `~astropy.units.Quantity`]
        A dictionary of computed wave frequencies in units rad/s.  The
        dictionary contains keys for each wave number, this will return
        an array  of value :math:`K x 4`.

    Raises
    ------
    `TypeError`
        If the argument is of an invalid type.
        `~astropy.units.UnitsError`
        If the argument is a `~astropy.units.Quantity` but is not
        dimensionless.

    `ValueError`
        If the number of frequencies for each ion isn't the same.

    `NoConvergence`
        If a solution cannot be found and the convergence failed to
        root.

    Notes
    -----
    The cold plasma function is defined by :cite:t:`stringer:1963`,
    this is equation 8 of :cite:t:`bellan:2012` and is presented below.
    It is assumed that the zero-order quantities are uniform in space
    and static in time; while the first-order quantities are assumed to
    vary as :math:`e^{\left [ i (\textbf{k}\cdot\textbf{r} - \omega t)
    \right ]}` :cite:t:`stix:1992`.

    .. math::
        (S\sin^{2}(\theta) + P\cos^{2}(\theta))(ck/\omega)^{4}
            - [
                RL\sin^{2}(\theta) + PS(1 + \cos^{2}(\theta))
            ](ck/\omega)^{2} + PRL = 0

    where,

    .. math::
        \mathbf{n} = \frac{c \mathbf{k}}{\omega}

    .. math::
        S = 1 - \sum \frac{\omega^{2}_{p\sigma}}{\omega^{2} -
            \omega^{2}_{c\sigma}}

    .. math::
        P = 1 - \sum_{\sigma} \frac{\omega^{2}_{p\sigma}}{\omega^{2}}

    .. math::
        D = \sum_{\sigma}
            \frac{\omega_{c\sigma}}{\omega}
            \frac{\omega^{2}_{p\sigma}}{\omega^{2} -
            \omega_{c\sigma}^{2}}

    The Cold plasma assumption, Following on section 1.6 of
    :cite:t:`bellan:2012` expresses following derived quantities as
    follows.

    .. math::
        R = S + D \hspace{1cm} L = S - D

    The equation is valid for all :math:`\omega` and :math:`k`
    providing that :math:`\frac{\omega}{k_{z}} >> \nu_{Te}` with
    :math:`\nu_{Ti}` and :math:`k_{x}r_{Le,i} << 1`.  The prediction of
    :math:`k \to 0` occurs when P, R or L cut off and predicts
    :math:`k \to \infty` for perpendicular propagation during wave
    resonance :math:`S \to 0`.

    Example
    -------
    >>> from astropy import units as u
    >>> from plasmapy.particles import Particle
    >>> from plasmapy.dispersion.numerical.stix_ import stix
    >>> inputs = {
    ...     "B": 8.3e-9 * u.T,
    ...     "k": 0.001 * u.rad / u.m,
    ...     "ions": [Particle("H+"), Particle("He+")],
    ...     "n_i": [4.0e5,2.0e5] * u.m**-3,
    ...     "theta": 30 * u.deg,
    >>> }
    >>> w = stix(**inputs)
    >>> print(w[0.001])

    """

    # validate ions argument
    if not isinstance(ions, (list, tuple)):
        ions = [ions]
    ions = ParticleList(ions)

    if not all(failed := [ion.is_ion and ion.charge_number > 0 for ion in ions]):
        raise ValueError(
            f"Particle(s) passed to 'ions' must be a positively charged"
            f" ion. The following particle(s) is(are) not allowed "
            f"{[ion for ion, fail in zip(ions, failed) if not fail]}"
        )

    # validate n_i argument
    if n_i.ndim not in (0, 1):
        raise ValueError(
            f"Argument 'n_i' must be a single valued or a 1D array of "
            f"size 1 or {len(ions)}, instead got shape of {n_i.shape}"
        )
    elif n_i.ndim == 1 and n_i.size != len(ions):
        raise ValueError(
            f"Argument 'n_i' and 'ions' need to be the same length, got"
            f" value of shape {len(ions)} and {len(n_i.shape)}."
        )

    n_i = n_i.value
    if n_i.ndim == 0:
        n_i = np.array([n_i] * len(ions))
    elif n_i.size == 1:
        n_i = np.repeat(n_i, len(ions))

    species = ions + [Particle("e-")]
    densities = np.zeros(n_i.size + 1)
    densities[:-1] = n_i
    densities[-1] = np.sum(n_i * ions.charge_number)

    # validate B argument
    B = B.squeeze()
    if B.ndim != 0:
        raise ValueError(
            f"Argument 'B' must be single valued and not an array of"
            f" shape  {B.shape}."
        )

    # validate k argument and dimension
    k = k.squeeze()
    if not (k.ndim == 0 or k.ndim == 1):
        raise ValueError(
            f"Argument 'k' needs to be a single value or a 1D array astropy Quantity,"
            f"got a value of shape {k.shape}."
        )
    if np.any(k <= 0):
        raise ValueError(f"Argument 'k' can not a or have negative value")
    if np.isscalar(k.value):
        k = np.array([k.value]) * u.rad / u.m

    # validate theta value
    theta = theta.squeeze()
    theta = theta.to(u.radian)
    if theta.ndim not in (0, 1):
        raise TypeError(
            f"Argument 'theta' needs to be a single value or 1D array "
            f" astropy Quantity, got array of shape {k.shape}."
        )
    #elif theta.ndim == 1 and theta.size != len(k):
    #    raise ValueError(
    #        f"Argument 'theta' and 'k' need to be the same length, got"
    #        f" value of shape {len(k)} and {len(theta.shape)}."
    #    )
    if np.isscalar(theta.value):
        theta = np.array([theta.value]) * u.rad

    wps = []
    wcs = []

    for par, dens in zip(species, densities.tolist()):
        wps.append(plasma_frequency(n=dens*u.m**-3, particle=par).value)
        wcs.append(gyrofrequency(B=B, particle=par, signed=False).value)
    wps = np.array(wps)
    wcs = np.array(wcs)

    # Stix method implemented
    w = Symbol("w")

    S = 1
    P = 1
    D = 0

    omegas = {}

    for i in range(len(species)):
        S += 0  # (wps[i] ** 2) / (w ** 2 + wcs[i] ** 2)
        P += 0  # (wps[i] / w) ** 2
        D += 0  # ((wps[i] ** 2) / (w ** 2 + wcs[i] ** 2)) * (
        # wcs[i] / w
        # )

    R = S + D
    L = S - D

    A = []
    B = []
    C = []

    for i in range(len(k)):
        A.append(S * (np.sin(theta[i].value) ** 2) + P * (np.cos(theta[i].value) ** 2))
        B.append(R * L * (np.sin(theta[i].value) ** 2) + P * S * (1 + np.cos(theta[i].value) ** 2))
        C.append(P * R * L)

    print("Note: Solution computation time may vary.")

    # solve the stix equation for single k value or an array
    for i in range(len(k)):
        omegas[k[i]] = {}
        for j in range(len(k)):
            eq = A[j] * ((c_si_unitless * k[i].value / w) ** 4) - B[j] * ((c_si_unitless * k[i].value / w) ** 2) + C[j]

            sol = solve(eq, w, warn=True)

            sol_omega_co = []
            for k in range(len(sol)):
                val = complex(sol[k]) * u.rad / u.s
                sol_omega_co.append(val)

            omegas[k[i]][theta[j]] = sol_omega_co
        omegas[k[i]] = omegas[k[i]].pop(i)
        print(f"{((i+j+k) / (2*len(k)+len(sol))) * 100:.2f} %", end="\r")

    return omegas