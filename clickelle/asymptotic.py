"""Asymptotic mixed-mode relations for the dipole-mode stretching
(Mosser et al. 2015 formalism).

Parameters live in an AttrDict; the names used here:

    dnu     large frequency separation [uHz]
    dpi1    dipole period spacing DPi1 [s]
    q       coupling factor
    eps_p   p-mode phase offset
    eps_g   g-mode phase offset (only needed by make_f / make_zeta)
    d01     dipole small separation (in units of dnu)
    numax   frequency of maximum power [uHz]
    alpha_p p-mode curvature (only when constant_eps_p=False)
    q_k     linear frequency gradient of q (only when constant_q=False)
"""

import numpy as np


class AttrDict(dict):
    """A dict whose keys are also attributes: params.dnu == params['dnu']."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def eps_p_fn(nu, params, constant_eps_p=False):
    if constant_eps_p:
        return params.eps_p
    return params.alpha_p*((nu - params.numax)/params.dnu)**2.0 + params.eps_p


def q_fn(nu, params, constant_q=False):
    if constant_q:
        return params.q
    return params.q_k*(nu - params.numax) + params.q


def d01_fn(nu, params, constant_d01=True):
    if constant_d01:
        return params.d01
    return params.d01*(params.numax/nu)


def make_f(nu, params, constant_q=False):
    """Stretched FREQUENCY: undo the g-mode perturbation so pure p modes
    fall on a vertical ridge of the f-echelle."""
    theta_g = np.pi*(params.eps_g - 1/(nu*1e-6)/params.dpi1)
    return nu - params.dnu/np.pi * np.arctan(
        q_fn(nu, params, constant_q=constant_q) / np.tan(theta_g))


def make_tau(nu, params, constant_q=False, constant_eps_p=False,
             constant_d01=True):
    """Stretched PERIOD tau [s]: undo the p-mode perturbation so pure g
    modes fall on a vertical ridge of the tau-echelle folded at dpi1."""
    theta_p = np.pi*(nu/params.dnu
                     - (1/2
                        + eps_p_fn(nu, params, constant_eps_p=constant_eps_p)
                        + d01_fn(nu, params, constant_d01=constant_d01)))
    return 1/(nu*1e-6) + params.dpi1/np.pi * np.arctan(
        q_fn(nu, params, constant_q=constant_q) / np.tan(theta_p))


def make_zeta(nu, params, constant_eps_p=False, constant_d01=True,
              constant_q=False):
    """Asymptotic mixed-mode g fraction zeta = E_g/(E_p+E_g)."""
    theta_p = np.pi*(nu/params.dnu
                     - (1/2
                        + eps_p_fn(nu, params, constant_eps_p=constant_eps_p)
                        + d01_fn(nu, params, constant_d01=constant_d01)))
    theta_g = np.pi*(params.eps_g - 1/(nu*1e-6)/params.dpi1)
    q1 = q_fn(nu, params, constant_q=constant_q)
    return 1/(1 + params.dpi1/(params.dnu*1e-6) * (nu*1e-6)**2 / q1
              * np.sin(theta_g)**2 / np.cos(theta_p)**2)
