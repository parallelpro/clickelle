"""clickelle: click modes on an echelle diagram, then fit them.

An interactive front end for asteroseismic peakbagging: identify modes
by clicking on (stretched) echelle diagrams, refine the guesses, and fit
per-mode Lorentzian frequencies with detection statistics -- the part of
peakbagging that comes before any heavy Bayesian machinery.
"""

__version__ = '0.1.0'

from .asymptotic import AttrDict, make_f, make_tau, make_zeta
from .echelle import frequency_echelle, period_echelle
from .utils import load_spectrum, smooth
from .widget import EchelleWidget, launch

__all__ = ['AttrDict', 'EchelleWidget', 'frequency_echelle', 'launch',
           'load_spectrum', 'make_f', 'make_tau', 'make_zeta',
           'period_echelle', 'smooth']


def __getattr__(name):
    # fit tools pull in jax/optax; import lazily so the widget works
    # even where they are not installed
    if name in ('fit_modes', 'fit_star', 'fit_group', 'plot_groups'):
        from . import fit
        return getattr(fit, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
