"""Shared helpers: spectrum loading, smoothing, per-degree plot styles."""

import os

import numpy as np
import pandas as pd

# per-degree marker (color, symbol), shared by the widget and fit plots
L_STYLE = {0: ('tab:blue', 'o'), 1: ('tab:red', '^'),
           2: ('tab:green', 's'), 3: ('tab:purple', 'D')}

FREQ_CANDIDATES = ('freq_uhz', 'freq', 'frequency', 'nu')
POWER_CANDIDATES = ('snr', 'power', 'psd', 'ps', 'sb')


def smooth(power, n_bins):
    """Boxcar smooth by n_bins (>= 1)."""
    n = max(int(n_bins), 1)
    return np.convolve(power, np.ones(n)/n, mode='same')


def load_spectrum(path, freq_col=None, power_col=None):
    """Load a power spectrum table (CSV or parquet by extension).

    Columns are resolved from freq_col/power_col if given, otherwise by
    common names (freq_uHz/freq/frequency/nu and snr/power/psd/ps), and
    finally by falling back to the first two columns. Returns (nu, power)
    sorted by frequency, with non-finite rows dropped.

    The spectrum should be BACKGROUND-NORMALIZED (S/B): the fit assumes
    a flat unit continuum with chi^2 2-dof statistics about the model.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.parquet', '.pq'):
        table = pd.read_parquet(path)
    else:
        table = pd.read_csv(path)

    lower = {c.lower(): c for c in table.columns}

    def resolve(requested, candidates, fallback_index):
        if requested is not None:
            if requested not in table.columns:
                raise KeyError(f'column {requested!r} not in {path} '
                               f'(has {list(table.columns)})')
            return requested
        for name in candidates:
            if name in lower:
                return lower[name]
        return table.columns[fallback_index]

    fcol = resolve(freq_col, FREQ_CANDIDATES, 0)
    pcol = resolve(power_col, POWER_CANDIDATES, 1)

    nu = table[fcol].to_numpy(float)
    power = table[pcol].to_numpy(float)
    good = np.isfinite(nu) & np.isfinite(power)
    order = np.argsort(nu[good])
    return nu[good][order], power[good][order]


def dnu_from_numax(numax):
    """Scaling-relation guess for the large separation [uHz]."""
    return 0.263 * numax**0.772
