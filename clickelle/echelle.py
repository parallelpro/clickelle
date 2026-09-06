"""Echelle-diagram construction: fold a power spectrum into frequency or
(stretched) period echelle images."""

import numpy as np


def make_fold(x, ps, period, n_stack, n_element, idx, echelle_type='single',
              reverse=False):
    """Fold the (x, ps) series into n_stack rows of length period.

    idx holds the array indices where a new stack starts (wrap points of
    x % period). With echelle_type='replicated' each row shows two
    periods; reverse=True puts stack i+1 in the LEFT half of row i
    (used for the period echelle, where tau runs against frequency).
    """
    if echelle_type == 'single':
        z = np.zeros([n_stack, n_element])
        base = np.linspace(0, period, n_element, endpoint=False)
        for istack in range(n_stack):
            z[-istack-1, :] = np.interp(base, x[idx[istack]:idx[istack+1]],
                                        ps[idx[istack]:idx[istack+1]],
                                        period=period)
    else:
        z = np.zeros([n_stack, 2*n_element])
        base = np.linspace(0, 2*period, 2*n_element, endpoint=False)
        for istack in range(n_stack-1):
            if reverse:
                z[-istack-1, :] = np.r_[
                    np.interp(base[:n_element],
                              x[idx[istack+1]:idx[istack+2]],
                              ps[idx[istack+1]:idx[istack+2]],
                              period=period),
                    np.interp(base[:n_element],
                              x[idx[istack]:idx[istack+1]],
                              ps[idx[istack]:idx[istack+1]],
                              period=period)]
            else:
                z[-istack-1, :] = np.r_[
                    np.interp(base[:n_element],
                              x[idx[istack]:idx[istack+1]],
                              ps[idx[istack]:idx[istack+1]],
                              period=period),
                    np.interp(base[:n_element],
                              x[idx[istack+1]:idx[istack+2]],
                              ps[idx[istack+1]:idx[istack+2]],
                              period=period)]
    return base, z


def period_echelle(nu, ps, dpi, tau=None, fmin=None, fmax=None,
                   echelle_type='single', plot_with='imshow'):
    """Generate a (stretched) period echelle of a power spectrum.

    Parameters
    ----------
    nu : 1D array-like
        Frequencies in microhertz (uHz).
    ps : 1D array-like
        Power spectrum.
    dpi : float
        Length of each stack in seconds (the period spacing DPi1).
    tau : 1D array-like, optional
        Stretched period in seconds. Defaults to the unstretched period
        1/nu.
    fmin, fmax : float, optional
        Frequency range to fold. Default: the full range of nu.
    echelle_type : {'single', 'replicated'}
        'replicated' shows two periods per row.
    plot_with : {'imshow', 'contour'}
        'imshow' returns (z, extent) for fast interactive rendering;
        'contour' returns (z, x, y) for accurate publication plots.

    Examples
    --------
    >>> z, ext = period_echelle(nu, ps, dpi, tau=tau,
    ...                         fmin=numax-4*dnu, fmax=numax+4*dnu)
    >>> plt.imshow(z, extent=ext, aspect='auto', interpolation='nearest')
    """
    if fmin is None:
        fmin = np.nanmin(nu)
    if fmax is None:
        fmax = np.nanmax(nu)
    if tau is None:
        tau = np.copy(1/(nu*1e-6))

    m = (nu > fmin) & (nu < fmax)
    nu, ps, tau = nu[m], ps[m], tau[m]

    # wrap points of tau % dpi define the stacks
    idx = np.unique(np.concatenate([[0], np.where(np.diff(tau % dpi) > 0)[0],
                                    [len(tau)-2]]))

    resolution = np.median(np.abs(np.diff(1/(nu*1e-6))))
    n_stack = len(idx) - 1
    n_element = int(np.ceil(dpi/resolution))

    base, z = make_fold(tau, ps, dpi, n_stack, n_element, idx,
                        echelle_type=echelle_type, reverse=True)

    if plot_with == 'imshow':
        extent = (0, np.max(base), np.nanmin(nu), np.nanmax(nu))
        return z, extent
    elif plot_with == 'contour':
        x = base
        y = np.array([np.median(nu[idx[istack]:idx[istack+1]])
                      for istack in range(n_stack)])
        z = np.repeat(z, 2, axis=0)
        yl = y - np.diff(y, prepend=y[0])/2
        yu = y + np.diff(y, append=y[-1])/2
        y = np.sort(np.array([yl, yu]).reshape(-1))[::-1]
        return z, x, y
    return None


def frequency_echelle(nu, ps, dnu, f=None, fmin=None, fmax=None,
                      echelle_type='single', plot_with='imshow'):
    """Generate a (stretched) frequency echelle of a power spectrum.

    Parameters
    ----------
    nu : 1D array-like
        Frequencies in microhertz (uHz).
    ps : 1D array-like
        Power spectrum.
    dnu : float
        Length of each stack in uHz (the large separation).
    f : 1D array-like, optional
        Stretched frequency in uHz. Defaults to the unstretched nu.
    fmin, fmax : float, optional
        Frequency range to fold. Default: the full range of nu.
    echelle_type : {'single', 'replicated'}
        'replicated' shows two large separations per row.
    plot_with : {'imshow', 'contour'}
        'imshow' returns (z, extent); 'contour' returns (z, x, y).

    Examples
    --------
    >>> z, ext = frequency_echelle(nu, ps, dnu,
    ...                            fmin=numax-4*dnu, fmax=numax+4*dnu)
    >>> plt.imshow(z, extent=ext, aspect='auto', interpolation='nearest')
    """
    if fmin is None:
        fmin = np.nanmin(nu)
    if fmax is None:
        fmax = np.nanmax(nu)
    if f is None:
        f = np.copy(nu)

    fmin = 1e-4 if fmin < dnu else fmin - (fmin % dnu)

    m = (nu > fmin) & (nu < fmax)
    nu, ps, f = nu[m], ps[m], f[m]

    idx = np.unique(np.concatenate([[0], np.where(np.diff(f % dnu) < 0)[0],
                                    [len(f)-2]]))

    resolution = np.median(np.abs(np.diff(nu)))
    n_stack = len(idx) - 1
    n_element = int(np.ceil(dnu/resolution))

    base, z = make_fold(f, ps, dnu, n_stack, n_element, idx,
                        echelle_type=echelle_type, reverse=False)

    if plot_with == 'imshow':
        extent = (0, np.max(base), np.nanmin(nu), np.nanmax(nu))
        return z, extent
    elif plot_with == 'contour':
        x = base
        y = np.array([np.median(nu[idx[istack]:idx[istack+1]])
                      for istack in range(n_stack)])
        z = np.repeat(z, 2, axis=0)
        yl = y - np.diff(y, prepend=y[0])/2
        yu = y + np.diff(y, append=y[-1])/2
        y = np.sort(np.array([yl, yu]).reshape(-1))[::-1]
        return z, x, y
    return None
