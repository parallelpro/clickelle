"""Non-rotating mode frequency fits from the initial guesses.

For a star with saved initial guesses (modes_init.csv), fit one
Lorentzian per (n, l) mode -- no rotational splitting, no m components --
to the background-normalized spectrum.

Grouping
    Modes are sorted in frequency and broken into groups wherever the gap
    between two consecutive modes exceeds gap_frac * Dnu (default 0.25,
    so an l=2/l=0 pair at ~0.12 Dnu is never split). Each group is fitted
    independently over a window extending half the break threshold beyond
    its outermost modes, which keeps the free-parameter count per
    optimization small (3 per mode + 1 baseline).

Model and likelihood
    m(nu) = b + sum_i H_i / (1 + 4 ((nu - fc_i)/G_i)^2)
    chi^2-2dof NLL = sum [ln m + s/m], MLE by optax.adam on unconstrained
    parameters: fc = fc0 + shift_max * tanh(u) (shift_max set from the
    distance to the neighbouring modes), H and b in log, G = G_floor +
    exp(lnG) with G_floor = half a frequency bin. 1-sigma errors from the
    Hessian at the optimum (delta method through the transforms).

Fit quality (cheap evidence proxy)
    For every mode the group is refitted WITHOUT that mode and
        dlnL = lnL(with) - lnL(without)        (likelihood ratio;
                                                2*dlnL ~ chi2_3 under H0)
        lnK  = dlnL - (3/2) ln(N_bins)         (BIC-approximated log
                                                Bayes factor, 3 params)
    A mode is flagged detected when lnK > 0 (the data prefer including
    it even after the Occam penalty); lnK > 5 is a strong detection.
    This costs one extra (smaller) optimization per mode instead of a
    full evidence integral.

Outputs (fit_star with a session dir): modes_freqs.csv with columns
    l, fc_init, fc, e_fc, height, e_height, gamma, e_gamma, dlnL, lnK,
    detected, group, n_bins
plus groups_fit.png, one panel per group.
"""

import os

import numpy as np
import pandas as pd
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from .utils import L_STYLE, smooth, dnu_from_numax

GAP_FRAC = 0.25      # group break when the mode gap exceeds this * Dnu
SMOOTH_FRAC = 0.002  # display smoothing (fraction of Dnu) in the plots


# ---------------------------------------------------------------------------
# group fitting
# ---------------------------------------------------------------------------

def build_groups(fc, gap_uhz):
    """indices (into the sorted mode list) split at gaps > gap_uhz."""
    order = np.argsort(fc)
    breaks = np.where(np.diff(fc[order]) > gap_uhz)[0]
    return [seg for seg in np.split(order, breaks + 1) if len(seg)]


def pack_init(fc0, shift_max, nu, snr, gamma_floor, dnu):
    """initial unconstrained parameter vector [lnb, (u_f, lnH, lnG)*n]."""
    res = np.median(np.diff(nu))
    sm = smooth(snr, max(0.01*dnu/res, 1.0))
    b0 = max(float(np.median(snr)), 0.5)
    theta = [np.log(b0)]
    for f in fc0:
        h0 = max(float(np.interp(f, nu, sm)) - b0, 0.5)
        g0 = max(3*res, 0.005*dnu)
        theta += [0.0, np.log(h0), np.log(max(g0 - gamma_floor, 0.1*res))]
    return np.array(theta)


def make_nll(nu, snr, fc0, shift_max, gamma_floor):
    nu_j = jnp.asarray(nu)
    s_j = jnp.asarray(snr)
    fc0_j = jnp.asarray(fc0)
    sh_j = jnp.asarray(shift_max)

    def unpack(theta):
        b = jnp.exp(theta[0])
        p = theta[1:].reshape(-1, 3)
        fc = fc0_j + sh_j * jnp.tanh(p[:, 0])
        h = jnp.exp(p[:, 1])
        g = gamma_floor + jnp.exp(p[:, 2])
        return b, fc, h, g

    def model(theta):
        b, fc, h, g = unpack(theta)
        x = 2.0 * (nu_j[None, :] - fc[:, None]) / g[:, None]
        return b + jnp.sum(h[:, None] / (1.0 + x**2), axis=0)

    def nll(theta):
        m = model(theta)
        return jnp.sum(jnp.log(m) + s_j/m)

    return nll, model, unpack


def optimize(nll, theta0, steps, learning_rate=2e-2):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(jnp.asarray(theta0))

    @jax.jit
    def step(theta, opt_state):
        value, grads = jax.value_and_grad(nll)(theta)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        return optax.apply_updates(theta, updates), opt_state, value

    theta = jnp.asarray(theta0)
    best_theta, best_value = theta, np.inf
    for _ in range(steps):
        theta, opt_state, value = step(theta, opt_state)
        if value < best_value:
            best_theta, best_value = theta, float(value)
    return np.asarray(best_theta), best_value


def fit_group(nu, snr, fc0, shift_max, dnu, steps=2000, quality=True):
    """MLE + Hessian errors + drop-one quality for one mode group."""
    res = np.median(np.diff(nu))
    gamma_floor = 0.5*res
    nll, model, unpack = make_nll(nu, snr, fc0, shift_max, gamma_floor)
    theta0 = pack_init(fc0, shift_max, nu, snr, gamma_floor, dnu)
    theta, value = optimize(nll, theta0, steps)

    hess = np.asarray(jax.hessian(nll)(jnp.asarray(theta)))
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hess)
    err = np.sqrt(np.abs(np.diag(cov)))

    b, fc, h, g = (np.asarray(a) for a in unpack(jnp.asarray(theta)))
    p_err = err[1:].reshape(-1, 3)
    u_f = theta[1:].reshape(-1, 3)[:, 0]
    e_fc = shift_max * (1.0 - np.tanh(u_f)**2) * p_err[:, 0]
    e_h = h * p_err[:, 1]
    e_g = (g - gamma_floor) * p_err[:, 2]

    dlnl = np.full(len(fc0), np.nan)
    if quality:
        for i in range(len(fc0)):
            keep = [j for j in range(len(fc0)) if j != i]
            if keep:
                nll_i, _, _ = make_nll(nu, snr, fc0[keep], shift_max[keep],
                                       gamma_floor)
                t0_i = np.concatenate([theta[:1]] + [
                    theta[1 + 3*j: 4 + 3*j] for j in keep])
                _, value_i = optimize(nll_i, t0_i, max(steps//2, 500))
            else:               # single-mode group: null = baseline only
                s_med = float(np.median(snr))
                value_i = float(len(nu)*np.log(s_med) + np.sum(snr)/s_med)
            dlnl[i] = value_i - value          # NLL drops when mode helps

    lnk = dlnl - 1.5*np.log(len(nu))
    return dict(baseline=b, fc=fc, e_fc=e_fc, height=h, e_height=e_h,
                gamma=g, e_gamma=e_g, dlnL=dlnl, lnK=lnk,
                nll=value, model=np.asarray(model(jnp.asarray(theta))))


# ---------------------------------------------------------------------------
# per-star driver
# ---------------------------------------------------------------------------

def fit_modes(nu, snr, modes_init, dnu, star='star', gap_frac=GAP_FRAC,
              steps=2000, quality=True):
    """Fit every mode of one star; the core of the pipeline.

    Parameters
    ----------
    nu, snr : array-like
        Background-normalized power spectrum (S/B) and its frequencies
        [uHz].
    modes_init : DataFrame
        Initial guesses with columns 'l' and 'fc' (from the widget, or
        any other source).
    dnu : float
        Large separation [uHz]; sets the grouping scale and width floors.
    gap_frac : float
        Group break threshold as a fraction of dnu.
    steps : int
        adam steps per group fit.
    quality : bool
        Run the drop-one lnK refits (slower; set False to skip).

    Returns
    -------
    out : DataFrame
        One row per mode (see module docstring for columns).
    panels : list
        Per-group plot data, for plot_groups().
    """
    nu = np.asarray(nu, float)
    snr = np.asarray(snr, float)
    res = np.median(np.diff(nu))

    fc_all = modes_init['fc'].to_numpy(float)
    l_all = modes_init['l'].to_numpy(int)
    inside = (fc_all > nu[0] + dnu*0.05) & (fc_all < nu[-1] - dnu*0.05)
    if not inside.all():
        print(f'  {star}: {np.sum(~inside)} guess(es) outside the '
              f'spectrum dropped')
    fc_all, l_all = fc_all[inside], l_all[inside]

    # max frequency shift per mode: stay clear of the neighbours
    order = np.argsort(fc_all)
    fc_s = fc_all[order]
    d_nn = np.full(len(fc_s), np.inf)
    if len(fc_s) > 1:
        gaps = np.diff(fc_s)
        d_nn[:-1] = np.minimum(d_nn[:-1], gaps)
        d_nn[1:] = np.minimum(d_nn[1:], gaps)
    shift_s = np.maximum(np.minimum(0.35*d_nn, 0.05*dnu), 3*res)
    shift_all = np.empty_like(shift_s)
    shift_all[order] = shift_s

    gap_uhz = gap_frac * dnu
    groups = build_groups(fc_all, gap_uhz)
    pad = 0.5 * gap_uhz

    rows, panels = [], []
    for gi, idx in enumerate(groups):
        fc0 = fc_all[idx]
        lo, hi = fc0.min() - pad, fc0.max() + pad
        win = (nu >= lo) & (nu <= hi)
        fit = fit_group(nu[win], snr[win], fc0, shift_all[idx], dnu,
                        steps, quality)
        for k, i in enumerate(idx):
            rows.append({'l': l_all[i], 'fc_init': fc_all[i],
                         'fc': fit['fc'][k], 'e_fc': fit['e_fc'][k],
                         'height': fit['height'][k],
                         'e_height': fit['e_height'][k],
                         'gamma': fit['gamma'][k],
                         'e_gamma': fit['e_gamma'][k],
                         'dlnL': fit['dlnL'][k], 'lnK': fit['lnK'][k],
                         'detected': bool(fit['lnK'][k] > 0),
                         'group': gi, 'n_bins': int(np.sum(win))})
        panels.append((gi, nu[win], snr[win], fit, fc0, l_all[idx]))

    out = pd.DataFrame(rows).sort_values('fc').reset_index(drop=True)
    return out, panels


def fit_star(star, nu, snr, workdir='clickelle_out', dnu=None, numax=None,
             gap_frac=GAP_FRAC, steps=2000, quality=True):
    """Fit one star from its saved widget session and write the results.

    Reads <workdir>/<star>/modes_init.csv (and params_init.csv for dnu
    when not given); writes modes_freqs.csv and groups_fit.png next to
    them. Returns the modes DataFrame.
    """
    session_dir = os.path.join(workdir, str(star))
    modes = pd.read_csv(os.path.join(session_dir, 'modes_init.csv'))
    pfile = os.path.join(session_dir, 'params_init.csv')
    if os.path.exists(pfile):
        saved = pd.read_csv(pfile).iloc[0]
        if dnu is None and np.isfinite(saved.get('dnu', np.nan)):
            dnu = float(saved['dnu'])
        if numax is None and np.isfinite(saved.get('numax', np.nan)):
            numax = float(saved['numax'])
    if dnu is None:
        if numax is None:
            raise ValueError('no dnu available: pass dnu or numax, or '
                             'save one from the widget session')
        dnu = dnu_from_numax(numax)

    out, panels = fit_modes(nu, snr, modes, dnu, star=star,
                            gap_frac=gap_frac, steps=steps, quality=quality)
    out.to_csv(os.path.join(session_dir, 'modes_freqs.csv'), index=False)
    plot_groups(star, panels, dnu, os.path.join(session_dir,
                                                'groups_fit.png'))

    det = int(out['detected'].sum())
    print(f'{star}: {len(out)} modes in {len(panels)} groups, '
          f'{det} detected (lnK>0), median e_fc = '
          f'{out["e_fc"].median():.4f} uHz', flush=True)
    return out


def plot_groups(star, panels, dnu, outpath):
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=[10, max(2.2*n, 3)],
                             constrained_layout=True, squeeze=False)
    for ax, (gi, nu_w, snr_w, fit, fc0, l_w) in zip(axes[:, 0], panels):
        res = np.median(np.diff(nu_w))
        ax.plot(nu_w, snr_w, color='0.85', lw=0.6)
        ax.plot(nu_w, smooth(snr_w, max(SMOOTH_FRAC*dnu/res, 1.0)),
                color='k', lw=0.8)
        ax.plot(nu_w, fit['model'], 'C3-', lw=1.2)
        ax.axhline(fit['baseline'], color='C3', lw=0.6, ls=':')
        ymax = ax.get_ylim()[1]
        for f0, f, lnk, l in zip(fc0, fit['fc'], fit['lnK'], l_w):
            color, marker = L_STYLE[int(l)]
            ax.axvline(f0, color=color, lw=0.7, ls='--', alpha=0.35)
            ax.axvline(f, color=color, lw=1.0, alpha=0.8)
            ax.plot(f, 0.95*ymax, marker, color=color, ms=7, mfc='none',
                    mew=1.5, clip_on=False)
            if np.isfinite(lnk):
                ax.annotate(f'{lnk:.0f}', (f, 0.86*ymax), ha='center',
                            fontsize=7, color=color)
        ax.set_ylabel(f'group {gi}')
    axes[-1, 0].set_xlabel('frequency [$\\mu$Hz]')
    axes[0, 0].set_title(f'{star}: group fits (dashed=init, solid='
                         'fitted; number = lnK)')
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
