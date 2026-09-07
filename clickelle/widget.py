"""Interactive echelle widget for mode identification / initial guesses.

Loads a background-normalized spectrum (S/B), smooths it by a fraction
of Dnu, and shows

  left   : replicated frequency echelle with the mode guesses overlaid
           -- l=0 blue circles, l=1 red triangles, l=2 green squares,
           l=3 purple diamonds
  bottom : the smoothed S/B spectrum with the guesses as vertical lines
  right  : (after pressing "stretching") a replicated STRETCHED PERIOD
           echelle (tau) with the same modes overlaid -- the l=1 mixed
           modes should fall on a vertical ridge when DPi1/q/d01/eps_p
           are right

Interactions
  Dnu slider      : re-fold the frequency echelle live
  eps_p slider    : moves the dashed l=0 ridge guide on the frequency
                    echelle (at x = (eps_p mod 1)*Dnu) and feeds Theta_p
                    of the stretching; tune until the guide sits on the
                    l=0 ridge
  l radio buttons : degree assigned to newly added modes
  add / remove    : what a click on the FREQUENCY echelle does
  refine          : snap every guess to the nearest local maximum of the
                    smoothed spectrum (uphill walk)
  refine Dnu      : linear fit of nu vs radial order for the l=0 guesses;
                    updates the Dnu and eps_p sliders
  stretching      : expand the canvas with the stretched period echelle
                    and its DPi1 / q / d01 sliders (toggles)
  save            : <session>/modes_init.csv plus params_init.csv
                    (dnu, eps_p, dpi1, q, d01, smooth, numax); both are
                    restored on the next launch

Clicks on the stretched period echelle add l=1 modes (that panel exists
for the dipole mixed modes). Zoom/pan with the normal toolbar; clicks
are ignored while a toolbar tool is active.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

from .asymptotic import AttrDict, make_tau
from .echelle import frequency_echelle, period_echelle
from .utils import L_STYLE, smooth, dnu_from_numax

ECHELLE_HALF_WIDTH = 7          # rows above/below numax
SMOOTH_FRAC = 0.005             # smoothing width = this * Dnu

# axis positions [x, y, w, h] for the two layouts
LAYOUT = {
    'compact': {
        'ech':   [0.07, 0.42, 0.72, 0.52], 'psd': [0.07, 0.06, 0.72, 0.16],
        'dnu':   [0.12, 0.325, 0.50, 0.020], 'eps': [0.12, 0.296, 0.50, 0.020],
        'smooth': [0.12, 0.267, 0.50, 0.020],
        'radio': [0.84, 0.68, 0.12, 0.24], 'action': [0.84, 0.61, 0.12, 0.05],
        'refine': [0.84, 0.545, 0.12, 0.05],
        'refine_dnu': [0.84, 0.48, 0.12, 0.05],
        'stretch': [0.84, 0.415, 0.12, 0.05],
        'save':  [0.84, 0.35, 0.12, 0.05],
        'markers': [0.84, 0.285, 0.12, 0.05],
        'tau':   None, 'dpi': None, 'q': None, 'd01': None,
        'size': (12.5, 9),
    },
    'expanded': {
        'ech':   [0.05, 0.42, 0.38, 0.51], 'psd': [0.05, 0.055, 0.86, 0.15],
        'dnu':   [0.08, 0.335, 0.27, 0.020], 'eps': [0.08, 0.307, 0.27, 0.020],
        'smooth': [0.08, 0.279, 0.27, 0.020],
        'radio': [0.92, 0.70, 0.075, 0.24], 'action': [0.925, 0.64, 0.065, 0.05],
        'refine': [0.925, 0.575, 0.065, 0.05],
        'refine_dnu': [0.925, 0.51, 0.065, 0.05],
        'stretch': [0.925, 0.445, 0.065, 0.05],
        'save':  [0.925, 0.38, 0.065, 0.05],
        'markers': [0.925, 0.315, 0.065, 0.05],
        'tau':   [0.50, 0.42, 0.40, 0.51],
        'dpi':   [0.53, 0.33, 0.27, 0.020], 'q': [0.53, 0.30, 0.27, 0.020],
        'd01':   [0.53, 0.27, 0.27, 0.020],
        'size': (19, 9),
    },
}

PARAM_KEYS = ('dnu', 'eps_p', 'dpi1', 'q', 'd01', 'smooth', 'numax')
DEFAULTS = {'eps_p': 1.0, 'dpi1': 100.0, 'q': 0.15, 'd01': 0.0,
            'smooth': SMOOTH_FRAC}


class EchelleWidget:
    def __init__(self, star, nu, snr_raw, numax, init, session_dir):
        self.star = star
        self.nu = np.asarray(nu)
        self.raw = np.asarray(snr_raw)
        self.dnu_bin = float(np.median(np.diff(self.nu)))
        self.dnu_init = init['dnu']
        self.power = smooth(self.raw,
                            init.get('smooth', SMOOTH_FRAC)
                            * init['dnu'] / self.dnu_bin)
        self.numax = numax
        self.modes = []                     # list of dict(l=..., fc=...)
        self.action = 'add'
        self.current_l = 0
        self.stretched = False
        self.show_markers = True
        self.session_dir = session_dir
        self.outfile = os.path.join(session_dir, 'modes_init.csv')
        self.paramfile = os.path.join(session_dir, 'params_init.csv')
        if os.path.exists(self.outfile):
            table = pd.read_csv(self.outfile)
            self.modes = [{'l': int(r.l), 'fc': float(r.fc)}
                          for r in table.itertuples(index=False)]
            print(f'loaded {len(self.modes)} existing guesses '
                  f'from {self.outfile}')

        self.fig = plt.figure(figsize=LAYOUT['compact']['size'])
        pos = LAYOUT['compact']
        self.ax_ech = self.fig.add_axes(pos['ech'])
        self.ax_psd = self.fig.add_axes(pos['psd'])
        self.ax_tau = self.fig.add_axes([0.5, 0.5, 0.1, 0.1])
        self.ax_tau.set_visible(False)

        # sliders (the stretch-panel ones start hidden)
        self.ax_sliders = {}

        def add_slider(key, rect, label, vmin, vmax, vinit, fmt):
            self.ax_sliders[key] = self.fig.add_axes(rect)
            return Slider(self.ax_sliders[key], label, vmin, vmax,
                          valinit=vinit, valfmt=fmt)
        dnu = init['dnu']
        self.s_dnu = add_slider('dnu', pos['dnu'], r'$\Delta\nu$',
                                0.9*dnu, 1.1*dnu, dnu, '%.3f')
        self.s_eps = add_slider('eps', pos['eps'], r'$\varepsilon_p$',
                                0.0, 2.0, init['eps_p'], '%.3f')
        self.s_smooth = add_slider('smooth', pos['smooth'],
                                   r'smooth [$\Delta\nu$]',
                                   0.001, 0.02,
                                   init.get('smooth', SMOOTH_FRAC), '%.4f')
        self.s_smooth.on_changed(self.on_smooth)
        hidden = [0.5, 0.02, 0.2, 0.02]
        self.s_dpi = add_slider('dpi', hidden, r'$\Delta\Pi_1$ [s]',
                                30.0, 360.0, init['dpi1'], '%.2f')
        self.s_q = add_slider('q', hidden, 'q', 0.02, 0.8,
                              init['q'], '%.3f')
        self.s_d01 = add_slider('d01', hidden, r'$d_{01}$', -0.2, 0.2,
                                init['d01'], '%.3f')
        for key in ('dpi', 'q', 'd01'):
            self.ax_sliders[key].set_visible(False)
        for s in (self.s_dnu, self.s_eps, self.s_dpi, self.s_q, self.s_d01):
            s.on_changed(self.redraw)

        # buttons / radio
        self.ax_radio = self.fig.add_axes(pos['radio'])
        self.radio = RadioButtons(self.ax_radio,
                                  (r'$\ell=0$', r'$\ell=1$',
                                   r'$\ell=2$', r'$\ell=3$'))
        self.radio.on_clicked(self.set_l)
        self.radio.set_label_props({'fontsize': [14]*4})
        # NOTE: set_radio_props re-captures the active colors from the
        # CURRENT facecolors (only the selected circle is colored at this
        # point), so the facecolor must be passed along or the l=1..3
        # circles select invisibly afterwards
        self.radio.set_radio_props({'s': [120]*4,
                                    'facecolor': ['tab:blue']*4})
        self.ax_radio.set_title('degree', fontsize=11)
        self.ax_action = self.fig.add_axes(pos['action'])
        self.btn_action = Button(self.ax_action, 'mode: add')
        self.btn_action.on_clicked(self.toggle_action)
        self.ax_refine = self.fig.add_axes(pos['refine'])
        self.btn_refine = Button(self.ax_refine, 'refine frequency')
        self.btn_refine.on_clicked(self.refine)
        self.ax_refine_dnu = self.fig.add_axes(pos['refine_dnu'])
        self.btn_refine_dnu = Button(self.ax_refine_dnu,
                                     r'refine $\Delta\nu$')
        self.btn_refine_dnu.on_clicked(self.refine_dnu)
        self.ax_stretch = self.fig.add_axes(pos['stretch'])
        self.btn_stretch = Button(self.ax_stretch, 'stretching')
        self.btn_stretch.on_clicked(self.toggle_stretch)
        self.ax_save = self.fig.add_axes(pos['save'])
        self.btn_save = Button(self.ax_save, 'save')
        self.btn_save.on_clicked(self.save)
        self.ax_markers = self.fig.add_axes(pos['markers'])
        self.btn_markers = Button(self.ax_markers, 'hide markers')
        self.btn_markers.on_clicked(self.toggle_markers)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.redraw(None)

    # ---- layout ---------------------------------------------------------
    def apply_layout(self, name):
        pos = LAYOUT[name]
        self.ax_ech.set_position(pos['ech'])
        self.ax_psd.set_position(pos['psd'])
        for key in ('dnu', 'eps', 'smooth'):
            self.ax_sliders[key].set_position(pos[key])
        for attr, key in (('ax_radio', 'radio'), ('ax_action', 'action'),
                          ('ax_refine', 'refine'),
                          ('ax_refine_dnu', 'refine_dnu'),
                          ('ax_stretch', 'stretch'), ('ax_save', 'save'),
                          ('ax_markers', 'markers')):
            getattr(self, attr).set_position(pos[key])
        show = pos['tau'] is not None
        self.ax_tau.set_visible(show)
        for key in ('dpi', 'q', 'd01'):
            self.ax_sliders[key].set_visible(show)
            if show:
                self.ax_sliders[key].set_position(pos[key])
        if show:
            self.ax_tau.set_position(pos['tau'])
        self.fig.set_size_inches(*pos['size'], forward=True)

    def toggle_stretch(self, _):
        self.stretched = not self.stretched
        self.apply_layout('expanded' if self.stretched else 'compact')
        self.redraw(None)
        self.fig.canvas.draw()

    def on_smooth(self, _):
        self.power = smooth(self.raw,
                            self.s_smooth.val * self.dnu_init/self.dnu_bin)
        self.redraw(None)

    # ---- model params ---------------------------------------------------
    def params(self):
        return AttrDict(dnu=self.s_dnu.val, dpi1=self.s_dpi.val,
                        q=self.s_q.val, eps_p=self.s_eps.val,
                        d01=self.s_d01.val, numax=self.numax)

    # ---- drawing --------------------------------------------------------
    def redraw(self, _):
        dnu = self.s_dnu.val
        fmin = self.numax - ECHELLE_HALF_WIDTH*dnu
        fmax = self.numax + (ECHELLE_HALF_WIDTH + 1)*dnu
        z, ext = frequency_echelle(self.nu, self.power, dnu,
                                   fmin=fmin, fmax=fmax,
                                   echelle_type='replicated')
        self.extent = ext
        # row-band geometry: the extent maps z.shape[0] stacks LINEARLY
        # onto [ext2, ext3], which is not exactly Dnu per band -- markers
        # and click decoding must go through the stack index, not nu/Dnu
        self.n_rows = z.shape[0]
        self.band = (ext[3] - ext[2]) / self.n_rows
        # radial order of the bottom row: taken from the lowest frequency
        # actually folded (ext[2]), NOT from the requested window start --
        # a spectrum trimmed above fmin starts at a higher order
        self.k0 = int(np.floor(ext[2] / dnu))

        self.ax_ech.clear()
        vmax = np.nanpercentile(z, 99)
        self.ax_ech.imshow(z, aspect='auto', cmap='Greys', vmin=0,
                           vmax=vmax, interpolation='nearest', extent=ext)
        self.ax_ech.axvline(dnu, color='tab:orange', lw=0.8, alpha=0.6)
        # eps_p guide: expected x of the l=0 ridge (and its replica)
        x0 = (self.s_eps.val % 1.0) * dnu
        for xg in (x0, x0 + dnu):
            self.ax_ech.axvline(xg, color='tab:blue', lw=1.0, ls='--',
                                alpha=0.6)
        for m in self.modes if self.show_markers else []:
            x = m['fc'] % dnu
            istack = int(np.floor(m['fc'] / dnu)) - self.k0
            color, marker = L_STYLE[m['l']]
            # y = center of the mode's row band, via the stack index
            if 0 <= istack < self.n_rows:
                self.ax_ech.plot(x, ext[2] + (istack + 0.5)*self.band,
                                 marker, ms=9, mfc='none', mew=2,
                                 color=color)
            if 0 <= istack - 1 < self.n_rows:
                self.ax_ech.plot(x + dnu,
                                 ext[2] + (istack - 0.5)*self.band,
                                 marker, ms=9, mfc='none', mew=2,
                                 color=color)
        self.ax_ech.set_xlim(0, 2*dnu)
        self.ax_ech.set_xlabel(r'$\nu$ mod $\Delta\nu$ [$\mu$Hz]')
        self.ax_ech.set_ylabel(r'frequency [$\mu$Hz]')
        self.ax_ech.set_title(
            f'{self.star}: $\\Delta\\nu$={dnu:.3f} $\\mu$Hz '
            f'({len(self.modes)} modes; click to {self.action})')

        self.ax_psd.clear()
        window = (self.nu > ext[2]) & (self.nu < ext[3])
        self.ax_psd.plot(self.nu[window], self.power[window], 'k-', lw=0.6)
        for m in self.modes if self.show_markers else []:
            self.ax_psd.axvline(m['fc'], color=L_STYLE[m['l']][0], lw=1,
                                alpha=0.7)
        self.ax_psd.set_xlim(ext[2], ext[3])
        self.ax_psd.set_xlabel(r'frequency [$\mu$Hz]')
        self.ax_psd.set_ylabel('smoothed S/B')

        if self.stretched:
            self.redraw_tau(fmin, fmax)
        self.fig.canvas.draw_idle()

    def redraw_tau(self, fmin, fmax):
        p = self.params()
        window = (self.nu > fmin) & (self.nu < fmax)
        nu_w = self.nu[window]
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_w = make_tau(nu_w, p, constant_q=True, constant_eps_p=True,
                             constant_d01=True)
        z, ext = period_echelle(nu_w, self.power[window], p.dpi1,
                                tau=tau_w, echelle_type='replicated')

        self.ax_tau.clear()
        vmax = np.nanpercentile(z, 99)
        self.ax_tau.imshow(z, aspect='auto', cmap='Greys', vmin=0,
                           vmax=vmax, interpolation='nearest', extent=ext)
        self.ax_tau.axvline(p.dpi1, color='tab:orange', lw=0.8, alpha=0.6)
        n_rows = z.shape[0]
        band = (ext[3] - ext[2]) / n_rows
        # stack boundaries exactly as period_echelle finds them: wraps of
        # tau % DPi1 along the array. NOTE tau is not monotonic -- the
        # arctan term jumps by ~DPi1 at every radial-order crossing, and
        # each jump is one more stack, so net-tau counting is wrong.
        wraps = np.diff(tau_w % p.dpi1) > 0
        # keep the fold state so clicks on this panel can be inverted
        # (row + half -> stack -> tau segment -> frequency)
        boundaries = np.concatenate([[0], np.flatnonzero(wraps) + 1,
                                     [len(tau_w)]])
        self.tau_state = {'nu': nu_w, 'tau': tau_w, 'bounds': boundaries,
                          'ext': ext, 'band': band, 'n_rows': n_rows,
                          'dpi': p.dpi1}
        for m in self.modes if self.show_markers else []:
            if not (ext[2] < m['fc'] < ext[3]):
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                tau_m = float(make_tau(np.atleast_1d(m['fc']), p,
                                       constant_q=True, constant_eps_p=True,
                                       constant_d01=True)[0])
            x = tau_m % p.dpi1
            i_mode = int(np.searchsorted(nu_w, m['fc']))
            istack = int(np.sum(wraps[:max(i_mode, 0)]))
            color, marker = L_STYLE[m['l']]
            # make_fold(reverse=True): a row shows [stack i+1 | stack i],
            # so stack s sits in the RIGHT half of row s and in the LEFT
            # half of row s-1 (one row lower -- tau runs against nu)
            if 0 <= istack < n_rows:
                self.ax_tau.plot(x + p.dpi1,
                                 ext[2] + (istack + 0.5)*band, marker,
                                 ms=9, mfc='none', mew=2, color=color)
            if 0 <= istack - 1 < n_rows:
                self.ax_tau.plot(x, ext[2] + (istack - 0.5)*band, marker,
                                 ms=9, mfc='none', mew=2, color=color)
        self.ax_tau.set_xlim(0, 2*p.dpi1)
        self.ax_tau.set_xlabel(r'$\tau$ mod $\Delta\Pi_1$ [s]')
        self.ax_tau.set_title(
            f'stretched period echelle: $\\Delta\\Pi_1$={p.dpi1:.2f} s, '
            f'q={p.q:.3f}, $d_{{01}}$={p.d01:.3f}')

    # ---- interactions ---------------------------------------------------
    def set_l(self, label):
        self.current_l = int(''.join(c for c in label if c.isdigit()))

    def toggle_markers(self, _):
        self.show_markers = not self.show_markers
        self.btn_markers.label.set_text(
            'show markers' if not self.show_markers else 'hide markers')
        self.redraw(None)
        self.fig.canvas.draw()

    def toggle_action(self, _):
        self.action = 'remove' if self.action == 'add' else 'add'
        self.btn_action.label.set_text(f'mode: {self.action}')
        self.redraw(None)

    def on_click(self, event):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar is not None and getattr(toolbar, 'mode', ''):
            return                          # zoom/pan tool active
        if event.inaxes is self.ax_tau and self.stretched:
            self.on_click_tau(event)
            return
        if event.inaxes is not self.ax_ech:
            return
        dnu = self.s_dnu.val
        if not (self.extent[2] < event.ydata < self.extent[3]):
            return
        # decode via the stack index (the y axis is banded, not linear
        # in nu/dnu); x > dnu lands in the replicated half = next order
        istack = int((event.ydata - self.extent[2]) // self.band)
        fc = (self.k0 + istack)*dnu + event.xdata

        if self.action == 'add':
            self.modes.append({'l': self.current_l, 'fc': fc})
            print(f'added l={self.current_l} at {fc:.3f} uHz')
        else:
            if not self.modes:
                return
            dist = [abs(m['fc'] - fc) for m in self.modes]
            i = int(np.argmin(dist))
            if dist[i] < 0.5*dnu:
                removed = self.modes.pop(i)
                print(f'removed l={removed["l"]} at {removed["fc"]:.3f} uHz')
        self.redraw(None)

    def on_click_tau(self, event):
        """Click on the stretched period echelle: invert (x, row, half)
        back to a frequency. Adds are always l=1 (the panel exists for
        the dipole mixed modes); remove deletes the nearest mode."""
        st = getattr(self, 'tau_state', None)
        if st is None:
            return
        ext, band, dpi = st['ext'], st['band'], st['dpi']
        if not (ext[2] < event.ydata < ext[3] and 0 < event.xdata < 2*dpi):
            return
        row = int((event.ydata - ext[2]) // band)
        # replicated with reverse=True: left half of row r = stack r+1,
        # right half = stack r
        s = row + 1 if event.xdata < dpi else row
        if not 0 <= s < len(st['bounds']) - 1:
            return
        lo, hi = st['bounds'][s], st['bounds'][s + 1]
        if hi <= lo:
            return
        seg_tau = st['tau'][lo:hi] % dpi
        i_best = lo + int(np.argmin(np.abs(seg_tau - event.xdata % dpi)))
        fc = float(st['nu'][i_best])

        if self.action == 'add':
            self.modes.append({'l': 1, 'fc': fc})
            print(f'added l=1 at {fc:.3f} uHz (from period echelle)')
        else:
            if not self.modes:
                return
            dist = [abs(m['fc'] - fc) for m in self.modes]
            i = int(np.argmin(dist))
            if dist[i] < 0.5*self.s_dnu.val:
                removed = self.modes.pop(i)
                print(f'removed l={removed["l"]} at '
                      f'{removed["fc"]:.3f} uHz')
        self.redraw(None)

    def refine(self, _):
        """Uphill walk on the smoothed spectrum: each guess climbs to the
        local maximum it already sits on."""
        shifts = []
        for m in self.modes:
            i = int(np.searchsorted(self.nu, m['fc']))
            i = np.clip(i, 1, len(self.nu) - 2)
            while 0 < i < len(self.nu) - 1:
                if self.power[i+1] > self.power[i]:
                    i += 1
                elif self.power[i-1] > self.power[i]:
                    i -= 1
                else:
                    break
            shifts.append(float(self.nu[i]) - m['fc'])
            m['fc'] = float(self.nu[i])
        if shifts:
            print(f'refined {len(shifts)} guesses; |shift| median '
                  f'{np.median(np.abs(shifts)):.4f}, max '
                  f'{np.max(np.abs(shifts)):.4f} uHz')
        self.redraw(None)
        self.fig.canvas.draw()          # force immediate repaint

    def refine_dnu(self, _):
        """Linear fit of nu vs radial order for the identified l=0 modes:
        nu = Dnu*(n + eps_p). Orders are assigned from the current slider
        values, so run this after the ridge is roughly aligned; updates
        both the Dnu and eps_p sliders (which triggers a redraw)."""
        fc0 = np.sort([m['fc'] for m in self.modes if m['l'] == 0])
        if len(fc0) < 3:
            print('refine dnu: need at least 3 l=0 modes '
                  f'(have {len(fc0)})')
            return
        n = np.round(fc0/self.s_dnu.val - self.s_eps.val)
        slope, intercept = np.polyfit(n, fc0, 1)
        eps = intercept/slope
        eps -= np.floor(eps)            # into [0, 1)
        if eps < 0.5:
            eps += 1.0                  # conventional range [0.5, 1.5)
        print(f'refine dnu: {len(fc0)} l=0 modes -> '
              f'Dnu={slope:.4f} uHz, eps_p={eps:.4f} '
              f'(rms of fit {np.std(fc0 - (slope*n + intercept)):.4f} uHz)')
        lo, hi = self.s_dnu.valmin, self.s_dnu.valmax
        if not lo < slope < hi:
            print(f'  fitted Dnu outside slider range [{lo:.2f}, {hi:.2f}]; '
                  'not applied')
            return
        self.s_eps.set_val(np.clip(eps, self.s_eps.valmin,
                                   self.s_eps.valmax))
        self.s_dnu.set_val(slope)       # triggers redraw

    def save(self, _):
        os.makedirs(self.session_dir, exist_ok=True)
        table = pd.DataFrame([{'star': self.star, 'l': m['l'],
                               'fc': round(m['fc'], 4)}
                              for m in sorted(self.modes,
                                              key=lambda m: (m['l'], m['fc']))])
        table.to_csv(self.outfile, index=False)
        pd.DataFrame([{'star': self.star, 'dnu': self.s_dnu.val,
                       'eps_p': self.s_eps.val, 'dpi1': self.s_dpi.val,
                       'q': self.s_q.val, 'd01': self.s_d01.val,
                       'smooth': self.s_smooth.val, 'numax': self.numax}]
                     ).round(4).to_csv(self.paramfile, index=False)
        print(f'saved {len(table)} modes -> {self.outfile}; params -> '
              f'{self.paramfile}')


def initial_values(session_dir, numax=None, **overrides):
    """Start values for dnu/eps_p/dpi1/q/d01/smooth/numax.

    Precedence: package defaults < saved session (params_init.csv in
    session_dir) < explicit keyword overrides (ignored when None). numax
    must come from somewhere: the argument or a saved session. dnu falls
    back to the numax scaling relation.
    """
    init = dict(DEFAULTS, dnu=None, numax=numax)

    paramfile = os.path.join(session_dir, 'params_init.csv')
    if os.path.exists(paramfile):
        saved = pd.read_csv(paramfile).iloc[0]
        for key in PARAM_KEYS:
            if key in saved and np.isfinite(saved[key]):
                init[key] = float(saved[key])
        print(f'restored saved session params from {paramfile}')

    if numax is not None:
        init['numax'] = float(numax)
    for key, value in overrides.items():
        if key not in DEFAULTS and key != 'dnu':
            raise TypeError(f'unknown parameter {key!r}')
        if value is not None:
            init[key] = float(value)

    if init['numax'] is None:
        raise ValueError('numax is required (no saved session to restore '
                         'it from)')
    if init['dnu'] is None:
        init['dnu'] = dnu_from_numax(init['numax'])
        print(f'no dnu given; scaling relation gives {init["dnu"]:.2f} uHz')
    return init


def launch(star, nu, snr, numax=None, workdir='clickelle_out', show=True,
           **overrides):
    """Open the echelle widget for one star.

    Parameters
    ----------
    star : str
        Label for the star; the session lives in <workdir>/<star>/.
    nu, snr : array-like
        Background-normalized power spectrum (S/B) and its frequencies
        [uHz].
    numax : float, optional
        Frequency of maximum power [uHz]; restored from a saved session
        when omitted.
    workdir : str
        Where sessions are stored.
    show : bool
        Call plt.show() (blocks until the window closes).
    **overrides
        dnu, eps_p, dpi1, q, d01, smooth start values.
    """
    session_dir = os.path.join(workdir, str(star))
    init = initial_values(session_dir, numax=numax, **overrides)
    widget = EchelleWidget(str(star), nu, snr, init['numax'], init,
                           session_dir)
    if show:
        plt.show()
    return widget
