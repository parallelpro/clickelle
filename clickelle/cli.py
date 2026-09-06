"""Command-line interface.

    clickelle init SPECTRUM --star NAME --numax 500     # open the widget
    clickelle fit  SPECTRUM --star NAME                 # fit the guesses

SPECTRUM is a CSV or parquet table holding a BACKGROUND-NORMALIZED power
spectrum (S/B): frequency in uHz plus the normalized power (columns
auto-detected, or use --freq-col/--power-col). Each star's session lives
in <workdir>/<star>/.
"""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        prog='clickelle', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def add_common(p):
        p.add_argument('spectrum',
                       help='CSV/parquet table with the S/B spectrum')
        p.add_argument('--star', required=True,
                       help='star label; session dir is <workdir>/<star>')
        p.add_argument('--workdir', default='clickelle_out',
                       help='where sessions are stored '
                            '(default: %(default)s)')
        p.add_argument('--freq-col', default=None,
                       help='frequency column name (default: auto)')
        p.add_argument('--power-col', default=None,
                       help='power column name (default: auto)')

    p_init = sub.add_parser('init', help='interactive echelle widget for '
                                         'mode initial guesses')
    add_common(p_init)
    p_init.add_argument('--numax', type=float, default=None,
                        help='frequency of maximum power [uHz]; required '
                             'unless restored from a saved session')
    p_init.add_argument('--dnu', type=float, default=None,
                        help='start value for the large separation [uHz] '
                             '(default: saved session, else numax scaling)')
    p_init.add_argument('--eps-p', type=float, default=None)
    p_init.add_argument('--dpi1', type=float, default=None,
                        help='start value for the period spacing [s]')
    p_init.add_argument('--q', type=float, default=None,
                        help='start value for the coupling factor')
    p_init.add_argument('--d01', type=float, default=None)

    p_fit = sub.add_parser('fit', help='Lorentzian MLE fits of the saved '
                                       'initial guesses')
    add_common(p_fit)
    p_fit.add_argument('--dnu', type=float, default=None,
                       help='large separation [uHz] (default: the value '
                            'saved by the widget)')
    p_fit.add_argument('--numax', type=float, default=None,
                       help='numax [uHz], only used for a scaling-relation '
                            'dnu when neither --dnu nor a session value '
                            'exists')
    p_fit.add_argument('--gap-frac', type=float, default=0.25,
                       help='group break threshold as a fraction of dnu '
                            '(default: %(default)s)')
    p_fit.add_argument('--steps', type=int, default=2000,
                       help='adam steps per group fit (default: '
                            '%(default)s)')
    p_fit.add_argument('--no-quality', action='store_true',
                       help='skip the drop-one lnK refits (much faster)')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    from .utils import load_spectrum
    nu, snr = load_spectrum(args.spectrum, freq_col=args.freq_col,
                            power_col=args.power_col)

    if args.command == 'init':
        from .widget import launch
        launch(args.star, nu, snr, numax=args.numax, workdir=args.workdir,
               dnu=args.dnu, eps_p=args.eps_p, dpi1=args.dpi1, q=args.q,
               d01=args.d01)
    else:
        import matplotlib
        matplotlib.use('Agg')           # headless: outputs are files
        from .fit import fit_star
        fit_star(args.star, nu, snr, workdir=args.workdir, dnu=args.dnu,
                 numax=args.numax, gap_frac=args.gap_frac,
                 steps=args.steps, quality=not args.no_quality)


if __name__ == '__main__':
    main()
