#! /usr/bin/env python

# /// script
# dependencies = [
#   "aplpy",
#   "astropy<6.0",
#   "matplotlib",
#   "nmmn @ git+https://github.com/rsnemmen/nmmn",
#   "numpy",
#   "scipy",
#   "uncertainties",
# ]
# ///

"""
Simple script for spectrum plotting.
"""

import nmmn.sed
import argparse
from pathlib import Path
from pylab import plot
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument("--spectrum_path", type=Path, required=True)
    parser.add_argument("--plot_path", type=Path, required=True)

    args = parser.parse_args()

    if not args.spectrum_path.exists():
        print(f"File {args.spectrum_path} does not exist")
        exit(1)

    if not args.plot_path.parent.exists():
        print(f"Parent directory of {args.plot_path} does not exist")
        exit(1)

    s = nmmn.sed.SED()
    s.grmonty(str(args.spectrum_path.resolve()))

    plt.figure()
    plot(s.lognu, s.ll)
    plt.savefig(str(args.plot_path.resolve()), dpi=600, bbox_inches="tight")

    print(f"Spectrum plot saved to {args.plot_path}")
