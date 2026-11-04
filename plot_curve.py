import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("pip install matplotlib")

from contacts import BANDS

STYLE = {
    "short":  ("#a8b0ba", "^", ":",  1.3, 4.5),
    "medium": ("#9f7aea", "s", "--", 1.5, 5.0),
    "long":   ("#1f5fa8", "o", "-",  2.2, 6.5),
}


def band_label(name, lo, hi):
    span = f"|i-j| >= {lo}" if hi is None else f"{lo} <= |i-j| < {hi}"
    return f"{name.capitalize():<7}({span})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./results/sweep.parquet")
    ap.add_argument("--out", default="./results/scaling_curve.png")
    ap.add_argument("--min-length", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        sys.exit(f"no cache at {cache} — run run_sweep.py first")

    df = pd.read_parquet(cache)
    if args.min_length:
        before = df.entry_id.nunique()
        df = df[df.length >= args.min_length]
        print(f"length filter: {before} -> {df.entry_id.nunique()} chains")

    n_chains = df.entry_id.nunique()
    print(f"{n_chains} chains, {df.params_m.nunique()} model sizes")

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=150)

    for name in STYLE:                       # short, medium, long: long on top
        lo, hi = BANDS[name]
        color, marker, ls, lw, ms = STYLE[name]
        g = df.groupby("params_m")[f"p_at_L_{name}"]

        mean = g.mean()
        # Standard error of the mean, not the standard deviation: the claim
        # is about means, so these bars say whether two sizes really differ.
        sem = g.std(ddof=1) / np.sqrt(g.count())

        ax.errorbar(mean.index, mean.values, yerr=sem.values,
                    color=color, marker=marker, linestyle=ls,
                    linewidth=lw, markersize=ms, capsize=3,
                    label=band_label(name, lo, hi),
                    zorder=3 if name == "long" else 2)

    ax.set_xscale("log")
    ax.set_xticks(sorted(df.params_m.unique()))
    # Override the default log formatter so ticks read "150" not "10^2".
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, 1)                        # fixed, so runs are comparable
    ax.set_xlabel("ESM-2 parameters (millions, log scale)")
    ax.set_ylabel("contact precision@L")
    ax.set_title(f"Contact prediction vs model scale, {n_chains} held-out chains",
                 fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")

    print("\nmean +/- SEM by model size (paste into the README table):\n")
    for name in ("short", "medium", "long"):
        g = df.groupby("params_m")[f"p_at_L_{name}"]
        mean, sem = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
        cells = "  ".join(f"{p:>5.0f}M {m:.3f}+/-{s:.3f}"
                          for p, m, s in zip(mean.index, mean.values, sem.values))
        print(f"  {name:<7} {cells}")

    # ---- last automated sanity check ------------------------------------
    lo_p, hi_p = df.params_m.min(), df.params_m.max()
    gain = (df[df.params_m == hi_p].p_at_L_long.mean()
            - df[df.params_m == lo_p].p_at_L_long.mean())
    print(f"\nlong-range gain, {lo_p}M -> {hi_p}M: {gain:+.3f}")

    if gain < 0.1:
        print("  Smaller than expected. Check index alignment before concluding")
        print("  anything — see the troubleshooting table.")

    # Each band's headroom. Precision@L cannot exceed n_true / L, and the
    # short band is sparse by construction: an alpha-helix makes contacts at
    # separation 3-4 and none at 6-11, so that band is fed only by hairpin
    # turns and loops. Expect its ceiling near 0.2-0.3, well under 1.
    print("\nceilings (mean n_true / L — the most precision@L could reach):")
    for name in ("short", "medium", "long"):
        ceil = (df[f"n_true_{name}"] / df.length).clip(upper=1.0).mean()
        got = df[df.params_m == hi_p][f"p_at_L_{name}"].mean()
        print(f"  {name:<7} ceiling {ceil:.2f}   best model reached {got:.3f} "
              f"({got / ceil:.0%} of it)")


if __name__ == "__main__":
    main()