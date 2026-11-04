#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import esm
except ImportError:
    sys.exit("pip install fair-esm")

from contacts import contact_map, precision_at_L

BANDS = {"short": (6, 12), "medium": (12, 24), "long": (24, None)}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def band_precision(pred, true, L, lo, hi):
    i, j = np.triu_indices(L, k=lo)
    if hi is not None:
        keep = (j - i) < hi
        i, j = i[keep], j[keep]
    if len(i) == 0:
        return float("nan"), 0, 0
    k = min(L, len(i))
    order = np.argsort(-pred[i, j])[:k]
    return float(true[i, j][order].mean()), len(i), int(true[i, j].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    ap.add_argument("--entry", default=None, help="PDB id; default is the first in the manifest")
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    args = ap.parse_args()

    pdb_dir = Path(args.data_dir) / "pdb"
    manifest = pd.read_csv(pdb_dir / "manifest.csv")

    row = (manifest[manifest.entry_id == args.entry].iloc[0] if args.entry
           else manifest.iloc[0])
    cif = pdb_dir / f"{row.entry_id}.cif"

    print(f"chain      : {row.entry_id} chain {row.chain_id}")

    true, seq = contact_map(cif, row.chain_id)
    L = len(seq)
    print(f"length     : {L} residues")
    print(f"sequence   : {seq[:60]}{'...' if L > 60 else ''}")

    if L != true.shape[0]:
        sys.exit(f"FAIL  seq is {L} but map is {true.shape[0]} — contact_map is broken")

    i, j = np.triu_indices(L, k=24)
    n_long = int(true[i, j].sum())
    print(f"long-range contacts: {n_long}  ({n_long / L:.2f} per residue)")
    if n_long < 0.5 * L:
        print("  note: fewer than L/2 long-range contacts — the top-L ceiling will bite")

    device = get_device()
    print(f"\ndevice     : {device}")
    print(f"model      : {args.model}")

    model, alphabet = esm.pretrained.__dict__[args.model]()
    model = model.eval().to(device)

    _, _, toks = alphabet.get_batch_converter()([(row.entry_id, seq)])
    print(f"tokens     : {toks.shape[1]}  (= {L} residues + BOS + EOS)")

    with torch.no_grad():
        out = model(toks.to(device), return_contacts=True)
    pred = out["contacts"][0].float().cpu().numpy()

    print(f"contacts   : {pred.shape}  (the head already stripped BOS/EOS)")
    if pred.shape[0] != L:
        sys.exit(f"FAIL  model returned {pred.shape[0]}x{pred.shape[0]} for a {L}-residue chain")

    print()
    rng = np.random.default_rng(0)
    chance = np.mean([precision_at_L(rng.random((L, L)), true, L) for _ in range(50)])

    print(f"{'band':<8}{'P@L':>8}{'pairs':>9}{'true':>8}")
    for name, (lo, hi) in BANDS.items():
        p, n_pairs, n_true = band_precision(pred, true, L, lo, hi)
        print(f"{name:<8}{p:>8.3f}{n_pairs:>9}{n_true:>8}")
    print(f"{'random':<8}{chance:>8.3f}   (long-range chance level)")

    order = np.argsort(-pred[i, j])[:5]
    print("\ntop 5 long-range predictions:")
    for rank, idx in enumerate(order, 1):
        a, b = int(i[idx]), int(j[idx])
        hit = "contact" if true[a, b] else "no"
        print(f"  {rank}.  {seq[a]}{a:<4} - {seq[b]}{b:<4} "
              f"p={pred[a, b]:.3f}  sep={b - a:<4} {hit}")

    p_long, _, _ = band_precision(pred, true, L, 24, None)
    p_short, _, _ = band_precision(pred, true, L, 6, 12)
    print()
    if p_long < chance * 3:
        print("FAIL  long-range is near chance. Sequence and contact map probably")
        print("      describe different residues. Check the chain and the glycine branch.")
    elif p_short < 0.4:
        print("WARN  short-range is low. Even a small model should find local structure;")
        print("      suspect the ground truth rather than the model.")
    else:
        print("OK    numbers are sane. Wrap this in two loops and you have run_sweep.py.")


if __name__ == "__main__":
    main()