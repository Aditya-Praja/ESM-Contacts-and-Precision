import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import esm
except ImportError:
    sys.exit("pip install fair-esm")

from contacts import BANDS, band_counts, contact_map, precision_at_L

CHECKPOINTS = [
    ("esm2_t6_8M_UR50D",      8),
    ("esm2_t12_35M_UR50D",   35),
    ("esm2_t30_150M_UR50D", 150),
    ("esm2_t33_650M_UR50D", 650),
    ("esm2_t36_3B_UR50D",  3000),
]

MIN_LONG_RANGE_LENGTH = 48   


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def load_truth(pdb_dir, manifest):
    truth = {}
    for row in manifest.itertuples():
        cif = pdb_dir / f"{row.entry_id}.cif"
        if not cif.exists():
            print(f"  skip {row.entry_id}: no cif")
            continue
        try:
            true, seq = contact_map(cif, row.chain_id)
        except Exception as exc:                   
            print(f"  skip {row.entry_id}: {exc}")
            continue

        L = len(seq)
        if L < MIN_LONG_RANGE_LENGTH:
            print(f"  skip {row.entry_id}: {L} residues, too short to score long-range")
            continue

        counts = {band: band_counts(true, L, lo, hi)
                  for band, (lo, hi) in BANDS.items()}

        truth[row.entry_id] = {"chain_id": row.chain_id, "seq": seq,
                               "true": true, "counts": counts}
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"))
    ap.add_argument("--out", default="./results/sweep.parquet")
    ap.add_argument("--max-params", type=int, default=None,
                    help="skip checkpoints larger than this, in millions")
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore existing results instead of resuming")
    args = ap.parse_args()

    pdb_dir = Path(args.data_dir) / "pdb"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(pdb_dir / "manifest.csv")
    print(f"manifest   : {len(manifest)} chains")

    records, done = [], set()
    if out_path.exists() and not args.overwrite:
        prev = pd.read_parquet(out_path)
        records = prev.to_dict("records")
        done = set(zip(prev.model, prev.entry_id))
        print(f"resuming   : {len(records)} rows already cached")

    print("\nparsing structures...")
    truth = load_truth(pdb_dir, manifest)
    print(f"{len(truth)} chains with usable contact maps")

    if not truth:
        sys.exit("no usable chains — run build_heldout.py first")

    device = get_device()
    print(f"device     : {device}")

    todo = [c for c in CHECKPOINTS
            if args.max_params is None or c[1] <= args.max_params]
    print(f"checkpoints: {', '.join(n for n, _ in todo)}")

    for name, params_m in todo:
        pending = [k for k in truth if (name, k) not in done]
        if not pending:
            print(f"\n--- {name}: already complete, skipping")
            continue

        print(f"\n--- {name} ({params_m}M), {len(pending)} chains")
        t0 = time.time()

        model, alphabet = esm.pretrained.__dict__[name]()
        model = model.eval().to(device)
        bc = alphabet.get_batch_converter()

        for n, entry_id in enumerate(pending, 1):
            rec = truth[entry_id]
            seq, true = rec["seq"], rec["true"]
            L = len(seq)

            _, _, toks = bc([(entry_id, seq)])
            with torch.no_grad():
                out = model(toks.to(device), return_contacts=True)
            pred = out["contacts"][0].float().cpu().numpy()

            assert pred.shape[0] == L, \
                f"{entry_id}: contacts {pred.shape[0]} vs sequence {L}"

            row = {"model": name, "params_m": params_m, "entry_id": entry_id,
                   "chain_id": rec["chain_id"], "length": L}
            for band, (lo, hi) in BANDS.items():
                n_pairs, n_true = rec["counts"][band]
                row[f"p_at_L_{band}"] = precision_at_L(pred, true, L, lo, hi)
                row[f"n_pairs_{band}"] = n_pairs
                row[f"n_true_{band}"] = n_true
            records.append(row)

            if n % 20 == 0:
                print(f"  {n}/{len(pending)}  ({time.time() - t0:.0f}s)")

        del model
        free(device)

        pd.DataFrame(records).to_parquet(out_path)
        print(f"  done in {time.time() - t0:.0f}s, cached to {out_path}")

    df = pd.DataFrame(records)
    print(f"\n{len(df)} rows, {df.entry_id.nunique()} chains, "
          f"{df.params_m.nunique()} model sizes\n")
    cols = [f"p_at_L_{b}" for b in BANDS]
    print(df.groupby(["params_m", "model"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()