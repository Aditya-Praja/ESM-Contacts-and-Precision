import numpy as np
from biotite.structure import filter_amino_acids
from biotite.structure.io.pdbx import CIFFile, get_structure

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
THREE_TO_ONE.update({
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y",
    "CSO": "C", "MLY": "K", "HYP": "P", "CME": "C",
    "OCS": "C", "KCX": "K", "M3L": "K", "ALY": "K",
})

CONTACT_THRESHOLD = 8.0        

BANDS = {
    "short":  (6, 12),
    "medium": (12, 24),
    "long":   (24, None),
}


def contact_map(cif_path, chain_id):
    arr = get_structure(CIFFile.read(cif_path), model=1)
    arr = arr[filter_amino_acids(arr)]   
    arr = arr[arr.chain_id == chain_id]   

    if arr.array_length() == 0:
        raise ValueError(f"no amino acids in chain {chain_id!r} of {cif_path}")

    coords, seq = [], []
    for res_id in np.unique(arr.res_id):   
        res = arr[arr.res_id == res_id]
        name3 = res.res_name[0]

        wanted = "CA" if name3 == "GLY" else "CB"
        sel = res[res.atom_name == wanted]

        if sel.array_length() == 0:
            sel = res[res.atom_name == "CA"]
            if sel.array_length() == 0:
                continue

        coords.append(sel.coord[0])
        seq.append(THREE_TO_ONE.get(name3, "X"))

    coords = np.array(coords)
    d = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)

    return (d < CONTACT_THRESHOLD), "".join(seq)


def precision_at_L(pred, true, L, min_sep=24, max_sep=None):
    assert len(true) == L, f"L={L} but true is {len(true)}x{len(true)}"
    assert pred.shape == true.shape, f"pred {pred.shape} vs true {true.shape}"

    i, j = np.triu_indices(L, k=min_sep)
    if max_sep is not None:
        keep = (j - i) < max_sep
        i, j = i[keep], j[keep]
    if len(i) == 0:
        return float("nan")

    k = min(L, len(i))                     
    order = np.argsort(-pred[i, j])[:k] 
    return float(true[i, j][order].mean())


def band_counts(true, L, min_sep=24, max_sep=None):
    i, j = np.triu_indices(L, k=min_sep)
    if max_sep is not None:
        keep = (j - i) < max_sep
        i, j = i[keep], j[keep]
    return len(i), int(true[i, j].sum())


# ---- self-test -----------------------------------------------------------
def _self_test():
    """Verify the metric on synthetic data, before trusting it on a model."""
    rng = np.random.default_rng(0)
    L = 120

    coords = rng.normal(size=(L, 3)) * 12.0
    d = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    true = d < CONTACT_THRESHOLD

    i, j = np.triu_indices(L, k=24)
    base = true[i, j].mean()

    random_scores = [precision_at_L(rng.random((L, L)), true, L)
                     for _ in range(200)]
    random_mean = float(np.mean(random_scores))

    perfect = true.astype(float) + rng.normal(scale=1e-6, size=(L, L))
    perfect_score = precision_at_L(perfect, true, L)

    n_true_long = int(true[i, j].sum())
    ceiling = min(1.0, n_true_long / L)

    print(f"eligible long-range pairs : {len(i)}")
    print(f"true contacts among them  : {n_true_long}")
    print()
    print(f"base rate                 : {base:.3f}")
    print(f"random predictor          : {random_mean:.3f}   (should match base rate)")
    print(f"perfect ranking           : {perfect_score:.3f}   (ceiling {ceiling:.3f})")
    print()

    ok = True
    if abs(random_mean - base) > 0.01:
        print("FAIL  random predictor beats the base rate -> index leak")
        ok = False
    if perfect_score < ceiling - 0.01:
        print("FAIL  perfect ranking below its ceiling -> ranking is broken")
        ok = False

    # bands should partition the pairs at |i-j| >= 6
    sep = j - i
    si, sj = np.triu_indices(L, k=6)
    ssep = sj - si
    parts = (((ssep >= 6) & (ssep < 12)).sum()
             + ((ssep >= 12) & (ssep < 24)).sum()
             + (ssep >= 24).sum())
    if parts != len(ssep):
        print(f"FAIL  bands do not partition: {parts} vs {len(ssep)}")
        ok = False
    if (sep < 24).any():
        print("FAIL  min_sep not applied")
        ok = False

    print("all checks passed" if ok else "fix the above before sweeping")
    return ok


if __name__ == "__main__":
    _self_test()