import os
import csv
import sys
import time
from pathlib import Path

try:
    import requests
    from rcsbapi.search import search_attributes as attrs
    from rcsbapi.search import GroupBy, RankingCriteriaType
except ImportError:
    sys.exit("pip install rcsb-api requests biotite")

RELEASED_AFTER = "2021-06-01"   # after ESM-2's training cutoff
MAX_RESOLUTION = 2.5
MIN_LEN, MAX_LEN = 50, 400
IDENTITY_CUTOFF = 30            # one of 100, 95, 90, 70, 50, 30
WANT = 80                       # final count; guide says 60-100

OUT = Path(os.environ.get("DATA_DIR", "./data")) / "pdb"
OUT.mkdir(parents=True, exist_ok=True)

query = (
    (attrs.rcsb_accession_info.initial_release_date >= RELEASED_AFTER)
    & (attrs.exptl.method == "X-RAY DIFFRACTION")
    & (attrs.rcsb_entry_info.resolution_combined <= MAX_RESOLUTION)
    & (attrs.rcsb_entry_info.polymer_entity_count_protein == 1)
    & (attrs.entity_poly.rcsb_sample_sequence_length >= MIN_LEN)
    & (attrs.entity_poly.rcsb_sample_sequence_length <= MAX_LEN)
)

print(f"querying RCSB: released >= {RELEASED_AFTER}, <= {MAX_RESOLUTION} A, "
      f"{MIN_LEN}-{MAX_LEN} aa, clustered at {IDENTITY_CUTOFF}% identity")

results = query(
    return_type="polymer_entity",          
    group_by=GroupBy(
        aggregation_method="sequence_identity",
        similarity_cutoff=IDENTITY_CUTOFF,
        ranking_criteria_type=RankingCriteriaType(
            sort_by="rcsb_entry_info.resolution_combined",
            direction="asc",     
        ),
    ),
    group_by_return_type="representatives",
)

entity_ids = [str(hit) for hit in results]
print(f"  {len(entity_ids)} cluster representatives")

if len(entity_ids) < WANT:
    print(f"  fewer than {WANT} available — using all of them")
entity_ids = entity_ids[:WANT]

import biotite.structure as struc                    
from biotite.structure.io.pdbx import CIFFile, get_structure  

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

rows, fasta = [], []

for n, ent in enumerate(entity_ids, 1):
    entry_id = ent.split("_")[0]
    entity_id = ent.split("_")[1] if "_" in ent else "1"
    cif = OUT / f"{entry_id}.cif"

    if not cif.exists():
        try:
            r = requests.get(f"https://files.rcsb.org/download/{entry_id}.cif",
                             timeout=60)
            r.raise_for_status()
            cif.write_bytes(r.content)
        except Exception as exc:                        
            print(f"  skip {entry_id}: {exc}")
            continue
        time.sleep(0.1)                                

    try:
        arr = get_structure(CIFFile.read(cif), model=1)
        arr = arr[struc.filter_amino_acids(arr)]
        if arr.array_length() == 0:
            print(f"  skip {entry_id}: no amino acids parsed")
            continue
        chain = sorted(set(arr.chain_id))[0]        
        arr = arr[arr.chain_id == chain]

        seq = "".join(
            THREE_TO_ONE.get(arr[arr.res_id == rid].res_name[0], "X")
            for rid in sorted(set(arr.res_id))
        )
    except Exception as exc:                           
        print(f"  skip {entry_id}: parse failed ({exc})")
        continue

    if not (MIN_LEN <= len(seq) <= MAX_LEN):
       
        print(f"  skip {entry_id}: {len(seq)} resolved residues, out of range")
        continue

    rows.append({"entry_id": entry_id, "entity_id": entity_id,
                 "chain_id": chain, "length": len(seq)})
    fasta.append(f">{entry_id}_{chain}\n{seq}")

    if n % 20 == 0:
        print(f"  {n}/{len(entity_ids)}")


with open(OUT / "manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["entry_id", "entity_id", "chain_id", "length"])
    w.writeheader()
    w.writerows(rows)

(OUT / "sequences.fasta").write_text("\n".join(fasta) + "\n")

lengths = [r["length"] for r in rows]
print(f"\n{len(rows)} chains in {OUT}")
if lengths:
    lengths.sort()
    print(f"  length: min {lengths[0]}, median {lengths[len(lengths)//2]}, "
          f"max {lengths[-1]}")
print("  manifest.csv and sequences.fasta written")
print("\nThe chain_id column matters: your contact-map code must parse the")
print("SAME chain this script measured, or indices will not line up.")