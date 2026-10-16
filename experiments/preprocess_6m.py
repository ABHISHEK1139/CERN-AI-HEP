import os
from glob import glob

import awkward as ak
import numpy as np
import torch
import uproot

PARTICLE_FEATURES = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi", "part_d0val", "part_d0err",
    "part_dzval", "part_dzerr", "part_charge",
    "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
    "part_isElectron", "part_isMuon"
]

LABEL_BRANCHES = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q", "label_Hqql",
    "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl"
]

def preprocess_files(root_files, output_dir, max_particles=128, chunk_size=50000):
    os.makedirs(output_dir, exist_ok=True)

    chunk_idx = 0

    for fpath in root_files:
        print(f"Processing {fpath}...")
        try:
            with uproot.open(fpath) as file:
                tree_key = next((k for k in file.keys() if k.split(";")[0].lower() == "tree"), None)
                if not tree_key:
                    print(f"  Skipping {fpath}: no 'tree'.")
                    continue
                tree = file[tree_key]

                for arrays in tree.iterate(PARTICLE_FEATURES + LABEL_BRANCHES, step_size=chunk_size):
                    out_file = os.path.join(output_dir, f"chunk_{chunk_idx}.pt")
                    if os.path.exists(out_file):
                        print(f"  Skipping {out_file} (already exists)")
                        chunk_idx += 1
                        continue
                    # Trim to max_particles by highest pT (pT = hypot(px, py);
                    # original order kept among survivors), matching
                    # JetClassDataset — a first-N clip would keep different
                    # constituents for the same jet on this path.

                    n_jets = len(arrays)

                    pt = ak.hypot(arrays["part_px"], arrays["part_py"])
                    order = ak.argsort(pt, axis=1, ascending=False)
                    keep = ak.sort(order[:, :max_particles], axis=1)
                    feats = {feat: arrays[feat][keep] for feat in PARTICLE_FEATURES}
                    lengths = ak.to_numpy(ak.num(feats["part_px"]))

                    padded_feats = []
                    for feat in PARTICLE_FEATURES:
                        padded = ak.fill_none(ak.pad_none(feats[feat], max_particles, clip=True), 0.0)
                        padded_feats.append(ak.to_numpy(padded).astype(np.float32))

                    node_feats_all = np.stack(padded_feats, axis=-1)

                    is_qcd = ak.to_numpy(arrays["label_QCD"]).astype(bool)
                    binary_labels = (~is_qcd).astype(np.int64)

                    # We will save the dense array, the exact lengths, and the labels
                    chunk_data = {
                        'x': torch.from_numpy(node_feats_all),
                        'lengths': torch.from_numpy(lengths.astype(np.int64)),
                        'y': torch.from_numpy(binary_labels)
                    }

                    torch.save(chunk_data, out_file)
                    print(f"  Saved {out_file} ({n_jets} jets)")
                    chunk_idx += 1

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

if __name__ == "__main__":
    bg_files = sorted(glob("data/jetclass/ZJetsToNuNu_*.root"))
    out_dir = "data/jetclass/processed_chunks"

    print(f"Found {len(bg_files)} ROOT files. Starting preprocessing...")
    preprocess_files(bg_files, out_dir, max_particles=128, chunk_size=50000)
    print("Preprocessing complete!")
