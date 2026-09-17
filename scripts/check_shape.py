"""Print the structure of JetClass particle branches (no heavy data needed beyond one file)."""
import sys
from pathlib import Path

import uproot


def main(path="data/jetclass/val_5M/ZJetsToNuNu_120.root"):
    p = Path(path)
    if not p.exists():
        sys.exit(f"Missing {p}. Download JetClass data first (see README / scripts/).")
    with uproot.open(p) as file:
        tree_key = next((k for k in file.keys() if "tree" in k.lower()), None)
        if tree_key is None:
            sys.exit(f"No 'tree' in {p}; keys: {list(file.keys())}")
        tree = file[tree_key]

    array = tree["part_px"].array()
    print(f"Type of part_px: {array.type}")
    print(f"Tree entries: {tree.num_entries}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect JetClass branch structure")
    parser.add_argument("--path", default="data/jetclass/val_5M/ZJetsToNuNu_120.root")
    args = parser.parse_args()
    main(args.path)
