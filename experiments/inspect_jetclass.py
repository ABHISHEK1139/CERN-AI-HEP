import sys
from pathlib import Path

try:
    import uproot
except ImportError:
    sys.exit("uproot not installed. Run: pip install -r requirements.txt")

path = "data/jetclass/val_5M/ZJetsToNuNu_120.root"
if not Path(path).exists():
    sys.exit(f"Missing {path}. Download JetClass data first (see README / scripts/).")
with uproot.open(path) as file:
    print("Keys:", file.keys())
    tree = file["tree"]
    print(f"\nTree has {tree.num_entries} events.")
    print("Branches:")
    for k in tree.keys():
        print(" -", k)
