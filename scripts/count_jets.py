"""Count JetClass jets across data/jetclass/ (recursive: covers val_5M/ layout)."""
import glob

import uproot


def _count(pattern):
    total = 0
    for f in glob.glob(pattern, recursive=True):
        try:
            with uproot.open(f) as file:
                tree_key = next((k for k in file.keys() if "tree" in k.lower()), None)
                if tree_key is None:
                    print(f"  Skipping {f}: no 'tree'.")
                    continue
                total += file[tree_key].num_entries
        except Exception as e:
            print(f"  Skipping {f}: {e}")
    return total


def main():
    bg_files = glob.glob("data/jetclass/**/ZJetsToNuNu_*.root", recursive=True)
    sig_files = glob.glob("data/jetclass/**/HTo*.root", recursive=True)
    if not bg_files and not sig_files:
        print("No JetClass ROOT files found under data/jetclass/.")
        return

    bg_count = _count("data/jetclass/**/ZJetsToNuNu_*.root")
    sig_count = _count("data/jetclass/**/HTo*.root")

    print(f"Background (ZJets): {bg_count:,}")
    print(f"Signal (Higgs): {sig_count:,}")
    print(f"Total Jets: {bg_count + sig_count:,}")


if __name__ == "__main__":
    main()
